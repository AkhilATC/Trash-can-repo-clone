### Multi-head attention runs h small attentions side by side, so each head can look for a different kind of relationship.

```text
X (B,T,d_model) -> W_q, W_k, W_v -> (B,T,d_model)
                -> split heads   -> (B,h,T,dk)        dk = d_model / h
                -> scaled_dot_product_attention per head -> (B,h,T,dk)
                -> join heads    -> (B,T,d_model)
                -> W_o           -> (B,T,d_model)
```
Our numbers: d_model = 16, h = 4, so dk = 4.

Problem
```text
One head produces ONE softmax row per token: one set of weights that has to
do everything at once.

But a token may need several things at the same time, e.g. "what is the weather today":
  - "weather" wants to know WHEN   -> look at "today"
  - "weather" wants the QUESTION   -> look at "what"
One softmax row has to split its 100% across both, so each signal gets blurred.
```

Solution: several heads, each with its own W_q, W_k, W_v
```text
Each head has its own small projections (16 -> 4), so it learns its own idea of
"what am I looking for" and gets its own softmax row.

Real output, "Howdy Aki", row 0 ([CLS] query), random init:
  head 0: [0.000, 0.001, 0.999, 0, 0, ...]   <- almost all on position 2 ([UNK] = "Aki")
  head 1: [0.768, 0.232, 0.000, 0, 0, ...]   <- mostly on [CLS] itself
Same sentence, same token, two different attention patterns.
(Numbers change every run because the weights start random.)
```

Step 1: project with ONE big Linear per Q/K/V
```text
The simple way: 4 heads x 3 Linears(16 -> 4) = 12 small Linears.
The usual way: 3 Linears(16 -> 16). Rows 0-3 of the weight are head 0, rows 4-7 head 1, ...
It is the same weights stacked into one matrix, but one big matmul is faster than 12 small ones.

  q = self.W_q(X)          (B, T, 16)
```

Step 2: split heads   (B, T, d_model) -> (B, h, T, dk)
```text
view(B, T, h, dk) cuts the 16 numbers of each token into 4 slices of 4.
Real output for one token holding [0, 1, ..., 15]:

  head 0: [ 0,  1,  2,  3]
  head 1: [ 4,  5,  6,  7]
  head 2: [ 8,  9, 10, 11]
  head 3: [12, 13, 14, 15]

transpose(1, 2): (B, T, h, dk) -> (B, h, T, dk)
Why? Matmul only works on the LAST TWO dims and treats everything before them as a batch.
With (B, h, T, dk), the last two are (T, dk), so q @ kᵀ gives (B, h, T, T): all heads at once,
no loop needed.
```

Step 3: attention per head (the same function as before)
```text
scaled_dot_product_attention(q, k, v, mask) works unchanged:
  dk = q.size(-1) = 4  -> divides by √4 = 2 (not √16 = 4)
  scores  (B, h, T, T)
  weights (B, h, T, T)      one 12x12 attention map per head
  out     (B, h, T, dk)
```

The mask needs one more dim
```text
scores is now 4-D: (B, h, T, T). The mask must be (B, 1, 1, T):
  dim 1 = 1 -> same mask for every head
  dim 2 = 1 -> same mask for every query row
  dim 3 = T -> which KEY columns are PAD

MultiHeadAttention passes mask.unsqueeze(1)          (B, T) -> (B, 1, T)
and the function does its own unsqueeze(1)           (B, 1, T) -> (B, 1, 1, T)

Trap: passing the plain (B, T) mask gives (B, 1, T), and broadcasting lines up
dims from the RIGHT, so it acts as (1, B, 1, T): the batch dim lands on the HEAD dim.
With B = 1 this happens to work and hides the bug; with B = 2 and h = 4 it crashes.
Always test with more than one sentence.
```

Step 4: join heads   (B, h, T, dk) -> (B, T, d_model)
```text
transpose(1, 2)   -> (B, T, h, dk)
contiguous()      -> transpose only changes how memory is read; view needs it laid out in order
view(B, T, 16)    -> the 4 head outputs are put side by side again: [head0 | head1 | head2 | head3]

join_heads(split_heads(x)) == x   -> checked: True
```

Step 5: W_o mixes the heads
```text
After joining, numbers 0-3 came ONLY from head 0, numbers 4-7 ONLY from head 1, etc.
The heads never talked to each other.
W_o (16 -> 16) lets every output number use information from all 4 heads.
(This is also why b_v was safe to drop: W_o can learn any constant shift.)
```

Shapes, real run ("Howdy Aki")
```text
Q SHAPE (B,h,T,dk)  torch.Size([1, 4, 12, 4])
WEIGHTS SHAPE       torch.Size([1, 4, 12, 12])
HEADS OUT SHAPE     torch.Size([1, 4, 12, 4])
OUT SHAPE           torch.Size([1, 12, 16])     <- same shape as X, so layers can be stacked
```

Cost
```text
Parameters (bias=False), measured:
  SelfAttention(16)          768  = 3 x 16 x 16          (W_q, W_k, W_v)
  MultiHeadAttention(16, 4)  1024 = 4 x 16 x 16          (W_q, W_k, W_v, W_o)
Splitting into heads costs nothing extra. The only new weights are W_o.
```

Checked against PyTorch
```text
nn.MultiheadAttention(16, 4, bias=False, batch_first=True), with our weights copied in:
  in_proj_weight = cat([W_q, W_k, W_v])     out_proj.weight = W_o
  key_padding_mask = (mask == 0)            (PyTorch: True = IGNORE, the opposite of ours)

  "Howdy Aki"                                   -> matches: True
  ["Howdy Aki", "what is the weather today"]    -> matches: True, out (2, 12, 16), weights (2, 4, 12, 12)
```

Try it yourself
```text
1. Predict shapes for num_heads = 2: dk, q after split_heads, weights, out. Run and check.
2. Try MultiHeadAttention(16, 3). What happens, and why?
3. Make x = torch.arange(16.).view(1, 1, 16) and print mha.split_heads(x).
   Which numbers land in head 2?
4. Remove .contiguous() from join_heads and run. Read the error. What does it tell you?
5. Run with ["Howdy Aki", "what is the weather today"]. For sentence 1, row 0,
   how many non-zero weights does each head have? Predict it from the mask first.
6. In MultiHeadAttention.forward, pass `mask` instead of `mask.unsqueeze(1)`.
   Run with 1 sentence, then with 2. Explain what you see (see "The mask needs one more dim").
7. Think: with h = 16 heads, dk = 1. What does each head's q·k become? Is that useful?
```
