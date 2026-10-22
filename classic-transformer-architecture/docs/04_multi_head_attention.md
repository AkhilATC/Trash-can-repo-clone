# Step 4: Multi-Head Attention

File: `attention.py` (class `MultiHeadAttention`)
Run: `.venv/bin/python attention.py` (step 3 demo: `.venv/bin/python attention.py 3`)

## 1. The idea in one sentence

Instead of one attention head, run **h smaller heads in parallel**. Each head
gets its own Q/K/V, so each one can learn a **different kind of
relationship**. Then glue their outputs back together.

Why? One head produces **one** softmax row per token, so it has to put all of
its attention in one place. A word often needs several things at once. In
"i love my family", *love* might need:

- **who** is doing it → *i*
- **what** is loved → *family*
- the overall sentence summary → *[CLS]*

With 4 heads, *love* can look at all of those at the same time.

## 2. Same size, split into pieces

We do **not** make the model bigger. We cut `d_model` into `h` slices:

```
d_model = 16,  h = 4   →   d_k = d_model / h = 4 numbers per head
```

From our run, here is how one token's 16 numbers are shared out:

```
one token's 16 numbers: [0, 1, 2, ..., 15]
  head 0 gets: [0, 1, 2, 3]
  head 1 gets: [4, 5, 6, 7]
  head 2 gets: [8, 9, 10, 11]
  head 3 gets: [12, 13, 14, 15]
```

Each head does the **exact same Step 3 attention**, just on its own
4-number slice of Q, K and V.

## 3. The pipeline and the shapes

Real shapes from our run (B=2, T=max_len=12, d_model=16, h=4, d_k=4):

```
x                (2, 12, 16)       (B, T, d_model)
x @ W_q          (2, 12, 16)       one big projection (also for K and V)
split_heads      (2, 4, 12, 4)     (B, h, T, d_k)   ← heads become a "batch" dim
attention        (2, 4, 12, 12)    (B, h, T, T)     one T×T map PER HEAD
weights @ V      (2, 4, 12, 4)     (B, h, T, d_k)
merge_heads      (2, 12, 16)       (B, T, d_model)  ← "concat" the heads
@ W_o            (2, 12, 16)       (B, T, d_model)  same shape as x
```

```
Multi-Head(x) = Concat(head_0, …, head_{h-1}) W_o
  where head_i = Attention(x W_q_i, x W_k_i, x W_v_i)
```

### The split, step by step

```
(B, T, d_model)  ──view──►  (B, T, h, d_k)  ──transpose(1,2)──►  (B, h, T, d_k)
   (2, 12, 16)                (2, 12, 4, 4)                       (2, 4, 12, 4)
```

- `view` only **relabels** the last dimension: 16 → 4 × 4. No data moves.
- `transpose(1, 2)` moves the head dim in front of T. Now every head has its
  own `(T, d_k)` block, just like the `(T, d_k)` in Step 3.

**Merge** is the reverse: `transpose(1, 2)` then `view(B, T, 16)`.
(`.contiguous()` is needed because after a transpose the data is no longer
laid out in order in memory, and `view` requires that it is.)

### Why we can reuse `scaled_dot_product_attention` unchanged

`@` treats **all leading dims as batch dims**. In Step 3 the batch was `B`.
Now it is `B × h`. The function never knows there are heads. It just does
the attention 2 × 4 = 8 times in parallel.

The only change is to the mask. `mask` is `(B, T)` and the scores are
`(B, h, T, T)`, so we reshape the mask to `(B, 1, 1, T)`:

```
mask (B, T)
  .unsqueeze(1) in MultiHeadAttention           → (B, 1, T)
  .unsqueeze(1) in scaled_dot_product_attention → (B, 1, 1, T)
                                                     │  │  └ which keys are real
                                                     │  └ same for every query row
                                                     └ same for every head
```

## 4. Why one big W_q instead of 4 small ones?

A `(16 × 16)` matrix is the same as four `(16 × 4)` matrices placed side by
side. Multiplying once and slicing gives the **same numbers** as 4 separate
multiplies, but it is one fast operation. So "4 heads" does not mean 4
`SelfAttention` objects. It means **one projection, cut into 4**.

## 5. What is W_o for?

After concat, the 16 numbers are just 4 separate reports placed side by side:
numbers 0–3 come only from head 0, and so on. `W_o` (16 × 16) **mixes** them,
so every output number can use what every head found.

Parameters:
```
params: 1024 = 4 x 16 x 16  (W_q, W_k, W_v, W_o)
```
Notice that the number of heads does **not** change the parameter count.
h=1 and h=4 both use 1024. Heads change *how* the numbers are used, not how
many there are.

## 6. Each head looks somewhere different

From our run, this is the attention row for the query **love** in every head
(real tokens only; **untrained** model):

```
           [CLS]       i    love      my  family
head 0     0.99    0.01    0.00    0.00    0.00
head 1     0.01    0.00    0.00    0.26    0.73
head 2     0.00    0.35    0.00    0.00    0.64
head 3     0.00    0.04    0.95    0.00    0.00
```

- Head 0 → almost all on *[CLS]*
- Head 1 → mostly *family*
- Head 2 → split between *i* and *family*
- Head 3 → itself (*love*)

Even with random weights, the heads already differ, because each one has
its own slice of W_q and W_k. After training (step 7), they tend to
**specialise**. Step 8 will draw these maps.

Mask check across all heads:
```
total weight on [PAD] keys, all heads: 0.0
```

## 7. Sanity check against PyTorch

We copied our weights into PyTorch's `nn.MultiheadAttention`
(`in_proj_weight` = W_q, W_k, W_v stacked; `out_proj` = W_o). Watch out:
PyTorch's `key_padding_mask` uses the **opposite** convention. `True` means
"ignore", so we pass `mask == 0`.

```
matches torch's nn.MultiheadAttention: True
```

## Key takeaways

- Multi-head = **split d_model into h slices, do attention in each, concat,
  mix with W_o**.
- `d_k = d_model / h`. Total size and parameter count stay the same.
- Heads become an extra batch dim: `(B, h, T, d_k)`. The attention function
  is unchanged.
- Output shape `(B, T, d_model)` = input shape. That is what lets us **stack**
  layers in Step 5.
- Different heads → different relationships, all at the same time.

## Try it yourself

1. **Shapes by hand:** with `B=2`, `max_len=12`, `d_model=16`, `h=2`, write
   the shapes after `split_heads`, of `weights`, and after `merge_heads`.
   Then change `N_HEADS` in `demo_multi_head` and check.
2. **Bad head count:** set `N_HEADS = 3`. What happens, and why?
   (Hint: 16 / 3.)
3. **Param count:** predict the parameter count for `h = 1, 2, 8`. Run it to
   check that it never changes.
4. **Another query:** in section 4 of the demo, change `"love"` to `"family"`.
   Do the 4 heads still look at different places?
5. **Without W_o:** in `forward`, return `self.merge_heads(out)` and skip
   `W_o`. The shape is still `(B, T, d_model)`. So what is lost? (Think: can
   output number 0 use anything head 3 found?) Put it back afterwards.
6. **Second sentence:** print the per-head rows for `"again"` in
   `"he lied to me again"` (use `ids[1]`, `mask[1]`, `w[1]`).

**Next → Step 5: Encoder Block** (multi-head attention + feed-forward + residual + LayerNorm).
