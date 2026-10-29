### We set the scores of [PAD] keys to -inf before softmax, so no token pays any attention to padding.

```text
scores  = Q · Kᵀ / √dk                                   (B, T, T)
scores  = scores.masked_fill(mask.unsqueeze(1) == 0, -inf)
weights = softmax(scores, dim=-1)                        (B, T, T), each row sums to 1
```

Problem
```text
Every sentence is padded to MAX_LENGTH = 12. For "Howdy Aki":

  ids : [ 0, 28,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1]    [CLS] howdy [UNK] [PAD]...
  mask: [ 1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0]    shape (1, 12), int64

Without a mask, softmax hands weight to the 9 [PAD] positions too.
PAD carries no meaning, so the output would be diluted by junk, and the result
would depend on how much padding a sentence happens to have.
```

Why -inf and not 0
```text
softmax uses exp(score):  exp(0) = 1 (still counts),  exp(-inf) = 0 (gone).

Same row, last 2 positions are PAD (mask = [1, 1, 0, 0]):

  scores                  : [2.0,    1.0,    0.5,    0.3   ]
  softmax, no mask        : [0.5638, 0.2074, 0.1258, 0.1030]   <- PAD gets ~23%
  softmax, PAD set to 0   : [0.6103, 0.2245, 0.0826, 0.0826]   <- PAD still gets ~16%
  softmax, PAD set to -inf: [0.7311, 0.2689, 0.0000, 0.0000]   <- PAD gets exactly 0

With -inf, the real tokens share 100% of the weight between them.
```

Which dimension to mask: columns (keys), not rows (queries)
```text
scores[b, i, j] = how much query token i looks at key token j.
   rows    i = "who is looking"
   columns j = "who is being looked at"

We want: nobody looks AT a PAD token  ->  mask the PAD columns, the same in every row.

  mask                 (B, T)    [1 1 1 0 0 ...]
  mask.unsqueeze(1)    (B, 1, T) [[1 1 1 0 0 ...]]
  broadcast vs scores  (B, T, T) [1 1 1 0 0 ...]   row 0
                                 [1 1 1 0 0 ...]   row 1
                                 ...               (same pattern copied down)

unsqueeze(2) -> (B, T, 1) would mask ROWS instead: PAD queries would be fully -inf (see below).
```

What PAD rows (queries) do
```text
Rows 3-11 are PAD *queries*. They are not masked, so they still attend to the
3 real tokens and produce some output. That is harmless: the classifier reads
only position 0 ([CLS]), so PAD rows are computed and then ignored.
```

Edge case: a row that is all -inf
```text
softmax([-inf, -inf, -inf, -inf]) = [nan, nan, nan, nan]   (0 / 0)

That would happen if a sentence had NO real tokens. With our tokenizer it can't:
position 0 is always [CLS] with mask = 1, so every row has at least one real key.
```

Real output from attention_layer.py ("Howdy Aki", random init, so your numbers will differ)
```text
SCORES SHAPE  torch.Size([1, 12, 12])
WEIGHTS SHAPE torch.Size([1, 12, 12])
row 0 ([CLS] query): [0.057, 0.525, 0.418, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
row sums:            [1.000, 1.000, ... 1.000]   (all 12 rows)
```

Try it yourself
```text
1. Predict: for the input "what is the weather today", which column index is
   the first PAD? Change input_sentence and check row 0 of the weights.
2. Replace float('-inf') with 0.0, run, and look at row 0. How much weight
   (sum of positions 3-11) leaks to PAD now?
3. Change mask.unsqueeze(1) to mask.unsqueeze(2). Run and print weights[0, 3].
   Why do you get nan? Which dimension did you mask by mistake?
4. Run with two sentences at once: input_sentence = ["Howdy Aki", "what is the weather today"].
   Print weights.shape, then weights[0, 0] and weights[1, 0]. Do they have
   different numbers of non-zero entries?
5. Hand calc: scores = [3.0, 0.0, 1.0], mask = [1, 0, 1]. Write the masked row,
   then the softmax on paper. Check it with torch.softmax.
```
