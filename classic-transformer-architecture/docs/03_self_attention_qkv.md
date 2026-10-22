# Step 3: Self-Attention with Q, K, V

File: `attention.py`
Run: `.venv/bin/python attention.py 3`

## 1. The idea in one sentence

Each token builds its new vector as a **weighted average of the other
tokens' vectors**. The weights are learned, and they depend on the input.

In "he lied to me **again**", the word *again* should pull information from
*lied*. In "i love my family", *love* should pull from *family*.
Attention is how the model learns to do that.

## 2. Query, Key, Value: a library analogy

From the same input `x`, each token makes three vectors using three learned
matrices:

| Vector | Made by | Meaning |
|--------|---------|---------|
| **Q**uery | `x @ W_q` | "What am I looking for?" (your search request) |
| **K**ey   | `x @ W_k` | "What do I contain?" (the label on a book's spine) |
| **V**alue | `x @ W_v` | "What do I hand over if picked?" (the book's content) |

1. Compare my **query** with every **key**. A dot product gives a match score.
2. Turn the scores into weights with softmax. They sum to 1.
3. Take the weighted mix of the **values**.

It is called **self**-attention because Q, K and V all come from the *same*
sentence. (In cross-attention, used in decoders, Q comes from one sequence
and K, V come from another.)

Why separate K and V? What makes a word *relevant* (K) is not always the
same as what it should *contribute* (V). Splitting them gives the model
more freedom.

## 3. The formula and the shapes

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k  + mask ) V
```

Shapes from our run (B=2 sentences, T=max_len=12 tokens, d_model=16, d_k=8):

```
x        (2, 12, 16)
Q, K, V  (2, 12, 8)     each = x @ W  where W is (16 × 8)
Q Kᵀ     (2, 12, 12)    (B,T,d_k) @ (B,d_k,T) → (B,T,T): every token vs every token
softmax  (2, 12, 12)    over the LAST dim (the keys) → each row sums to 1
@ V      (2, 12, 8)     (B,T,T) @ (B,T,d_k) → (B,T,d_k)
```

Learnable parameters: only the 3 projection matrices,
`3 × 16 × 8 = 384`. Attention itself has no weights. It is just matrix
multiplication and softmax.

The `(T, T)` matrix is why attention costs **O(T²)**. Long contexts are
expensive because of this.

## 4. Reading the attention matrix

Real output for "i love my family" with `max_len = 12` (an **untrained**
model, so the pattern is still random):

```
            [CLS]      i   love     my family  [PAD]  [PAD]  [PAD]  [PAD]  [PAD]  [PAD]  [PAD]
    [CLS]    0.06   0.91   0.01   0.00   0.01   0.00   0.00   0.00   0.00   0.00   0.00   0.00
        i    0.01   0.00   0.91   0.00   0.08   0.00   0.00   0.00   0.00   0.00   0.00   0.00
     love    0.62   0.00   0.00   0.38   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
       my    0.00   0.00   0.00   0.01   0.99   0.00   0.00   0.00   0.00   0.00   0.00   0.00
   family    0.00   0.96   0.00   0.04   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
    [PAD]    0.38   0.18   0.05   0.27   0.11   0.00   0.00   0.00   0.00   0.00   0.00   0.00
    ... 6 more [PAD] rows, same idea ...
```

It is a 12 × 12 matrix: 12 queries × 12 keys. Only 5 tokens are real
(`[CLS] i love my family`). The other 7 are padding.

- **Row** = the token that is asking (the query).
- **Column** = the token it reads from (the key).
- Row `i`: "i" takes 91% of its new vector from "love".
- Every row sums to 1 (verified: `row sums = 1.00 …`).
- After training (step 7), these rows will become meaningful. Step 8
  visualises them.

## 5. The padding mask

`[PAD]` is filler and must not leak into real tokens. Before softmax, the
scores for **pad keys** are set to `-inf`. `e^(-inf) = 0`, so they get
zero weight:

```
total weight on [PAD] keys: 0.0
```

The `[PAD]` columns above are all `0.00`. The `[PAD]` *rows* still produce
an output, but nothing reads from them, and the classifier only reads
`[CLS]`. So they are harmless.

Mask shape trick: `mask` is `(B, T)`. `mask.unsqueeze(1)` gives `(B, 1, T)`,
which broadcasts over every query row.

## 6. Why divide by √d_k?

If the entries of q and k have variance 1, then `q·k` sums `d_k` terms, so
its variance is `d_k`. Big scores make softmax nearly **one-hot**. Gradients
then vanish and learning stalls.

From our run with d_k=512:
```
std of raw scores = 18.7,  scaled = 0.8
softmax raw    : [0.68, 0.00, 0.00, 0.24, 0.08, 0.00]   ← spiky
softmax scaled : [0.23, 0.03, 0.18, 0.22, 0.21, 0.13]   ← smooth, trainable
```
Dividing by √512 ≈ 22.6 brings the std back to about 1.

## 7. Sanity check against PyTorch

PyTorch has a built-in `F.scaled_dot_product_attention`. It is the fast,
fused version. Ours gives the same numbers:
```
matches torch's F.scaled_dot_product_attention: True
```

## Key takeaways

- Attention = **softmax(QKᵀ/√d_k) V**: score, normalise, mix.
- Q/K/V are 3 linear projections of the same `x`. Those are the only weights.
- Output shape = input token count, `(B, T, d_k)`. Every token gets a new,
  context-aware vector.
- The mask stops tokens from reading `[PAD]`. The √d_k keeps softmax smooth.
- One head can only focus on one kind of relationship at a time. That is
  why we need **multi-head attention** (Step 4).

## Try it yourself

1. **Mask off:** in the demo, call `attn(x)` without the mask. How much
   weight goes to `[PAD]` keys now? Why is that bad?
2. **Shapes by hand:** with `B=4`, `max_len=12`, `d_model=32`, `d_k=16`, write
   down the shapes of `Q`, `Q Kᵀ`, `weights` and `out`. Then check by changing
   the constants in `attention.py`.
3. **Scaling:** remove `/ math.sqrt(d_k)` from `scaled_dot_product_attention`.
   Do the attention rows look more "one-hot"? (Check the max value per row.)
4. **Cost:** if `max_len` doubles from 12 to 24, how many numbers does the
   attention matrix hold per sentence? (Hint: T².)
5. **Your sentence:** put your own sentence in `sentences`, including a word
   that is not in the vocab. Which token shows up in the matrix?

**Next → Step 4: Multi-Head Attention** (several heads in parallel, each learning a different relationship).
