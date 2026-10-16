# Step 2: Embeddings and Positional Encoding

File: `embeddings.py`
Run: `.venv/bin/python embeddings.py`

## 1. From ids to vectors: Token Embedding

A token id like `99` ("love") is just a label. The number 99 isn't "bigger"
than 81 ("i") in any useful way. So every id is mapped to a **vector of
`d_model` numbers** that the model *learns*:

```
embedding table  (vocab_size × d_model) = (193 × 16)

 id 0  [PAD]  → [0, 0, 0, ..., 0]         (kept at zero: padding_idx)
 id 1  [UNK]  → [ 0.3, -1.2, ...,  0.8]
 ...
 id 99 love   → [-0.5,  0.9, ..., -0.1]
```

`nn.Embedding` is just this table plus a row lookup. It has
`vocab_size × d_model` learnable numbers (193 × 16 = **3088** here).
During training, words used in similar ways (e.g. "happy" and "great") end up
with similar vectors.

```
ids (B, T)  ──►  embedding  ──►  (B, T, d_model)
(2, 12)                           (2, 12, 16)
```

**`d_model`** is the "width" of the transformer. Every layer reads and writes
vectors of this size. The original paper used 512, BERT-base uses 768, and ours uses 16–32.

**Why multiply by √d_model?** The paper does this. The positional values
are in [-1, 1]. Scaling the token vectors up keeps the word meaning from
being drowned out when the two are added together.

## 2. The problem: attention has no idea of order

Self-attention (next step) compares every token with every other token. But
it treats the input as a **set**, not a sequence. So without extra
information, these two sentences look the same to it:

```
"i love my family"
"my family love i"
```

Our demo confirms it:
```
[no positions] same set of word vectors in both sentences?  True
```

RNNs get order for free because they read word by word. Transformers read
all words **at once (in parallel)**, which makes them fast but blind to order.
So we must **inject position information**.

## 3. Sinusoidal Positional Encoding

For each position `pos` and dimension pair `i`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
```

Think of it like the **hands of a clock**:
- dims 0,1 → a fast hand (it changes a lot from one position to the next)
- higher dims → slower and slower hands
- Reading all the hands together gives each position a **unique fingerprint**.

Real values from our run (positions 0–3, first 6 dims):
```
pos 0: [ 0.00,  1.00,  0.00,  1.00,  0.00,  1.00]   ← sin(0)=0, cos(0)=1
pos 1: [ 0.84,  0.54,  0.31,  0.95,  0.10,  1.00]
pos 2: [ 0.91, -0.42,  0.59,  0.81,  0.20,  0.98]
pos 3: [ 0.14, -0.99,  0.81,  0.58,  0.30,  0.96]
        └ fast ┘     └ medium ┘    └ slow ┘
```

Nearby positions have more similar fingerprints:
```
pos 0 vs pos  1: 0.936
pos 0 vs pos  2: 0.796
pos 0 vs pos 10: 0.456
```

So the model can tell both *where* a word is and *how far apart* two words are.
A nice property: `PE[pos+k]` is a fixed linear function of `PE[pos]`, so
"k steps away" is easy for attention to learn.

## 4. Add them, don't concatenate

```
x = TokenEmbedding(ids) * √d_model  +  PE[:T]
     (B, T, d_model)                  (1, T, d_model)  ← broadcast over the batch
```

Now "love" at position 2 and "love" at position 3 are **different vectors**:
```
[with positions] 'love' @pos2 == 'love' @pos3 ?   False
```

Adding keeps the size at `d_model` (no extra cost). The network learns to
use some directions for "meaning" and others for "position".

## 5. Fixed vs learned positions

| type | how | used by |
|------|-----|---------|
| Sinusoidal (ours) | fixed formula, 0 parameters, `register_buffer` | original Transformer |
| Learned | `nn.Embedding(max_len, d_model)`, trained like words | BERT, GPT-2 |
| Rotary (RoPE) | rotates Q and K inside attention | LLaMA, most modern LLMs |

`register_buffer("pe", ...)`: the tensor is saved with the model and moved
by `.to(device)`, but it is **not** a parameter, so the optimizer never changes it.

## 6. Dropout

`Embeddings` ends with `nn.Dropout`. During training it randomly zeroes some values.
This regularises our tiny dataset so the model can't just memorise it.
It is switched off automatically in `model.eval()`.

## Key takeaways

- Embedding = a learnable lookup table: id → vector of size `d_model`.
- Attention alone ignores word order, so we **add** positional encodings.
- Sinusoidal PE: sin/cos waves at different frequencies give a unique, smooth fingerprint per position.
- Output of this step, `(B, T, d_model)`, is the input to every encoder block.

## Try it yourself

1. Change `D_MODEL` to 64 in the demo. Does `pos 0 vs pos 10` similarity change?
2. Replace `SinusoidalPositionalEncoding` with a learned `nn.Embedding(max_len, d_model)`. How many parameters does that add?
3. Check that the `[PAD]` row of `emb.tok.table.weight` is all zeros.

**Next → Step 3: Self-Attention with Q, K, V** (how each word looks at the others).
