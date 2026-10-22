# Step 5: The Encoder Block

File: `encoder.py`
Run: `.venv/bin/python encoder.py`

## 1. The idea in one sentence

An encoder block is **attention (tokens talk to each other)** followed by a
**feed-forward network (each token thinks on its own)**. Each part is wrapped
in a **residual connection + LayerNorm** so it can be stacked many times and
still train.

```
x ──► [ x + MultiHeadAttention(x) ] ──► LayerNorm
  ──► [ x + FeedForward(x) ]        ──► LayerNorm ──► out
```

This block is the repeating unit of BERT, GPT and every other transformer.
Stack N of them and you have "the transformer".

## 2. The four pieces

| Piece | What it does | Mixes across tokens? |
|-------|--------------|----------------------|
| Multi-head attention (Step 4) | Gathers information from other tokens | ✅ yes |
| Feed-forward network (FFN) | Processes each token's vector on its own | ❌ no |
| Residual `x + sublayer(x)` | Keeps the original signal and only *adds* to it | — |
| LayerNorm | Rescales each token vector to mean 0 and std 1 | ❌ no |

A good way to remember it: **attention = communicate, FFN = compute.**

## 3. Shapes

Real output (B=2, T=max_len=12, d_model=16, h=4, d_ff=64):

```
x            : (2, 12, 16)   (B, T, d_model)
attention    : (2, 12, 16)   (B, T, d_model)
ffn hidden   : (2, 12, 64)   (B, T, d_ff)  wider on the inside
block output : (2, 12, 16)   (B, T, d_model)  same as x
```

Input shape = output shape. That is the whole reason we can stack blocks.

## 4. The feed-forward network

```
FFN(x) = Linear2( ReLU( Linear1(x) ) )
         16 ──► 64 ──► ReLU ──► 16
```

- It widens to `d_ff = 4 × d_model` (as in the paper), applies a
  non-linearity, then narrows back.
- Attention is mostly **weighted averaging**, which is close to linear.
  The ReLU in the FFN gives the model its real non-linear "thinking" power.
- `nn.Linear` acts only on the **last** dim, so every token goes through the
  same FFN **separately**. T behaves like a batch dimension.

Proof from our run. We change **only token 2**, then measure how much each
token's output changes:

```
  ffn       : [0.00, 0.00, 2.60, 0.00, 0.00]
  attention : [2.18, 5.48, 2.48, 2.84, 1.96]   (all tokens change: attention mixes)
```

The FFN reacts only at token 2. Attention spreads the change to everyone.

## 5. Residual connections: `x + sublayer(x)`

The sublayer does not *replace* x. It computes a **change** that gets added:

```
x = x + attn(x)     "keep what you had, plus what you learned from others"
```

Why this helps:
- **Gradients have a highway.** The derivative of `x + f(x)` is `1 + f'(x)`.
  The `1` lets the gradient flow straight back through many layers without
  shrinking to zero.
- **A block can easily do "nothing".** If `f(x) ≈ 0`, the block passes x
  through. So adding layers never has to make things worse.
- **Information is not lost.** The token's identity and position (Step 2)
  survive all the way up.

## 6. LayerNorm

For **each token vector separately**, LayerNorm subtracts its mean, divides by
its std, then applies a learned scale `γ` and shift `β` (16 each):

```
LayerNorm(v) = γ · (v − mean(v)) / std(v) + β
```

From our run (first 5 tokens of "i love my family"):

```
input x : mean [0.76, 1.04, 0.44, 1.00, 0.82]   std [3.69, 4.07, 3.94, 4.24, 3.62]
output  : mean [0.00, 0.00, 0.00, 0.00, 0.00]   std [1.00, 1.00, 1.00, 1.00, 1.00]
```

(γ starts at 1 and β at 0, so before training the output is exactly mean 0,
std 1.) Without it, every `x + f(x)` adds a bit more, and the numbers grow
layer after layer. LayerNorm keeps them in a stable range.

**Layer**Norm vs **Batch**Norm: BatchNorm averages over the batch.
LayerNorm averages over the features of **one token**, so it works the same
for any batch size and does not mix sentences.

## 7. Post-norm vs pre-norm

```
post-norm (paper, our default):  x = LayerNorm(x + sublayer(x))
pre-norm  (GPT-2 and later)   :  x = x + sublayer(LayerNorm(x))
```

Pre-norm keeps a "clean" residual path with nothing on it. That makes deep
models easier to train, which is why most modern LLMs use it. It needs one
extra LayerNorm at the very end (`Encoder.final_norm`). Our model is tiny,
so either works. Switch with `norm_first=True`.

## 8. Parameters

```
params in one block:
  attention :  1024  = 4 x 16 x 16
  ffn       :  2128  = (16 x 64 + 64) + (64 x 16 + 16)
  2 x norm  :    64  = 2 x (gamma 16 + beta 16)
  total     :  3216
```

The FFN holds **about 2/3** of each block's parameters. That is true in real
LLMs too.

Stacking:
```
encoder with 2 blocks: (2, 12, 16), params 6432 = 2 x 3216
```
Each block has its **own** weights (`nn.ModuleList` of separate blocks).
The blocks are the same design but do not share weights.

## 9. Padding is still safe

We filled the `[PAD]` vectors with huge random numbers and ran the block again:
```
real-token outputs unchanged when [PAD] vectors are garbage: True
```
Attention masks pad keys (Step 3). FFN and LayerNorm are per-token. So
nothing from a pad position can ever reach a real token.

## 10. Sanity check against PyTorch

We copied our weights into `nn.TransformerEncoderLayer`. Its attention has
biases, so we set them to zero.

```
matches torch's nn.TransformerEncoderLayer: True
```

We compare real tokens only, because PyTorch's fast path may write zeros into
`[PAD]` rows.

## Key takeaways

- Block = **attention → add & norm → FFN → add & norm**.
- Attention mixes **across** tokens. The FFN and LayerNorm work **per token**.
- Residuals make deep stacks trainable. LayerNorm keeps the numbers stable.
- In → out shape is `(B, T, d_model)` → `(B, T, d_model)`, so we can stack N
  blocks.
- The FFN holds most of the parameters.

## Try it yourself

1. **Param count by hand:** with `d_model=32`, `h=4`, `d_ff=128`, compute
   the parameter count of one block. Then check by changing the constants.
2. **Prove the FFN is per-token:** in section 4 of the demo, change token 4
   instead of token 2. Which FFN entry is non-zero now?
3. **Pre-norm:** create the block with `norm_first=True`. Is the output still
   mean 0 / std 1 per token? Why not? (Look at what the last operation is.)
4. **Remove the residual:** temporarily change the post-norm lines to
   `x = self.norm1(self.drop1(self.attn(x, mask)))` (and the same for the FFN).
   It still runs. Read section 5 and explain what the model has lost.
5. **Deeper:** set `N_LAYERS = 6` (as in the original paper). How many
   parameters does the encoder have? Does the output shape change?
6. **No mask:** in section 5 of the demo, call `block(x3)` and `block(x)`
   without `mask`. Are the real-token outputs still unchanged? Why not?

**Next → Step 6: Full Model** (embeddings + encoder + `[CLS]` classifier head → emoji scores).
