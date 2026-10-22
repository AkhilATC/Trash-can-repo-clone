# Step 6: The Full Model

File: `model.py` (class `EmotionTransformer`, helper `encode_batch`)
Run: `.venv/bin/python model.py`

## 1. The idea in one sentence

Connect everything: **ids → embeddings → encoder → take the `[CLS]` vector →
one linear layer → a score per emoji.**

## 2. The whole pipeline, with real shapes

(B=2, T=max_len=12, d_model=16, h=4, d_ff=64, N=2 blocks, 6 classes)

```
ids    : (2, 12)      (B, T)                        step 1: tokenizer
x      : (2, 12, 16)  (B, T, d_model)  after embeddings   step 2
h      : (2, 12, 16)  (B, T, d_model)  after encoder      steps 3-5
cls    : (2, 16)      (B, d_model)     h[:, 0]            ← NEW
logits : (2, 6)       (B, n_classes)                      ← NEW
probs  : (2, 6)       (B, n_classes)  row sums: [1., 1.]
```

The new part is only three lines in `forward`:

```python
h = self.encoder(x, mask)   # (B, T, d_model)
cls = h[:, 0]               # (B, d_model)   take position 0 of every sentence
return self.head(cls)       # (B, n_classes)
```

## 3. Why the `[CLS]` token?

The encoder gives us **12 vectors per sentence**, but we need **one answer**
per sentence. We have to squeeze `(T, d_model)` into `(d_model,)`.

`[CLS]` is a token we add at position 0 of every sentence (Step 1). Its
input embedding is the same in every sentence, so it starts with no meaning
at all. But in every encoder block it **attends to all the real words**. By
the top of the encoder, its vector is a **summary of the whole sentence**.

Proof from our run. Change only the last word. Position 0 has the same input,
yet its output changes:

```
[CLS] vector, 'i love my family' vs 'i love my dog'
  same input at position 0 ([CLS])  : True
  [CLS] output changed by           : 2.41
```

Training (Step 7) will teach `[CLS]` *what* to collect: the parts of the
sentence that matter for the emotion.

Other options (not used here):
- **Mean pooling:** average all real (non-pad) token vectors. Also common.
- **Last token:** GPT-style decoders use the final token, because only it
  has seen the whole sentence.

## 4. The classifier head

```
cls (16) ──► Dropout ──► Linear(16 → 6) ──► logits (6)
```

That's all: **one linear layer** (`16 × 6 + 6 = 102` parameters). It works
because the encoder has already done the hard work. The head just reads the
answer off the `[CLS]` vector.

## 5. Logits vs probabilities

- **Logits** are raw scores. They can be any number, positive or negative.
- **Softmax** turns them into probabilities (all ≥ 0, summing to 1).

The model returns **logits**, not probabilities. That is on purpose.
PyTorch's `cross_entropy` loss (Step 7) applies softmax itself, in a
numerically safer way. We only call `softmax` ourselves when we want to
*show* probabilities.

## 6. Parameters

```
params:
  embeddings :  3088  = vocab 193 x 16
  encoder    :  6432  = 2 blocks x 3216
  head       :   102  = 16 x 6 + 6
  total      :  9622
```

The whole model is under 10k parameters. GPT-2 small has about 124 million.
The architecture is the same; only the sizes differ.

(The vocab size depends on `data.py`. If you add sentences or a class, these
numbers change. The number of classes is read from `EMOJIS` automatically.)

## 7. Untrained = guessing

```
                           😊     😢     😠     😨     😍     😲
  i love my family       0.20   0.25   0.15   0.13   0.18   0.09   -> 😢
  he lied to me again    0.20   0.22   0.17   0.12   0.21   0.08   -> 😢
```

The probabilities are spread almost evenly, and the "prediction" is wrong.
Over the whole dataset:

```
whole dataset (72 sentences) before training:
  loss     = 1.880   (random guessing = ln(6) = 1.792)
  accuracy = 0.17    (random guessing = 1/6 = 0.17)
```

**Why ln(6)?** Cross-entropy loss is `−ln(probability given to the right
class)`. If the model gives every class `1/6`, the loss is `−ln(1/6) = ln(6)
≈ 1.79`. A starting loss close to `ln(n_classes)` is a useful sanity check
that the model is wired up correctly. If it were much higher, something
would be off. Step 7 will push this number down.

## 8. `model.train()` vs `model.eval()`

Dropout randomly zeros some values, but **only in train mode**:

```
train mode, same input twice, same logits? False
eval mode,  same input twice, same logits? True
```

Rule: call `model.train()` before training and `model.eval()` before
predicting or measuring accuracy.

## 9. `encode_batch`, a small helper

```python
ids, mask = encode_batch(tok, ["i love my family", "he lied to me again"], max_len=12)
```

It turns a list of strings into the `(B, T)` tensors the model expects. We'll
reuse it in Steps 7 and 8.

## Key takeaways

- Full model = **Embeddings → Encoder → `h[:, 0]` → Linear**.
- `[CLS]` starts with no meaning and becomes the sentence summary through
  attention.
- The model outputs **logits**. Softmax is only for showing probabilities.
- Untrained loss ≈ `ln(n_classes)` and accuracy ≈ `1/n_classes`. That is
  the baseline to beat.
- Use `train()` / `eval()` to switch dropout on and off.

## Try it yourself

1. **Shapes by hand:** with `B=8` sentences and 7 classes (if you add
   "tired"), write the shapes of `h`, `cls` and `logits`.
2. **Param count:** change `n_layers` to 4. Predict the new total before you
   run it. (Only one line of the param table should change.)
3. **Logits vs probs:** print `logits` for one sentence. Then apply softmax
   yourself and check it sums to 1. Are any logits negative?
4. **Mean pooling:** in `forward`, replace `h[:, 0]` with the average of the
   real tokens: `(h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)`.
   What shape is it? Why multiply by the mask first? Put `[CLS]` back
   afterwards.
5. **Loss sanity check:** if you had 7 classes, what loss would you expect
   before training?
6. **Seed:** change `torch.manual_seed(0)` to another number. The untrained
   predictions change, but does the loss stay near `ln(6)`?

**Next → Step 7: Training** (loss, backprop, optimizer: make the loss go down from ~1.79).
