# Step 8: Inference + Looking Inside the Attention

File: `predict.py` (plus `Tokenizer.from_vocab` in `tokenizer.py`)
Run: `.venv/bin/python predict.py` or `.venv/bin/python predict.py "your sentence"`
(Needs `checkpoint.pt`. Run `train.py` first.)

## 1. The idea in one sentence

Load the trained model, feed it **new sentences**, read off the emoji, and
look at **which words it used** in two ways: the attention weights, and
removing words one at a time.

## 2. Loading the checkpoint

```python
ckpt  = torch.load("checkpoint.pt")
tok   = Tokenizer.from_vocab(ckpt["itos"])            # SAME vocab as training
model = EmotionTransformer(tok.vocab_size, len(EMOJIS), **ckpt["config"])  # SAME shape
model.load_state_dict(ckpt["model"])                  # the learned weights
model.eval()                                          # dropout OFF
```

Three things must match training exactly:
- **Vocab:** "love" must get the same id it had during training, or the
  embedding row is wrong.
- **Config:** `d_model`, `n_heads`, ... must match, or the weight shapes don't
  fit and `load_state_dict` raises an error.
- **`eval()` mode:** otherwise dropout makes every prediction random.

Inference is then just the forward pass from Step 6 plus softmax. No loss,
no backward, no optimizer, and `torch.no_grad()` to save memory.

## 3. Predictions (real output)

| Sentence | Where it comes from | Tokens the model sees | Top prediction |
|----------|--------------------|-----------------------|----------------|
| i passed my exam | training set | `[CLS] i passed my exam` | 😊 joy 1.00 ✅ |
| i love you so much | validation (never trained on) | `[CLS] i love you so much` | 😍 love 0.99 ✅ |
| my dog is so happy today | new | `[CLS] my [UNK] is so happy today` | 😊 joy 0.99 ✅ |
| i am so angry | new | `[CLS] i am so angry` | 😠 anger 1.00 ✅ |
| i am scared of the dark | new | `[CLS] i am [UNK] of the dark` | 😊 joy 0.59 ❌ (fear) |
| wow i never expected that | new | `[CLS] wow i never [UNK] that` | 😠 anger 0.48 ❌ (surprise) |
| my cat died | new | `[CLS] my cat [UNK]` | 😊 joy 0.54 ❌ (sadness) |

Look at the failures: in **every** one, the word that carries the emotion
(*scared*, *expected*, *died*) is `[UNK]`. The model never saw it in
training, so it simply cannot know it. The wrong answers are also
**unsure** (0.48–0.59) compared with the confident right ones (0.99–1.00).
This is the overfitting from Step 7, seen from the user's side.

## 4. Attention maps: where did `[CLS]` look?

During the forward pass, every `MultiHeadAttention` saves its weights in
`.weights`, with shape `(B, h, T, T)`:

```
attention maps: [(5, 4, 12, 12), (5, 4, 12, 12)]  one (B, h, T, T) per layer
```

The classifier reads only `[CLS]`, so the interesting part is **row 0**: how
much `[CLS]` took from each word. Last layer, real tokens only:

```
'i passed my exam'
             [CLS]       i  passed      my    exam
  head 0      0.26    0.34    0.18    0.15    0.07
  head 1      0.32    0.51    0.05    0.10    0.02
  head 2      0.05    0.05    0.32    0.08    0.49
  head 3      0.06    0.28    0.12    0.49    0.04
  average     0.17    0.30    0.17    0.21    0.16
```

```
'i am so angry'
             [CLS]       i      am      so   angry
  average     0.10    0.20    0.13    0.17    0.40      ← strongest: angry
```

- The **heads differ** (Step 4). Head 2 focuses on *passed* and *exam*,
  head 3 on *my*.
- For "i am so angry", `[CLS]` puts the most weight on *angry*, which looks
  like a nice explanation.

But look at this one:

```
'my dog is so happy today'
             [CLS]      my   [UNK]      is      so   happy   today
  average     0.08    0.13    0.08    0.34    0.11    0.04    0.22
```

*happy* gets only **0.04**, the lowest of all. Yet the model says joy
with 0.99. Is it ignoring the word "happy"?

## 5. A better test: remove one word at a time

`word_importance` deletes each word in turn and measures how much the
probability of the predicted emoji **drops**. This is called *occlusion*.

```
'my dog is so happy today'     drop in p(😊)
    my         -0.00
    dog        +0.00
    is         +0.01
    so         -0.00
    happy      +0.87 █████████████████     ← the word that really matters
    today      +0.02

'i passed my exam'             drop in p(😊)
    passed     +1.00 ████████████████████

'i am so angry'                drop in p(😠)
    angry      +1.00 ████████████████████
```

So the model **does** rely on *happy*. The last-layer attention just didn't
show it.

### Why can attention be misleading?

In the **last** layer, `[CLS]` does not attend to the raw words. It attends to
token vectors that have **already mixed** in layer 1. By then the vector at
position *is* or *today* may already carry the "happy" information it
picked up from its neighbours. `[CLS]` can collect "happy" through them.

```
layer 1:  "is", "today" ... attend to "happy"  → they now contain happiness
layer 2:  [CLS] attends to "is", "today"       → gets happiness indirectly
```

**Lesson:** attention weights show *where information was copied from in
that layer*, not *which input word caused the answer*. They are a useful
hint, not a proof. Tests like word removal (or gradient-based methods) are
more reliable for "why did it decide this?".

### When the model is unstable

```
'i am scared of the dark'      drop in p(😊)
    i          -0.36
    am         +0.58 ████████████
    scared     -0.33
    of         -0.39
    the        +0.46 █████████
    dark       -0.15
```

Removing almost **any** word swings the answer by ±0.3–0.6, and the
"important" words are *am* and *the*. The model has no real idea what this
sentence means. It is reacting to surface patterns it memorised (Step 7).
A model that really understood would depend mostly on *scared* and *dark*.

## 6. The journey, start to finish

```
"my dog is so happy today"
  │ Step 1  tokenizer      [2, 99, 1, 76, 126, 63, 145, 0, 0, 0, 0, 0]  (B, T) = (1, 12)
  │ Step 2  embeddings     (1, 12, 16)  token vector × √d + position
  │ Step 3  attention      softmax(QKᵀ/√d_k)V: each token mixes in others
  │ Step 4  multi-head     4 heads × d_k=4, each its own pattern
  │ Step 5  encoder block  attn + FFN, residual + LayerNorm, × 2 layers
  │ Step 6  [CLS] → Linear (1, 16) → (1, 6) logits
  │ Step 7  trained weights from 60 sentences
  ▼ Step 8  softmax → 😊 0.99
```

(Real ids from our checkpoint. They change if you edit `data.py` and retrain.)

## Key takeaways

- Inference = **load vocab + config + weights → `eval()` → forward →
  softmax**. No loss, no backward.
- Save the **vocab** with the weights. A different vocab breaks everything.
- `[UNK]` on the key word is the main reason for wrong answers. That's a
  data problem, not an architecture problem.
- Attention maps show where each layer copied information from. They are a
  **hint, not an explanation**, because tokens mix in earlier layers.
- Removing words (occlusion) is a simple, more honest importance test.

## Try it yourself

1. **Your sentences:** `.venv/bin/python predict.py "i won the lottery" "my
   friend betrayed me"`. Look at the `tokens:` line first. Can you predict
   the failures from the `[UNK]`s alone?
2. **Fix a failure:** add "i am scared of the dark" style sentences (with
   *scared*) to the fear list in `data.py`, re-run `train.py`, then
   `predict.py`. Does it work now?
3. **Layer 1 vs layer 2:** in `show`, change `attn[-1]` to `attn[0]`. For
   "my dog is so happy today", does `[CLS]` look at *happy* more in layer 1?
4. **Who carries "happy"?** *happy* is token 5 in that sentence. Print
   `attn[0][b, :, :, 5].mean(0)`: how much every token looked at *happy* in
   layer 1 (averaged over heads). Do the tokens that took the most from
   *happy* match the ones `[CLS]` looked at in layer 2?
5. **Confidence:** find a sentence where the top probability is below 0.5.
   Is it always one with `[UNK]` words?
6. **Unstable model:** run word removal on 3 training sentences. Are the
   drops cleaner (one big bar) than on new sentences? Why?

---

🎉 **That's the full transformer.** Tokenizer → embeddings + positions → Q/K/V
attention → multi-head → encoder blocks → classifier → training → inference.
Every piece of BERT and GPT is here, just much smaller. Bonus ideas: a
decoder (masked self-attention + cross-attention), a pre-norm vs post-norm
comparison, or fine-tuning a real pre-trained model (LoRA/QLoRA).
