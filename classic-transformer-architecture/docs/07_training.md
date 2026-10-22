# Step 7: Training

File: `train.py`
Run: `.venv/bin/python train.py` (takes a few seconds on CPU, saves `checkpoint.pt`)

## 1. The idea in one sentence

Show the model sentences with their correct emoji, measure **how wrong** it
is (the loss), work out **which direction** to move every weight to be less
wrong (backprop), take a small step that way (the optimizer), and repeat.

## 2. The data split

```
train 60 sentences, val 12, vocab 167
```

- **Train (60):** the model learns from these.
- **Validation (12):** 2 per emotion, **never** trained on. This is our
  honest test: does the model work on sentences it has not seen?
- The vocab is built from **training sentences only**. Words that appear only
  in validation become `[UNK]`, exactly like new words in real use. (That is
  why the vocab is 167 here, not the 193 from Step 6.)

## 3. One training step in slow motion

Real output, one batch of 16 sentences:

```
logits (16, 6), labels (16,) -> loss 1.711
head weight .grad before backward(): None
head weight .grad after  backward(): (6, 16) same shape as the weight
optimizer.step() changed the head weights by 0.0030 on average
loss on the same batch after one step: 1.493
```

| Line | What happens |
|------|--------------|
| `logits = model(ids, mask)` | **Forward pass**: `(B, T)` → `(B, 6)` scores |
| `loss = cross_entropy(logits, labels)` | **One number** saying how wrong we are |
| `loss.backward()` | **Backprop**: fills `.grad` for *every* weight (embeddings, Q/K/V, FFN, LayerNorm, head) |
| `optimizer.step()` | **Update**: move every weight a little against its gradient |
| `optimizer.zero_grad()` | **Clear** `.grad`. PyTorch *adds* new gradients to old ones, so we must reset |

After one step the loss on that batch dropped from 1.711 to 1.493. Training
is just this, repeated hundreds of times.

### Cross-entropy loss

```
loss = −ln( softmax(logits)[correct class] )
```

| Probability given to the right emoji | Loss |
|--------------------------------------|------|
| 1/6 (guessing) | 1.79 |
| 0.5 | 0.69 |
| 0.9 | 0.11 |
| 0.99 | 0.01 |

It only looks at the probability of the **correct** class, and punishes
confident wrong answers very hard (−ln of a tiny number is huge).

### What does a gradient mean?

For every weight, `.grad` answers: *"if I increase this weight a tiny bit,
how much does the loss go up?"* So we move each weight in the **opposite**
direction. `loss.backward()` computes this for all 9,206 weights at once
by applying the chain rule backwards through every layer.

### AdamW

Plain gradient descent is `w = w − lr · grad`. **Adam** gives every weight
its own adaptive step size (it remembers recent gradients). **W** = weight
decay, which gently pulls weights towards 0 to discourage extreme values.
`lr = 3e-3` is the learning rate, the overall step size.

## 4. The training loop

```
for each epoch:                 1 epoch = one pass over all 60 training sentences
    model.train()               dropout ON
    shuffle the sentences
    for each mini-batch of 16:  60 / 16 → 4 steps per epoch
        forward, loss, zero_grad, backward, step
    model.eval()                dropout OFF
    measure train + val loss and accuracy
```

- **Mini-batches:** we update after every 16 sentences, not after all 60.
  This gives more updates per epoch and a little noise, which helps.
- **Shuffle every epoch:** so the batches are different each time.

## 5. The results: overfitting

Real output:

```
epoch  train loss  train acc  val loss  val acc
    1       1.858       0.17     1.880     0.17
   15       0.837       0.82     1.999     0.25
   30       0.190       1.00     2.549     0.25
   45       0.058       1.00     2.866     0.33
   60       0.026       1.00     3.088     0.33
   75       0.015       1.00     4.408     0.25
   90       0.011       1.00     3.945     0.33
  105       0.006       1.00     5.153     0.25
  120       0.008       1.00     3.717     0.25
  135       0.004       1.00     4.155     0.25
  150       0.002       1.00     5.144     0.25
```

Read it in two halves:

- **Train:** the loss falls from 1.86 to 0.002, and accuracy reaches **100%**
  by epoch 30. The model, the loss, backprop and the optimizer all **work**.
- **Val:** accuracy stays around **25–33%** (guessing is 17%), and val loss
  goes **up**, from 1.88 to over 5.

This is **overfitting**. The model has **memorised** the 60 training
sentences instead of learning general rules about emotions. As it gets more
and more confident on the training set, it also gets confidently **wrong**
on new sentences. That is why val loss rises even though val accuracy is
flat.

### Why so much overfitting here?

```
validation sentences (never trained on):
  ok  true 😊  pred 😊 0.92   today [UNK] a [UNK] [UNK]
  XX  true 😢  pred 😊 0.99   my [UNK] passed away
  XX  true 😠  pred 😢 0.98   i am [UNK] of this [UNK]
  XX  true 😨  pred 😢 0.98   i am [UNK] of [UNK]
  ok  true 😍  pred 😍 0.99   i love you so much
  ok  true 😍  pred 😍 0.81   you [UNK] me feel [UNK]
  ... (12 total, 3 correct)
```

1. **Tiny data.** 60 sentences, about 10 per emotion. The model has 9,206
   parameters, far more than it needs to memorise 60 examples.
2. **Unknown words.** 26 of the 68 validation words are `[UNK]`. "i am
   `[UNK]` of `[UNK]`" (fear) has lost exactly the words that carry the
   emotion. Nobody could classify that.
3. **No prior knowledge.** The model starts from random weights. It has
   never seen English before. It does not know that "furious" and "angry"
   are related unless both appear in the training data.

It *does* get "i love you so much" right (😍 0.99), because "love" appears
in training with that label.

**This is exactly why real models are pre-trained** on billions of
sentences first (BERT, GPT) and only then fine-tuned on a small task.
Pre-training gives them knowledge of words and grammar, so a few examples
per class are enough. Our model has to learn English *and* emotions from 60
sentences.

### How you would normally fight overfitting

| Fix | Idea |
|-----|------|
| More data | The real fix. Hundreds of sentences per class |
| Early stopping | Keep the weights from the epoch with the best **val** loss |
| More dropout / weight decay | Make memorising harder |
| Smaller model | Fewer parameters → less room to memorise |
| Pre-trained model | Start from a model that already knows the language |

## 6. The checkpoint

```
saved checkpoint.pt
```

It contains everything Step 8 needs to rebuild the model without
retraining:

```python
{"model": model.state_dict(),   # all learned weights
 "itos": tok.itos,              # the vocabulary (id → word)
 "config": CONFIG}              # d_model, n_heads, ... to rebuild the same shape
```

The vocab **must** be saved too. A different vocab would map words to
different ids, and the learned embeddings would be meaningless.
(`checkpoint.pt` is in `.gitignore`. Re-run `train.py` to recreate it.)

## Key takeaways

- One step = **forward → loss → zero_grad → backward → step**.
- Cross-entropy = `−ln(prob of the correct class)`. It starts near `ln(6)`
  and falls as the model improves.
- `.grad` is the direction that **increases** the loss. The optimizer
  moves the opposite way.
- Always keep a **validation set**. Train accuracy alone told us 100%; the
  truth was about 25%.
- Train loss ↓ while val loss ↑ means **overfitting**. Tiny data plus no
  pre-training makes it unavoidable here.

## Try it yourself

1. **Forget `zero_grad`:** comment out `optimizer.zero_grad()` in the loop.
   What happens to the training loss? Why?
2. **Learning rate:** try `LR = 3e-1` and `LR = 3e-5`. Which one explodes or
   jumps around, and which barely moves in 150 epochs?
3. **Early stopping:** track the best val loss and save the checkpoint only
   when it improves. At which epoch is the best model? Is its val accuracy
   any better?
4. **Fight overfitting:** set `dropout=0.3` and `weight_decay=0.1`. Does val
   loss still climb as fast?
5. **More data:** add 5 new sentences to each emotion in `data.py` (or add
   your "tired" class). Does val accuracy improve? (`len(EMOJIS)` updates the
   model automatically.)
6. **Predict the loss:** using the table in section 3, what probability does
   the model give the correct class when the train loss is 0.026?

**Next → Step 8: Inference + visualising attention** (load `checkpoint.pt`,
predict your own sentences, and see which words each head looks at).
