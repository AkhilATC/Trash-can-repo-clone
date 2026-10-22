"""
Step 7: Training. Make the loss go down.

One training step:

    logits = model(ids, mask)              forward pass            (B, n_classes)
    loss   = cross_entropy(logits, labels) how wrong are we?       one number
    loss.backward()                        backprop: d loss / d every weight -> .grad
    optimizer.step()                       nudge every weight against its gradient
    optimizer.zero_grad()                  clear .grad for the next step

Repeat over many shuffled mini-batches ("epochs"), and keep checking a
validation set the model never trains on, to see if it really generalises.
"""

import random

import torch
import torch.nn as nn

from data import EMOJIS, get_dataset
from model import EmotionTransformer, encode_batch
from tokenizer import Tokenizer

CONFIG = dict(d_model=16, n_heads=4, d_ff=64, n_layers=2, max_len=12, dropout=0.1)
LR, EPOCHS, BATCH_SIZE, VAL_PER_CLASS = 3e-3, 150, 16, 2
CHECKPOINT = "checkpoint.pt"


def split_data(data, val_per_class, seed=0):
    """Hold out `val_per_class` sentences of every class, so val has all emotions."""
    rng = random.Random(seed)
    train, val = [], []
    for label in range(len(EMOJIS)):
        items = [d for d in data if d[1] == label]
        rng.shuffle(items)
        val += items[:val_per_class]
        train += items[val_per_class:]
    return train, val


def to_tensors(tok, data, max_len):
    ids, mask = encode_batch(tok, [s for s, _ in data], max_len)
    labels = torch.tensor([y for _, y in data])
    return ids, mask, labels


@torch.no_grad()
def evaluate(model, ids, mask, labels):
    model.eval()                                   # dropout off
    logits = model(ids, mask)
    loss = nn.functional.cross_entropy(logits, labels)
    acc = (logits.argmax(-1) == labels).float().mean()
    return loss.item(), acc.item()


def train():
    torch.manual_seed(0)
    data = get_dataset()
    train_data, val_data = split_data(data, VAL_PER_CLASS)

    # Vocab from TRAINING sentences only: words seen only in val become [UNK], like in real life.
    tok = Tokenizer([s for s, _ in train_data])
    tr_ids, tr_mask, tr_y = to_tensors(tok, train_data, CONFIG["max_len"])
    va_ids, va_mask, va_y = to_tensors(tok, val_data, CONFIG["max_len"])
    print(f"train {len(train_data)} sentences, val {len(val_data)}, vocab {tok.vocab_size}")

    model = EmotionTransformer(tok.vocab_size, len(EMOJIS), **CONFIG)
    # AdamW: Adam (a per-weight adaptive step size) + weight decay (gently shrinks weights).
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # --- 1) One step in slow motion, on a single batch -------------------------------
    b_ids, b_mask, b_y = tr_ids[:BATCH_SIZE], tr_mask[:BATCH_SIZE], tr_y[:BATCH_SIZE]
    w = model.head[1].weight                        # watch one weight matrix
    model.train()
    logits = model(b_ids, b_mask)
    loss = nn.functional.cross_entropy(logits, b_y)
    print("\none step in slow motion:")
    print(f"  logits {tuple(logits.shape)}, labels {tuple(b_y.shape)} -> loss {loss.item():.3f}")
    print("  head weight .grad before backward():", w.grad)
    loss.backward()
    print("  head weight .grad after  backward():", tuple(w.grad.shape), "same shape as the weight")
    before = w.detach().clone()
    optimizer.step()
    print(f"  optimizer.step() changed the head weights by {(w.detach() - before).abs().mean():.4f} on average")
    optimizer.zero_grad()
    print(f"  loss on the same batch after one step: {evaluate(model, b_ids, b_mask, b_y)[0]:.3f}")

    # --- 2) The real loop --------------------------------------------------------------
    print(f"\n{'epoch':>5} {'train loss':>11} {'train acc':>10} {'val loss':>9} {'val acc':>8}")
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()                               # dropout on
        order = torch.randperm(len(tr_y))           # new shuffle every epoch
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(tr_ids[idx], tr_mask[idx])
            loss = nn.functional.cross_entropy(logits, tr_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tr = evaluate(model, tr_ids, tr_mask, tr_y)
        va = evaluate(model, va_ids, va_mask, va_y)
        history.append((epoch, *tr, *va))
        if epoch == 1 or epoch % 15 == 0:
            print(f"{epoch:5d} {tr[0]:11.3f} {tr[1]:10.2f} {va[0]:9.3f} {va[1]:8.2f}")

    # --- 3) Which validation sentences does it get right? ------------------------------
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(va_ids, va_mask), -1)
    print("\nvalidation sentences (never trained on):")
    for (s, y), p in zip(val_data, probs):
        pred = p.argmax().item()
        mark = "ok " if pred == y else "XX "
        words = " ".join(t for t in tok.decode(tok.encode(s, CONFIG["max_len"])[0]) if t not in ("[PAD]", "[CLS]"))
        print(f"  {mark} true {EMOJIS[y]}  pred {EMOJIS[pred]} {p[pred]:.2f}   {words}")

    # --- 4) Save everything step 8 needs to rebuild the model --------------------------
    torch.save({"model": model.state_dict(), "itos": tok.itos, "config": CONFIG}, CHECKPOINT)
    print(f"\nsaved {CHECKPOINT}")
    return history


if __name__ == "__main__":
    train()
