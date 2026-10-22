"""
Step 6: The full model. Everything from steps 1-5 plus a classifier head.

    ids  (B, T)                      [CLS] i love my family [PAD] ...
      ▼  Embeddings (step 2)
    x    (B, T, d_model)
      ▼  Encoder: N x EncoderBlock (steps 3-5)
    h    (B, T, d_model)             every token now "knows" the whole sentence
      ▼  take position 0             the [CLS] token
    cls  (B, d_model)                one summary vector per sentence
      ▼  Linear(d_model, n_classes)
    logits (B, n_classes)            one raw score per emoji
      ▼  softmax (only when we want probabilities)
    probs  (B, n_classes)            sums to 1
"""

import torch
import torch.nn as nn

from embeddings import Embeddings
from encoder import Encoder


class EmotionTransformer(nn.Module):
    def __init__(self, vocab_size, n_classes, d_model=16, n_heads=4, d_ff=64,
                 n_layers=2, max_len=12, dropout=0.1, pad_id=0):
        super().__init__()
        self.embed = Embeddings(vocab_size, d_model, max_len, pad_id, dropout)
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, ids, mask):
        x = self.embed(ids)            # (B, T) -> (B, T, d_model)
        h = self.encoder(x, mask)      # (B, T, d_model)
        cls = h[:, 0]                  # (B, d_model): [CLS] is always at position 0
        return self.head(cls)          # (B, n_classes) raw scores = "logits"


def encode_batch(tok, sentences, max_len):
    """List of strings -> ids (B, T) and mask (B, T) tensors."""
    enc = [tok.encode(s, max_len) for s in sentences]
    return torch.tensor([e[0] for e in enc]), torch.tensor([e[1] for e in enc])


if __name__ == "__main__":
    import math

    from data import EMOJIS, get_dataset
    from tokenizer import Tokenizer

    torch.manual_seed(0)
    torch.set_printoptions(precision=2, sci_mode=False)
    torch.set_grad_enabled(False)   # demo only: no training yet
    MAX_LEN, D_MODEL = 12, 16

    data = get_dataset()
    tok = Tokenizer([s for s, _ in data])
    model = EmotionTransformer(tok.vocab_size, len(EMOJIS), d_model=D_MODEL, max_len=MAX_LEN)
    model.eval()                    # turn dropout off so results are repeatable

    sentences = ["i love my family", "he lied to me again"]
    ids, mask = encode_batch(tok, sentences, MAX_LEN)

    # 1) Shapes through the whole pipeline.
    x = model.embed(ids)
    h = model.encoder(x, mask)
    logits = model(ids, mask)
    probs = torch.softmax(logits, dim=-1)
    print("ids    :", tuple(ids.shape), "     (B, T)")
    print("x      :", tuple(x.shape), " (B, T, d_model)  after embeddings")
    print("h      :", tuple(h.shape), " (B, T, d_model)  after encoder")
    print("cls    :", tuple(h[:, 0].shape), "     (B, d_model)     h[:, 0]")
    print("logits :", tuple(logits.shape), "      (B, n_classes)")
    print("probs  :", tuple(probs.shape), "      (B, n_classes)  row sums:", probs.sum(-1))

    # 2) Where the parameters are.
    def count(m):
        return sum(p.numel() for p in m.parameters())
    print("\nparams:")
    print(f"  embeddings : {count(model.embed):5d}  = vocab {tok.vocab_size} x {D_MODEL}")
    print(f"  encoder    : {count(model.encoder):5d}  = 2 blocks x {count(model.encoder) // 2}")
    print(f"  head       : {count(model.head):5d}  = {D_MODEL} x {len(EMOJIS)} + {len(EMOJIS)}")
    print(f"  total      : {count(model):5d}")

    # 3) Untrained predictions: basically guessing.
    print("\nuntrained predictions:")
    print("  " + " " * 22 + "  ".join(f"{e:>4}" for e in EMOJIS))
    for s, p in zip(sentences, probs):
        print(f"  {s:<22}" + "  ".join(f"{v:5.2f}" for v in p.tolist())
              + f"   -> {EMOJIS[p.argmax()]}")

    # 4) Loss on the whole dataset before training. Random guessing gives ln(n_classes).
    all_ids, all_mask = encode_batch(tok, [s for s, _ in data], MAX_LEN)
    labels = torch.tensor([y for _, y in data])
    all_logits = model(all_ids, all_mask)
    loss = nn.functional.cross_entropy(all_logits, labels)
    acc = (all_logits.argmax(-1) == labels).float().mean()
    print(f"\nwhole dataset ({len(data)} sentences) before training:")
    print(f"  loss     = {loss:.3f}   (random guessing = ln({len(EMOJIS)}) = {math.log(len(EMOJIS)):.3f})")
    print(f"  accuracy = {acc:.2f}    (random guessing = 1/{len(EMOJIS)} = {1 / len(EMOJIS):.2f})")

    # 5) [CLS] has no meaning of its own, but after the encoder it depends on every word.
    ids2, mask2 = encode_batch(tok, ["i love my dog"], MAX_LEN)
    cls_a = model.encoder(model.embed(ids[:1]), mask[:1])[0, 0]
    cls_b = model.encoder(model.embed(ids2), mask2)[0, 0]
    print("\n[CLS] vector, 'i love my family' vs 'i love my dog'   tokens:", tok.decode(ids2[0, :5].tolist()))
    print("  same input at position 0 ([CLS])  :", bool(ids[0, 0] == ids2[0, 0]))
    print("  [CLS] output changed by           :", f"{(cls_a - cls_b).abs().sum():.2f}")

    # 6) train() vs eval(): dropout makes train-mode outputs random.
    model.train()
    a, b = model(ids, mask)[0], model(ids, mask)[0]
    print("\ntrain mode, same input twice, same logits?", torch.allclose(a, b))
    model.eval()
    a, b = model(ids, mask)[0], model(ids, mask)[0]
    print("eval mode,  same input twice, same logits?", torch.allclose(a, b))
