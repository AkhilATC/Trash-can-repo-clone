"""
Step 8: Inference + looking inside the attention.

    checkpoint.pt ──► rebuild Tokenizer (same vocab) + EmotionTransformer (same config)
                  ──► load the trained weights
    "your sentence" ──► ids, mask ──► model.eval() ──► logits ──► softmax ──► emoji

Every MultiHeadAttention layer keeps its last attention weights in `.weights`
(B, h, T, T). Row 0 of each head = where [CLS] looked. [CLS] is the vector
the classifier reads, so that row shows which words fed the decision.

Run: python predict.py                       (example sentences)
     python predict.py "i won the lottery"   (your own)
"""

import sys

import torch

from data import EMOJIS, EMOTION_NAMES
from model import EmotionTransformer, encode_batch
from tokenizer import Tokenizer


def load(path="checkpoint.pt"):
    ckpt = torch.load(path)
    tok = Tokenizer.from_vocab(ckpt["itos"])
    model = EmotionTransformer(tok.vocab_size, len(EMOJIS), **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()                                    # dropout off: same input -> same answer
    return model, tok, ckpt["config"]["max_len"]


@torch.no_grad()
def predict(model, tok, max_len, sentences):
    ids, mask = encode_batch(tok, sentences, max_len)
    probs = torch.softmax(model(ids, mask), dim=-1)     # (B, n_classes)
    # One (B, h, T, T) map per encoder block, filled during the forward pass above.
    attn = [block.attn.weights for block in model.encoder.layers]
    return ids, mask, probs, attn


@torch.no_grad()
def word_importance(model, tok, max_len, sentence):
    """
    Remove one word at a time and see how much the predicted emoji's probability drops.
    A big drop = the model really relied on that word. (Called "occlusion".)
    """
    words = sentence.split()
    _, _, probs, _ = predict(model, tok, max_len, [sentence])
    c = probs[0].argmax().item()
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    _, _, p_without, _ = predict(model, tok, max_len, variants)
    return words, c, probs[0, c].item() - p_without[:, c]


def bar(v, width=20):
    return "█" * round(v * width)


def show(tok, sentence, ids, mask, probs, attn, b):
    n = int(mask[b].sum())                              # real tokens only
    words = tok.decode(ids[b, :n].tolist())
    top = probs[b].topk(3)
    print(f"\n{sentence!r}")
    print("  tokens:", " ".join(words))
    for p, c in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {EMOJIS[c]} {EMOTION_NAMES[c]:<8} {p:5.2f} {bar(p)}")

    # Where did [CLS] look? Last layer, every head, real tokens only.
    last = attn[-1][b, :, 0, :n]                        # (h, n)
    print("  [CLS] attention in the last layer (row = head):")
    print("          " + "".join(f"{w[:7]:>8}" for w in words))
    for h, row in enumerate(last):
        print(f"  head {h}  " + "".join(f"{v:8.2f}" for v in row.tolist()))
    avg = last.mean(0)
    print("  average " + "".join(f"{v:8.2f}" for v in avg.tolist()))
    print("  strongest word for [CLS] (avg of heads):", words[avg[1:].argmax() + 1])


def show_importance(model, tok, max_len, sentence):
    words, c, drop = word_importance(model, tok, max_len, sentence)
    print(f"  remove one word -> drop in p({EMOJIS[c]}):")
    for w, d in zip(words, drop.tolist()):
        print(f"    {w:<10} {d:+.2f} {bar(max(d, 0))}")


if __name__ == "__main__":
    model, tok, max_len = load()
    sentences = sys.argv[1:] or [
        "i passed my exam",                 # in the training set
        "i love you so much",               # validation set: never trained on
        "my dog is so happy today",         # new
        "i am scared of the dark",          # new
        "wow i never expected that",        # new
    ]
    ids, mask, probs, attn = predict(model, tok, max_len, sentences)
    print("attention maps:", [tuple(a.shape) for a in attn], " one (B, h, T, T) per layer")
    for b, s in enumerate(sentences):
        show(tok, s, ids, mask, probs, attn, b)
        show_importance(model, tok, max_len, s)
git