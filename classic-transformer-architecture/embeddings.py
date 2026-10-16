"""
Step 2: Token embeddings + positional encoding.

    ids (B, T)  ──TokenEmbedding──►  (B, T, d_model)
                                         +
                PositionalEncoding   (1, T, d_model)
                                         =
                                     (B, T, d_model)   ← input to the encoder
"""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    A learnable lookup table: row i is the vector for token id i.
    Shape of the table: (vocab_size, d_model).
    """

    def __init__(self, vocab_size, d_model, pad_id=0):
        super().__init__()
        # padding_idx: the [PAD] row stays all zeros and never gets gradient updates.
        self.table = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.d_model = d_model

    def forward(self, ids):
        # (B, T) -> (B, T, d_model)
        # The paper scales by sqrt(d_model) so token vectors are not drowned
        # out by the positional encoding (whose values are in [-1, 1]).
        return self.table(ids) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (not learned) position vectors from "Attention Is All You Need":

        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    Each pair of dimensions is a sine/cosine wave with a different frequency.
    Low dims change fast with position, high dims change slowly. Together they
    give every position a unique "fingerprint".
    """

    def __init__(self, d_model, max_len=64):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)                 # (max_len, 1)
        i2 = torch.arange(0, d_model, 2)                         # 0, 2, 4, ...
        freq = torch.exp(-math.log(10000.0) * i2 / d_model)      # 1 / 10000^(2i/d)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * freq)                      # even dims
        pe[:, 1::2] = torch.cos(pos * freq)                      # odd dims

        # A buffer is saved with the model and moved by .to(device),
        # but it is NOT a parameter, so the optimizer never changes it.
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model). Add the first T position vectors to every sentence.
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)                      # (1, T, d) broadcasts over B


class Embeddings(nn.Module):
    """Token embedding + positional encoding + dropout. This feeds the encoder."""

    def __init__(self, vocab_size, d_model, max_len=64, pad_id=0, dropout=0.1):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, d_model, pad_id)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids):
        return self.drop(self.pos(self.tok(ids)))


if __name__ == "__main__":
    from data import get_dataset
    from tokenizer import Tokenizer

    torch.manual_seed(0)
    D_MODEL, MAX_LEN = 16, 12

    data = get_dataset()
    tok = Tokenizer([s for s, _ in data])

    sentences = ["i love my family", "my family love i"]
    ids = torch.tensor([tok.encode(s, MAX_LEN)[0] for s in sentences])
    print("ids shape:", tuple(ids.shape), "  (B, T)")

    emb = Embeddings(tok.vocab_size, D_MODEL, max_len=MAX_LEN, dropout=0.0)
    x = emb(ids)
    print("out shape:", tuple(x.shape), "  (B, T, d_model)")
    print("learnable params:", sum(p.numel() for p in emb.parameters()),
          f"= vocab {tok.vocab_size} x d_model {D_MODEL}")

    # 1) Without positions, both sentences are the same bag of vectors.
    t = emb.tok(ids)
    same_bag = torch.allclose(t[0, 1:5].sum(0), t[1, 1:5].sum(0), atol=1e-5)
    print("\n[no positions] same set of word vectors in both sentences? ", same_bag)

    # 2) With positions, the word 'love' gets a different vector at each position.
    love_a = x[0, 2]   # 'love' at position 2 in sentence 0
    love_b = x[1, 3]   # 'love' at position 3 in sentence 1
    print("[with positions] 'love' @pos2 == 'love' @pos3 ?            ",
          torch.allclose(love_a, love_b))

    # 3) The positional fingerprints: nearby positions look more alike.
    pe = emb.pos.pe
    cos = nn.functional.cosine_similarity
    print("\ncosine similarity of position vectors (1.0 = identical):")
    for a, b in [(0, 1), (0, 2), (0, 5), (0, 10)]:
        print(f"  pos {a} vs pos {b:>2}: {cos(pe[a], pe[b], dim=0):.3f}")

    print("\nPE values for positions 0..3, first 6 dims:")
    torch.set_printoptions(precision=2, sci_mode=False)
    print(pe[:4, :6])
