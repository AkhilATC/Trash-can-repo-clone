import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):

    def __init__(self,vocab_size,d_model,pad_index=1):
        super().__init__()
        self.embedding_lookup_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_index
        )
        self.embedding_dimention = d_model

    def forward(self,ids):
        return self.embedding_lookup_table(ids) * math.sqrt(self.embedding_dimention)

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

    # pad_id has no default on purpose: it must come from the tokenizer (tok.pad_id).
    # A wrong guess silently zeroes and freezes some other token's vector.
    def __init__(self, vocab_size, d_model, pad_id, max_len=64, dropout=0.1):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, d_model, pad_id)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids):
        return self.drop(self.pos(self.tok(ids)))

if __name__ == "__main__":
    from data_sets import data,texts
    from tokenizer import Tokenizer

    D_MODEL = 16
    MAX_LENGTH = 12
    tok = Tokenizer(texts)
    input_sentence = ["Howdy Aki"]
    # Keep the mask too: the attention layer needs it to ignore [PAD] positions.
    ids, mask = tok.encode_batch(input_sentence, MAX_LENGTH)
    embedding = Embeddings(tok.vocab_size, D_MODEL, pad_id=tok.pad_id, max_len=MAX_LENGTH)
    X = embedding(ids)
    print("ids :", ids.shape, ids)
    print("mask:", mask.shape, mask)
    print("X   :", X.shape) # (1,12,16)
    print(embedding.tok(ids))

