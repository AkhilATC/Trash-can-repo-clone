import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v, mask):
    """
    The core of attention. Pure math, no learnable weights, so it lives
    outside the class; every head of multi-head attention will reuse it.

        scores  = Q · Kᵀ / √dk               (B, T, T)
        scores  = -inf where key is [PAD]
        weights = softmax(scores, dim=-1)    each row sums to 1
        out     = weights · V                (B, T, T) @ (B, T, dk) -> (B, T, dk)

    :param q, k, v: (B, T, dk)
    :param mask: (B, T), 1 = real token, 0 = [PAD]
    :return: out (B, T, dk), weights (B, T, T)
    """
    dk = q.size(-1)

    # row i, column j = how well token i's query matches token j's key
    # see why_we_need_scale_on_attention_score.md for the √dk
    scores = q @ k.transpose(-2, -1) / math.sqrt(dk)

    # unsqueeze(1) -> (B, 1, T) so it broadcasts down every query row:
    # we block PAD *keys* (columns), every query sees the same columns.
    # -inf because exp(-inf) = 0 -> softmax gives PAD exactly zero weight.
    # see how_mask_is_applied_to_attention_scores.md
    scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

    weights = torch.softmax(scores, dim=-1)

    # each output row = weighted average of the V rows,
    # out[i] = sum_j weights[i, j] * v[j]   (PAD rows of V get weight 0)
    out = weights @ v
    return out, weights


class SelfAttention(nn.Module):

    def __init__(self,d_model,num_head=0):
        """
        Learnable projections that give every token three roles:
            Q = X · W_q   "what am I looking for?"
            K = X · W_k   "what do I contain?"
            V = X · W_v   "what do I hand over if attended to?"
        X: (B, T, d_model)  ->  Q, K, V: (B, T, dk)

        Why bias=False (the paper writes Q = X·W_q, with no bias):
          - b_k is useless: it adds q·b_k, the SAME number, to every score in
            a query's row, and softmax ignores a constant added to a whole row.
          - b_v is redundant: the attention weights sum to 1, so b_v comes out
            unchanged as a constant shift that the output layer W_o can learn.
          - b_q is the only one with an effect (a fixed preference for some
            keys), and it is small enough to drop.
          Many libraries (BERT, nn.MultiheadAttention) keep bias=True; both work.

        nn.Linear stores weight as (out, in) = (dk, d_model) and computes x @ weight.T

        :param d_model: model dimension = 16
        """
        super().__init__()
        if num_head:
            assert d_model % num_head == 0, f"d_model={d_model} must be divisible by num_head={num_head}"
        dk = d_model//num_head if num_head else d_model
        self.W_q = nn.Linear(d_model,dk,bias=False)
        self.W_k = nn.Linear(d_model,dk,bias=False)
        self.W_v = nn.Linear(d_model,dk,bias=False)
        self.dk = dk

    def forward(self,X,mask):
        q,k,v = self.W_q(X), self.W_k(X), self.W_v(X)
        print("🦾 | Attention layer shapes")
        print("=============================")
        print("Q SHAPE ",q.shape)
        print("K SHAPE ",k.shape)
        print("V SHAPE ",v.shape)
        print("=============================")

        # Q SHAPE  torch.Size([1, 12, 16])
        # K SHAPE  torch.Size([1, 12, 16])
        # V SHAPE  torch.Size([1, 12, 16])

        out, weights = scaled_dot_product_attention(q, k, v, mask)
        print("WEIGHTS SHAPE", weights.shape)
        print("OUT SHAPE    ", out.shape)

        # out: (B, T, dk), weights: (B, T, T) returned too so we can inspect them
        return out, weights


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        """
        h heads, each one a SelfAttention with dk = d_model / h, run in parallel.

        Instead of h separate (d_model -> dk) Linears per Q/K/V we use ONE
        (d_model -> d_model) Linear and cut its output into h slices of dk:
        the weight is just the h small weights stacked, so it is the same thing,
        but one big matmul is faster than h small ones.

            X (B,T,d_model) -> W_q -> (B,T,d_model) -> split -> (B,h,T,dk)
            attention per head        -> (B,h,T,dk)
            join heads                -> (B,T,d_model)
            W_o mixes the heads       -> (B,T,d_model)

        see multihead_attention.md

        :param d_model: model dimension = 16
        :param num_heads: number of heads, must divide d_model (16 / 4 -> dk = 4)
        """
        super().__init__()
        assert d_model % num_heads == 0, f"d_model={d_model} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # W_o: after joining, each output number can mix information from all heads
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        # (B, T, d_model) -> (B, T, h, dk) -> (B, h, T, dk)
        # transpose so the head dim sits next to B: the attention matmuls
        # only touch the last two dims (T, dk), so every head runs at once.
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.dk).transpose(1, 2)

    def join_heads(self, x):
        # (B, h, T, dk) -> (B, T, h, dk) -> (B, T, d_model)
        # contiguous() because view needs the memory in order after transpose
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.dk)

    def forward(self, X, mask):
        q = self.split_heads(self.W_q(X))
        k = self.split_heads(self.W_k(X))
        v = self.split_heads(self.W_v(X))
        print("🐙 | Multi-head attention shapes")
        print("=============================")
        print("Q SHAPE (B,h,T,dk)", q.shape)

        # mask (B, T) -> (B, 1, T); the function adds one more dim -> (B, 1, 1, T)
        # so it broadcasts over every head (dim 1) and every query row (dim 2)
        out, weights = scaled_dot_product_attention(q, k, v, mask.unsqueeze(1))
        print("WEIGHTS SHAPE     ", weights.shape)
        print("HEADS OUT SHAPE   ", out.shape)

        out = self.W_o(self.join_heads(out))
        print("OUT SHAPE         ", out.shape)
        print("=============================")

        # out: (B, T, d_model), weights: (B, h, T, T)
        return out, weights


if __name__ == "__main__":
    from data_sets import data, texts
    from tokenizer import Tokenizer
    from embeddings import Embeddings

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
    print("X   :", X.shape)  # (1,12,16)
    print(embedding.tok(ids))
    attn = SelfAttention(D_MODEL)
    out, weights = attn(X, mask)
    torch.set_printoptions(precision=3, sci_mode=False)
    print("row 0 ([CLS] query):", weights[0, 0])
    print("row sums:", weights.sum(-1))

    # check against PyTorch's own version (attn_mask: True = allowed to attend)
    q, k, v = attn.W_q(X), attn.W_k(X), attn.W_v(X)
    ref = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1).bool())
    print("matches PyTorch:", torch.allclose(out, ref, atol=1e-6))

    # ---- multi-head ----
    mha = MultiHeadAttention(D_MODEL, num_heads=4)
    out, weights = mha(X, mask)
    print("head 0, row 0:", weights[0, 0, 0])
    print("head 1, row 0:", weights[0, 1, 0])

    # check against nn.MultiheadAttention with the same weights
    # (it packs W_q, W_k, W_v into one in_proj_weight; key_padding_mask: True = ignore)
    ref_mha = nn.MultiheadAttention(D_MODEL, 4, bias=False, batch_first=True)
    with torch.no_grad():
        ref_mha.in_proj_weight.copy_(torch.cat([mha.W_q.weight, mha.W_k.weight, mha.W_v.weight]))
        ref_mha.out_proj.weight.copy_(mha.W_o.weight)
    ref, _ = ref_mha(X, X, X, key_padding_mask=(mask == 0))
    print("matches nn.MultiheadAttention:", torch.allclose(out, ref, atol=1e-6))
