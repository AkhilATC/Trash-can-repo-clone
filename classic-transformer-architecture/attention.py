"""
Step 3: Single-head self-attention with Query, Key, Value.

    x (B, T, d_model)
      ├─ W_q ─► Q (B, T, d_k)   "what am I looking for?"
      ├─ W_k ─► K (B, T, d_k)   "what do I contain?"     (the label on the box)
      └─ W_v ─► V (B, T, d_k)   "what do I hand over?"   (the content of the box)

    scores  = Q @ K^T / sqrt(d_k)          (B, T, T)   how well each query matches each key
    scores  = masked where key is [PAD]    (-inf → softmax gives 0)
    weights = softmax(scores, dim=-1)      (B, T, T)   each row sums to 1
    out     = weights @ V                  (B, T, d_k) a weighted mix of the values

Step 4: Multi-head attention = run h smaller heads in parallel, then glue them back.

    x (B, T, d_model)
      ├─ W_q, W_k, W_v ─► (B, T, d_model) ─split─► (B, h, T, d_k)   d_k = d_model / h
      ├─ attention in every head at once        ─► (B, h, T, d_k)
      ├─ concat heads                           ─► (B, T, d_model)
      └─ W_o (mix the heads)                    ─► (B, T, d_model)

Run: python attention.py 3   (step 3 demo)
     python attention.py     (step 4 demo)
"""

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v : (B, T, d_k), or (B, h, T, d_k) for multi-head. @ treats every
              leading dim as a batch dim, so the same code works for both.
    mask    : (B, T) with 1 = real token, 0 = padding. Masks the KEYS,
              so no token can take information from a [PAD] position.
              For multi-head, pass (B, 1, T) so it broadcasts over the heads.
    returns : out (..., T, d_k), weights (..., T, T)
    """
    d_k = q.size(-1)

    # (B, T, d_k) @ (B, d_k, T) -> (B, T, T). Row i = how much token i "matches" every token j.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        # (B, T) -> (B, 1, T): the same key mask is used for every query row.
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)  # normalise over the keys (last dim)
    out = weights @ v                        # (B, T, T) @ (B, T, d_k) -> (B, T, d_k)
    return out, weights


class SelfAttention(nn.Module):
    """One attention head. Step 4 runs several of these in parallel."""

    def __init__(self, d_model, d_k):
        super().__init__()
        # Three learnable projections of the SAME input. That is why it is called *self*-attention.
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x, mask=None):
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)   # each (B, T, d_k)
        return scaled_dot_product_attention(q, k, v, mask)


class MultiHeadAttention(nn.Module):
    """
    h heads, each of size d_k = d_model // h. Same total size as one big head,
    but each head gets its own Q/K/V subspace, so it can learn its own pattern.

    We do NOT build h separate SelfAttention modules. One (d_model x d_model)
    matrix is the same as h (d_model x d_k) matrices side by side, so we
    project once and then split the result into heads. Same maths, faster.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # mixes the heads' outputs
        self.weights = None  # last attention weights (B, h, T, T), kept for step 8

    def split_heads(self, t):
        # (B, T, d_model) -> (B, T, h, d_k) -> (B, h, T, d_k)
        B, T, _ = t.shape
        return t.view(B, T, self.h, self.d_k).transpose(1, 2)

    def merge_heads(self, t):
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d_model)   ("concat")
        B, _, T, _ = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)

    def forward(self, x, mask=None):
        q = self.split_heads(self.W_q(x))   # (B, h, T, d_k)
        k = self.split_heads(self.W_k(x))
        v = self.split_heads(self.W_v(x))
        if mask is not None:
            mask = mask.unsqueeze(1)        # (B, T) -> (B, 1, T): same mask for every head
        out, self.weights = scaled_dot_product_attention(q, k, v, mask)
        return self.W_o(self.merge_heads(out))   # (B, T, d_model)


def setup(max_len=12, d_model=16):
    """Shared demo input: two sentences -> embeddings x (B, T, d_model) + mask (B, T)."""
    from data import get_dataset
    from embeddings import Embeddings
    from tokenizer import Tokenizer

    torch.manual_seed(0)
    torch.set_printoptions(precision=2, sci_mode=False)
    data = get_dataset()
    tok = Tokenizer([s for s, _ in data])
    sentences = ["i love my family", "he lied to me again"]
    enc = [tok.encode(s, max_len) for s in sentences]
    ids = torch.tensor([e[0] for e in enc])
    mask = torch.tensor([e[1] for e in enc])
    emb = Embeddings(tok.vocab_size, d_model, max_len=max_len, dropout=0.0)
    return tok, sentences, ids, mask, emb(ids)


def demo_single_head():
    D_MODEL, D_K = 16, 8
    tok, sentences, ids, mask, x = setup(d_model=D_MODEL)
    attn = SelfAttention(D_MODEL, D_K)
    out, w = attn(x, mask)
    q = attn.W_q(x)
    print("x       :", tuple(x.shape), "  (B, T, d_model)")
    print("Q, K, V :", tuple(q.shape), "   (B, T, d_k)  each")
    print("weights :", tuple(w.shape), "   (B, T, T)")
    print("out     :", tuple(out.shape), "   (B, T, d_k)")
    print("params  :", sum(p.numel() for p in attn.parameters()),
          f"= 3 x d_model {D_MODEL} x d_k {D_K}")

    # 1) The attention matrix for sentence 0 (untrained, so the pattern is random-ish).
    words = tok.decode(ids[0].tolist())
    print(f"\nattention weights for {sentences[0]!r}  (row = query, col = key)")
    print("          " + "".join(f"{t[:6]:>7}" for t in words))
    for i, t in enumerate(words):
        print(f"{t[:8]:>9} " + "".join(f"{v:7.2f}" for v in w[0, i].tolist()))

    # 2) Every row is a probability distribution over the keys.
    print("\nrow sums (should all be 1):", w[0].sum(-1))

    # 3) The mask works: no attention goes TO a [PAD] key.
    pad_cols = mask[0] == 0
    print("total weight on [PAD] keys:", w[0][:, pad_cols].sum().item())

    # 4) Why divide by sqrt(d_k)? Big dot products push softmax to one-hot.
    big_k = 512
    a, b = torch.randn(6, big_k), torch.randn(6, big_k)
    raw = a @ b.T
    print(f"\nwith d_k={big_k}: std of raw scores = {raw.std():.1f}, "
          f"scaled = {(raw / math.sqrt(big_k)).std():.1f}")
    print("softmax of one row, raw    :", torch.softmax(raw[0], -1))
    print("softmax of one row, scaled :", torch.softmax(raw[0] / math.sqrt(big_k), -1))

    # 5) Same result as PyTorch's built-in (which also does the masking for us).
    qq, kk, vv = attn.W_q(x), attn.W_k(x), attn.W_v(x)
    ref = nn.functional.scaled_dot_product_attention(
        qq, kk, vv, attn_mask=mask.bool().unsqueeze(1))
    print("\nmatches torch's F.scaled_dot_product_attention:", torch.allclose(out, ref, atol=1e-6))


def demo_multi_head():
    D_MODEL, N_HEADS = 16, 4
    tok, sentences, ids, mask, x = setup(d_model=D_MODEL)
    mha = MultiHeadAttention(D_MODEL, N_HEADS)

    # 1) Follow the shapes through every stage.
    q = mha.W_q(x)
    qh = mha.split_heads(q)
    out = mha(x, mask)
    w = mha.weights
    print("x              :", tuple(x.shape), "   (B, T, d_model)")
    print("x @ W_q        :", tuple(q.shape), "   (B, T, d_model)")
    print("split into h   :", tuple(qh.shape), " (B, h, T, d_k)  d_k =", mha.d_k)
    print("weights        :", tuple(w.shape), "(B, h, T, T)  one T x T map per head")
    print("merged heads   :", tuple(mha.merge_heads(qh).shape), "   (B, T, d_model)")
    print("out (after W_o):", tuple(out.shape), "   (B, T, d_model)  same as x")

    # 2) Parameters: 4 square matrices.
    n = sum(p.numel() for p in mha.parameters())
    print(f"\nparams: {n} = 4 x {D_MODEL} x {D_MODEL}  (W_q, W_k, W_v, W_o)")

    # 3) What split_heads really does: it just cuts the d_model numbers into h slices.
    t = torch.arange(16.0).view(1, 1, 16)         # one token, values 0..15
    print("\none token's 16 numbers:", t[0, 0].int().tolist())
    for i, piece in enumerate(mha.split_heads(t)[0, :, 0]):
        print(f"  head {i} gets:", piece.int().tolist())

    # 4) Each head looks at the sentence differently.
    words = tok.decode(ids[0].tolist())
    n_real = int(mask[0].sum())
    qi = words.index("love")
    print(f"\nwhere does 'love' look, per head?  ({sentences[0]!r}, real tokens only)")
    print("        " + "".join(f"{t:>8}" for t in words[:n_real]))
    for hd in range(N_HEADS):
        row = w[0, hd, qi, :n_real].tolist()
        print(f"head {hd} " + "".join(f"{v:8.2f}" for v in row))

    # 5) Mask still works in every head.
    pad_cols = mask[0] == 0
    print("\ntotal weight on [PAD] keys, all heads:", w[0][..., pad_cols].sum().item())

    # 6) Same result as PyTorch's nn.MultiheadAttention, given the same weights.
    ref = nn.MultiheadAttention(D_MODEL, N_HEADS, bias=False, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mha.W_q.weight, mha.W_k.weight, mha.W_v.weight]))
        ref.out_proj.weight.copy_(mha.W_o.weight)
    ref_out, _ = ref(x, x, x, key_padding_mask=(mask == 0))   # True = ignore this key
    print("matches torch's nn.MultiheadAttention:", torch.allclose(out, ref_out, atol=1e-6))


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["3"]:
        demo_single_head()
    else:
        demo_multi_head()
