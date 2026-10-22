"""
Step 5: One encoder block = multi-head attention + feed-forward, each wrapped
in a residual connection and LayerNorm. Then stack N of them.

    x (B, T, d_model)
      │
      ├──────────────┐
      ▼              │
    MultiHeadAttention   tokens talk to each other (mixes ACROSS tokens)
      ▼              │
    dropout          │
      ▼              │
      + ◄────────────┘   residual: x + attn(x)
      ▼
    LayerNorm
      │
      ├──────────────┐
      ▼              │
    FeedForward          each token thinks on its own (NO mixing across tokens)
      ▼              │
    dropout          │
      ▼              │
      + ◄────────────┘   residual: x + ffn(x)
      ▼
    LayerNorm
      ▼
    out (B, T, d_model)    same shape as x, so blocks can be stacked
"""

import torch
import torch.nn as nn

from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Two linear layers with a ReLU in between, applied to every token separately.
    d_model -> d_ff (wider, usually 4 x d_model) -> d_model
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # nn.Linear only acts on the LAST dim, so T is treated like a batch:
        # token 3's output depends on token 3's input only.
        return self.net(x)   # (B, T, d_model) -> (B, T, d_ff) -> (B, T, d_model)


class EncoderBlock(nn.Module):
    """
    norm_first=False: post-norm, as in "Attention Is All You Need":  x = norm(x + sublayer(x))
    norm_first=True : pre-norm, used by most modern models:          x = x + sublayer(norm(x))
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_first=False):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, mask=None):
        if self.norm_first:
            x = x + self.drop1(self.attn(self.norm1(x), mask))
            x = x + self.drop2(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop1(self.attn(x, mask)))
            x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class Encoder(nn.Module):
    """N identical blocks (each with its OWN weights), applied one after another."""

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1, norm_first=False):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderBlock(d_model, n_heads, d_ff, dropout, norm_first) for _ in range(n_layers)
        )
        # Pre-norm leaves the last output un-normalised, so it needs one final LayerNorm.
        self.final_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)   # (B, T, d_model) in, (B, T, d_model) out
        return self.final_norm(x)


if __name__ == "__main__":
    from attention import setup

    D_MODEL, N_HEADS, D_FF, N_LAYERS = 16, 4, 64, 2
    tok, sentences, ids, mask, x = setup(d_model=D_MODEL)
    torch.set_grad_enabled(False)   # demo only: no training here, keeps the printout clean
    block = EncoderBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)

    # 1) Shapes: everything comes back as (B, T, d_model).
    a = block.attn(x, mask)
    f = block.ffn.net[0](x)
    out = block(x, mask)
    print("x            :", tuple(x.shape), "  (B, T, d_model)")
    print("attention    :", tuple(a.shape), "  (B, T, d_model)")
    print("ffn hidden   :", tuple(f.shape), "  (B, T, d_ff)  wider on the inside")
    print("block output :", tuple(out.shape), "  (B, T, d_model)  same as x")

    # 2) Where the parameters are.
    def count(m):
        return sum(p.numel() for p in m.parameters())
    print("\nparams in one block:")
    print(f"  attention : {count(block.attn):5d}  = 4 x {D_MODEL} x {D_MODEL}")
    print(f"  ffn       : {count(block.ffn):5d}  = ({D_MODEL} x {D_FF} + {D_FF}) + ({D_FF} x {D_MODEL} + {D_MODEL})")
    print(f"  2 x norm  : {count(block.norm1) + count(block.norm2):5d}  = 2 x (gamma {D_MODEL} + beta {D_MODEL})")
    print(f"  total     : {count(block):5d}")

    # 3) LayerNorm: after it, every token vector has mean ~0 and std ~1.
    torch.set_printoptions(precision=2, sci_mode=False)
    print("\nper-token mean / std of 'i love my family' (first 5 tokens)")
    print("  input x : mean", x[0, :5].mean(-1), " std", x[0, :5].std(-1, unbiased=False))
    print("  output  : mean", out[0, :5].mean(-1), " std", out[0, :5].std(-1, unbiased=False))

    # 4) FFN works on each token alone: change token 2, only row 2 of the FFN output changes.
    x2 = x.clone()
    x2[0, 2] += 1.0
    diff = (block.ffn(x2) - block.ffn(x))[0].abs().sum(-1)
    print("\nchange token 2 only. How much does each token's output change?")
    print("  ffn       :", diff[:5])
    diff = (block.attn(x2, mask) - block.attn(x, mask))[0].abs().sum(-1)
    print("  attention :", diff[:5], " (all tokens change: attention mixes)")

    # 5) The mask keeps padding out: changing [PAD] vectors does not touch real tokens.
    x3 = x.clone()
    x3[0, mask[0] == 0] = torch.randn_like(x3[0, mask[0] == 0]) * 100
    n_real = int(mask[0].sum())
    same = torch.allclose(block(x3, mask)[0, :n_real], out[0, :n_real], atol=1e-5)
    print("\nreal-token outputs unchanged when [PAD] vectors are garbage:", same)

    # 6) Stack N blocks. The shape never changes, so we can stack as many as we like.
    enc = Encoder(N_LAYERS, D_MODEL, N_HEADS, D_FF, dropout=0.0)
    print(f"\nencoder with {N_LAYERS} blocks: {tuple(enc(x, mask).shape)}, params {count(enc)}"
          f" = {N_LAYERS} x {count(block)}")

    # 7) Same result as PyTorch's nn.TransformerEncoderLayer, given the same weights.
    ref = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0, batch_first=True)
    ref.eval()
    m = block.attn
    ref.self_attn.in_proj_weight.copy_(torch.cat([m.W_q.weight, m.W_k.weight, m.W_v.weight]))
    ref.self_attn.in_proj_bias.zero_()           # our attention has no biases
    ref.self_attn.out_proj.weight.copy_(m.W_o.weight)
    ref.self_attn.out_proj.bias.zero_()
    ref.linear1.load_state_dict(block.ffn.net[0].state_dict())
    ref.linear2.load_state_dict(block.ffn.net[3].state_dict())
    ref.norm1.load_state_dict(block.norm1.state_dict())
    ref.norm2.load_state_dict(block.norm2.state_dict())
    ref_out = ref(x, src_key_padding_mask=(mask == 0))   # True = ignore
    real = mask.bool()   # torch's fast path may zero the [PAD] rows, so compare real tokens only
    print("matches torch's nn.TransformerEncoderLayer:", torch.allclose(out[real], ref_out[real], atol=1e-5))
