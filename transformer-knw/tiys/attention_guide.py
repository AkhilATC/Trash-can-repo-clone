"""

 Shapes (B=batch, T=12, d_model=16):
  X (B,12,16) ──W_q,W_k,W_v──► Q,K,V (B,12,16)
  scores = Q @ K.transpose(-2,-1) / sqrt(d_k)   → (B,12,12)
  scores.masked_fill(mask==0, -inf)              ← mask must
  broadcast!
  weights = softmax(scores, dim=-1)              → (B,12,12), each
  row sums to 1
  out = weights @ V                              → (B,12,16)

"""