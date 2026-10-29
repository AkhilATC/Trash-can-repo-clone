### We divide attention scores by √dk so softmax stays soft and gradients keep flowing.

```text
scores = Q · Kᵀ / √dk          (B, T, dk) @ (B, dk, T) -> (B, T, T)
```

Problem
```text
A score is a dot product: q·k = q1·k1 + q2·k2 + ... + q_dk·k_dk  (dk terms).

If each q_i, k_i has mean 0 and variance 1, each product q_i·k_i has variance 1,
and adding dk of them gives variance dk  ->  std = √dk.
So the bigger dk is, the more widely the raw scores spread out.

Measured (10,000 random q, k vectors, torch.manual_seed(0)):

  dk  | std(q·k) raw | std after / √dk
   4  |     1.99     |      1.00
  16  |     3.99     |      1.00      <- our single head (d_model = 16)
  64  |     7.97     |      1.00
 256  |    16.08     |      1.01
```

Why large scores hurt: softmax saturates
```text
Same 5 scores, two sizes of dk (raw = unit scores × √dk):

dk = 16
  raw scores          : [6.16, -1.17, -8.72, 2.27, -4.34]
  softmax(raw)        : [0.979, 0.001, 0.0,   0.02,  0.0  ]   <- almost one-hot
  softmax(raw / √dk)  : [0.612, 0.098, 0.015, 0.231, 0.044]   <- spread out

dk = 256
  raw scores          : [24.66, -4.69, -34.86, 9.09, -17.35]
  softmax(raw)        : [1.0,   0.0,   0.0,    0.0,  0.0  ]   <- fully one-hot
  softmax(raw / √dk)  : [0.612, 0.098, 0.015, 0.231, 0.044]   <- same as dk=16

A one-hot softmax means each token copies exactly one other token and ignores the rest.
```

Why large scores hurt: gradients vanish
```text
Gradient of p0 (first softmax output) with respect to the 5 scores:

  normal scores     : [0.2375, -0.0598, -0.0091, -0.1415, -0.0271]
  scores × 16       : [0.0,    -0.0,    -0.0,    -0.0,    -0.0   ]

When softmax is saturated, nudging a score barely changes the output,
so the gradient is ~0 and W_q / W_k stop learning.
```

Solution
```text
Divide by √dk. That undoes the √dk growth of the std, so scores have std ≈ 1
whatever dk is (see the last column of the first table).

In attention_layer.py:
    scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)

Why √dk and not dk?  Dividing by dk would make the std 1/√dk: it shrinks as dk grows,
so softmax would become almost uniform (every token gets the same weight).
√dk is the one value that keeps std = 1.
```

Hand calc (dk = 2)
```text
q_token0 = [1, 2]
k_token0 = [1, 0]   k_token1 = [0, 3]

raw(0->0) = 1·1 + 2·0 = 1        raw(0->1) = 1·0 + 2·3 = 6
/ √2 ≈ 1.414                     ->  0.71      and      4.24
```

Try it yourself
```text
1. Predict: in attention_layer.py with num_head=4, what is dk, and what is √dk?
   Then print self.dk to check.
2. Remove the "/ math.sqrt(self.dk)" for a moment, run attention_layer.py, and print
   scores.std(). Put it back and print it again. Which one is closer to 1?
   (The weights start random, so the numbers won't match the tables exactly.)
3. Apply torch.softmax(scores, dim=-1) with and without the scaling and print
   row 0. Which one looks more "one-hot"?
4. Use the dk = 2 hand calc: compute softmax([1, 6]) and softmax([0.71, 4.24]) on paper.
   How much weight does token 0 put on token 1 in each case?
5. Think: rows 3-11 of scores come from [PAD] tokens. Does scaling fix the PAD problem?
   (Hint: no. That is what the mask is for, and it's the next step.)
```
