# Tiny Transformer: Text → Emotion Emoji

Goal: **learn the transformer architecture** by building a tiny one from scratch in PyTorch.
The task is small on purpose: read a sentence and predict one emoji.

```
"i just got the job!!"   ──►  Tiny Transformer  ──►  😊
"my dog passed away"     ──►  Tiny Transformer  ──►  😢
```

## The big picture (what we will build)

```
 text
  │
  ▼
[1] Tokenizer ............ "i love it" → [2, 14, 7, 31, 0, 0]   (numbers)
  │
  ▼
[2] Token Embedding ...... each id → a vector of size d_model
  + Positional Encoding .. add "where am I in the sentence" info
  │
  ▼
[3] Self-Attention (Q,K,V) ... every word looks at every other word
[4] Multi-Head Attention ..... do [3] several times in parallel
[5] Feed-Forward + Residual + LayerNorm  = one Encoder Block
  │   (stack N blocks)
  ▼
[6] Classifier head ...... take the [CLS] vector → 6 scores → softmax
  │
  ▼
[7] Training loop ........ loss, backprop, optimizer
[8] Inference + visualising attention weights
```

We build an **encoder-only** transformer (like BERT). This is the simplest
transformer that still has *all* the core ideas: Q/K/V, self-attention,
multi-head attention, positional encoding, residuals, LayerNorm.
(A decoder adds masked attention + cross-attention; we can do that as a bonus step.)

## Steps

| Step | Code file             | Doc                                  | Status |
|------|-----------------------|--------------------------------------|--------|
| 1    | `data.py`, `tokenizer.py` | `01_data_and_tokenization.md`    | ✅     |
| 2    | `embeddings.py`       | `02_embeddings_and_positions.md`     | ✅     |
| 3    | `attention.py`        | `03_self_attention_qkv.md`           | ⏳     |
| 4    | `attention.py`        | `04_multi_head_attention.md`         | ⏳     |
| 5    | `encoder.py`          | `05_encoder_block.md`                | ⏳     |
| 6    | `model.py`            | `06_full_model.md`                   | ⏳     |
| 7    | `train.py`            | `07_training.md`                     | ⏳     |
| 8    | `predict.py`          | `08_inference_and_attention_viz.md`  | ⏳     |

Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple`
Run any step with `.venv/bin/python <file>.py`.

Say **"next"** to continue to the next step.
