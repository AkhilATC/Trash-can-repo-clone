# Step 1: Data and Tokenization

Files: `data.py`, `tokenizer.py`
Run: `python3 data.py` and `python3 tokenizer.py`

## 1. The task

**Text classification**: one sentence in, one of 6 emotion classes out.

| index | emotion  | emoji |
|-------|----------|-------|
| 0     | joy      | 😊    |
| 1     | sadness  | 😢    |
| 2     | anger    | 😠    |
| 3     | fear     | 😨    |
| 4     | love     | 😍    |
| 5     | surprise | 😲    |

The model never sees an emoji. It predicts a **number (0–5)**, and we look
up the emoji in that table. Our dataset is 72 hand-written sentences
(12 per class), which is tiny but enough to learn how the architecture works.

## 2. Why tokenize?

A neural network does math on numbers, not text. So we need:

```
"i love my family"  →  ["i", "love", "my", "family"]  →  [81, 99, 110, 48]
        text                     tokens                     token ids
```

- **Token**: one piece of text. Ours are whole words. GPT and BERT use
  *sub-words* (`"playing"` → `"play" + "ing"`) so they never hit an unknown word.
- **Vocabulary (vocab)**: the list of every token the model knows. Each
  token's position in the list is its **id**. Our vocab has **193** entries.

## 3. Special tokens

| token   | id | why it exists |
|---------|----|---------------|
| `[PAD]` | 0  | Batches need equal-length sequences, so short sentences are filled with PAD. |
| `[UNK]` | 1  | Any word not in the vocab (e.g. `zebra`) becomes UNK. |
| `[CLS]` | 2  | Put at position 0 of every sentence. After the transformer, this slot's vector is used as **the summary of the whole sentence** and goes to the classifier. (The same idea as BERT.) |

Why use `[CLS]` and not just "the last word"? With self-attention (step 3),
**every token can look at every other token**. So the `[CLS]` vector can
collect information from the whole sentence. It has no meaning of its own,
which makes it a clean "summary" slot.

## 4. Padding and the attention mask

With `max_len = 12`:

```
tokens : [CLS]  i   love  my  family [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
ids    :   2   81   99   110   48     0     0     0     0     0     0     0
mask   :   1    1    1    1     1     0     0     0     0     0     0     0
```

The **mask** marks which positions are real (1) and which are padding (0).

⚠️ Remember this mask. In step 3 it is used to stop words from
attending to `[PAD]`. Without it, the model would mix "nothing" into its
understanding of the sentence.

## 5. Shapes (get used to these)

For a batch of `B` sentences, each padded to length `T`:

```
ids  : (B, T)   integers
mask : (B, T)   0/1
```

`T` is often called **sequence length**. Every later step adds a dimension:
the embedding turns `(B, T)` into `(B, T, d_model)`.

## Key takeaways

- Text → tokens → ids. The model only sees ids.
- Vocab = token ↔ id lookup table.
- `[PAD]` makes lengths equal. The **mask** tells attention to ignore it.
- `[CLS]` is a summary slot. Its final vector is what we classify.
- Unknown words → `[UNK]`. Sub-word tokenizers avoid most of these.

## Try it yourself

1. Encode a sentence longer than 12 words. What happens to the end?
2. Add a 7th emotion (e.g. 😴 tired) to `data.py` with a few sentences.
   How does the vocab size change?

**Next → Step 2: Embeddings and positional encoding** (ids → vectors, plus word order).
