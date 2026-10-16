"""
Step 1b: A tiny word-level tokenizer.

Neural networks only understand numbers, so we turn text into a list of
integer ids. Real models (GPT, BERT) use sub-word tokenizers (BPE/WordPiece),
but a simple word tokenizer shows the same idea with less noise.
"""

import re

# Special tokens. Their ids are fixed so the rest of the code can rely on them.
PAD = "[PAD]"  # id 0: filler so every sentence in a batch has the same length
UNK = "[UNK]"  # id 1: any word we never saw during training
CLS = "[CLS]"  # id 2: a "summary" slot at the start; the classifier reads it
SPECIAL_TOKENS = [PAD, UNK, CLS]


def split_words(text):
    """'Wow, I did it!' -> ['wow', ',', 'i', 'did', 'it', '!']"""
    return re.findall(r"[a-z']+|[^\sa-z']", text.lower())


class Tokenizer:
    def __init__(self, sentences):
        # Build the vocabulary: every unique word gets an id, after the specials.
        words = sorted({w for s in sentences for w in split_words(s)})
        self.itos = SPECIAL_TOKENS + words                 # id -> string
        self.stoi = {w: i for i, w in enumerate(self.itos)}  # string -> id

        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.cls_id = self.stoi[CLS]

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text, max_len):
        """
        Text -> (ids, mask), both of length max_len.

        ids  : [CLS] w1 w2 ... wn [PAD] [PAD] ...
        mask : 1 for real tokens, 0 for padding (attention will ignore 0s later)
        """
        ids = [self.cls_id] + [self.stoi.get(w, self.unk_id) for w in split_words(text)]
        ids = ids[:max_len]  # truncate if too long

        mask = [1] * len(ids)
        padding = max_len - len(ids)
        ids += [self.pad_id] * padding
        mask += [0] * padding
        return ids, mask

    def decode(self, ids):
        return [self.itos[i] for i in ids]


if __name__ == "__main__":
    from data import get_dataset

    data = get_dataset()
    tok = Tokenizer([s for s, _ in data])
    print("vocab size:", tok.vocab_size)
    print("first 10 vocab entries:", tok.itos[:10])

    MAX_LEN = 12
    for text in ["i love my family", "wow, my zebra did NOT expect that!"]:
        ids, mask = tok.encode(text, MAX_LEN)
        print()
        print("text   :", text)
        print("tokens :", tok.decode(ids))
        print("ids    :", ids)
        print("mask   :", mask)
