"""
A tiny word-level tokenizer for the intent-classification dataset.

Pipeline:  raw text -> split_words() -> token strings -> ids (+ attention mask)

Every encoded sentence looks like:
    [CLS] w1 w2 ... wn [PAD] [PAD] ...
      - [CLS] is always first; its final hidden state is used for classification.
      - [PAD] fills the sequence up to a fixed length so sentences can be batched.
      - [UNK] replaces any word that was not seen when the vocabulary was built.
"""
import re

import torch

from data_sets import texts

# Special tokens. They are placed at the start of the vocabulary, so their ids
# are 0, 1, 2 in the order written here.
vocabulary_defined_tokens = {
    "CLS": "[CLS]",
    "PAD": "[PAD]",
    "UNK": "[UNK]",
}


def split_words(text):
    """
    Lowercase the text and split it into tokens.

    The regex keeps two kinds of tokens:
      [a-z']+       -> words, apostrophes kept inside them ("what's")
      [^\\sa-z']    -> any single non-space, non-letter char (",", "?", "!", digits)
    Whitespace is dropped.

    >>> split_words("Hello, how are you?")
    ['hello', ',', 'how', 'are', 'you', '?']
    """
    return re.findall(r"[a-z']+|[^\sa-z']", text.lower())


class Tokenizer:

    def __init__(self, sentences):
        """
        Build the vocabulary from a corpus.

        :param sentences: list of strings used to build the vocabulary
        """
        self.corpus = sentences

        # Every unique token in the corpus, sorted so ids are deterministic
        # across runs (a plain set has no stable order).
        vocabulary = sorted({w for s in sentences for w in split_words(s)})

        # itos = "index to string": position in the list is the token id.
        # stoi = "string to index": the reverse lookup used by encode().
        self.itos = list(vocabulary_defined_tokens.values()) + vocabulary

        self.stoi = {w: i for i, w in enumerate(self.itos)}

        # Cache the special-token ids so we don't look them up every call.
        self.cls_id = self.stoi[vocabulary_defined_tokens["CLS"]]
        self.pad_id = self.stoi[vocabulary_defined_tokens["PAD"]]
        self.unk_id = self.stoi[vocabulary_defined_tokens["UNK"]]

    @property
    def vocab_size(self):
        """Number of distinct ids; this is what nn.Embedding needs."""
        return len(self.itos)

    @property
    def max_length(self):
        """
        Length (in tokens, including [CLS]) of the longest sentence in the corpus.

        Note: this counts tokens, not characters. len(s) on the raw string
        would count characters and give a much larger, mostly-padding length.
        """
        return max(len(split_words(s)) + 1 for s in self.corpus)

    def encode(self, text, max_length):
        """
        Convert one sentence into fixed-length token ids and an attention mask.

        :param text: raw input string
        :param max_length: final length of both returned lists
                           (longer sentences are truncated, shorter ones padded)
        :return: (ids, mask)
                 ids  - list[int], token ids starting with [CLS]
                 mask - list[int], 1 for real tokens, 0 for [PAD]
        """
        # [CLS] first, then each word's id; unseen words fall back to [UNK].
        ids = [self.cls_id] + [self.stoi.get(w, self.unk_id) for w in split_words(text)]

        # Truncate if the sentence is too long (keeps [CLS] since it's at index 0).
        ids = ids[:max_length]

        # Mask marks which positions are real tokens, so attention can ignore padding.
        mask = [1] * len(ids)

        # Pad both lists up to max_length.
        padding = max_length - len(ids)
        ids += [self.pad_id] * padding
        mask += [0] * padding
        return ids, mask

    def encode_batch(self, texts, max_length):
        """
        Encode several sentences into tensors the model can consume.

        :param texts: list of raw input strings
        :param max_length: length every sentence is padded/truncated to
        :return: (ids, mask)
                 ids  - LongTensor (B, T), token ids
                 mask - LongTensor (B, T), 1 for real tokens, 0 for [PAD]
                        attention uses this to ignore padded keys
        """
        pairs = [self.encode(t, max_length) for t in texts]
        ids = torch.tensor([p[0] for p in pairs])
        mask = torch.tensor([p[1] for p in pairs])
        return ids, mask

    def decode(self, ids):
        """Map ids back to token strings (handy for debugging)."""
        return [self.itos[i] for i in ids]


if __name__ == '__main__':
    tknzr = Tokenizer(texts)
    MAX_LENGTH = 12  # fixed for the whole tutorial; tknzr.max_length is the corpus minimum
    print("vocab size:", tknzr.vocab_size)
    print("longest sentence (tokens incl. [CLS]):", tknzr.max_length)
    for e in texts[:3]:
        ids, mask = tknzr.encode(e, MAX_LENGTH)
        print(f"\n{e!r}")
        print("  tokens:", tknzr.decode(ids))
        print("  ids   :", ids)
        print("  mask  :", mask)
