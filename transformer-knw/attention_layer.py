from torch import nn


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
        dk = d_model/num_head if num_head else d_model
        self.W_q = nn.Linear(d_model,dk,bias=False)
        self.W_k = nn.Linear(d_model,dk,bias=False)
        self.W_v = nn.Linear(d_model,dk,bias=False)

    def forward(self,X):
        q,k,v = self.W_q(X), self.W_k(X), self.W_v(X)
        print("🦾 | Attention layer shapes")
        print("")


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