"""
token : [cls] [i] [love] [my] [family] [pad] [pad] [pad]
ids : 0 23 34 45 3 1 1 1

"""
import torch
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=64, # Vocab_size
    embedding_dim=16,
    padding_idx=1
)

ids = torch.tensor([[0,23,34,45,3,1,1,1]])
embedded = embedding(ids)
print(embedded)
# print("IDs SHAPE:", ids.shape)
# print("Embedding: SHAPE", embedded.shape)

"""
IDs SHAPE: torch.Size([1, 8])
Embedding: SHAPE torch.Size([1, 8, 16])
"""