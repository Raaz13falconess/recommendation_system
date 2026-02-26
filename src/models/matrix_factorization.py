import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        self.global_bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx, item_idx):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        dot_product = (user_emb * item_emb).sum(dim=1)

        user_b = self.user_bias(user_idx).squeeze()
        item_b = self.item_bias(item_idx).squeeze()

        return dot_product + user_b + item_b + self.global_bias