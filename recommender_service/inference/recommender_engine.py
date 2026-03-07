import torch


class RecommenderEngine:

    def __init__(self, model, device):

        self.device = device
        self.model = model

        self.user_embeddings = model.user_embedding.weight.data
        self.item_embeddings = model.item_embedding.weight.data

        self.user_bias = model.user_bias.weight.data.squeeze()
        self.item_bias = model.item_bias.weight.data.squeeze()

        self.global_bias = model.global_bias.data

    def recommend(self, user_id, k=10, seen_items=None):

        user_vec = self.user_embeddings[user_id]

        # Dot product scores
        scores = torch.matmul(self.item_embeddings, user_vec)

        # Add bias terms
        scores = scores + self.user_bias[user_id]
        scores = scores + self.item_bias
        scores = scores + self.global_bias

        if seen_items is not None:
            scores[seen_items] = -1e9

        top_k = torch.topk(scores, k=k).indices.cpu().numpy()
        
        return top_k