import numpy as np


class RecommenderEngine:

    def __init__(self, user_embedding_path, item_embedding_path):

        self.user_embeddings = np.load(user_embedding_path)
        self.item_embeddings = np.load(item_embedding_path)

    def recommend(self, user_id, k=10, seen_items=None):

        user_vector = self.user_embeddings[user_id]

        scores = self.item_embeddings @ user_vector

        if seen_items is not None:
            scores[seen_items] = -1e9

        top_k = np.argsort(scores)[-k:][::-1]

        return top_k