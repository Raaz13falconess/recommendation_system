import json
import numpy as np


class RecommenderEngine:

    def __init__(self, user_embedding_path, item_embedding_path, movie_map_path):

        self.user_embeddings = np.load(user_embedding_path)
        self.item_embeddings = np.load(item_embedding_path)

        with open(movie_map_path) as f:
            self.movie_map = json.load(f)

    def recommend(self, user_id, k=10, seen_items=None):

        user_vector = self.user_embeddings[user_id]

        scores = self.item_embeddings @ user_vector

        if seen_items is not None:
            scores[seen_items] = -1e9

        top_k_idx = np.argsort(scores)[-k:][::-1]

        movie_ids = [self.movie_map[str(i)] for i in top_k_idx]

        return movie_ids