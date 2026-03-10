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
    
    def similar_movies(self, movie_id, k=10):

        # Convert MovieLens ID → embedding index
        movie_idx = None
        for idx, mid in self.movie_map.items():
            if mid == movie_id:
                movie_idx = int(idx)
                break

        if movie_idx is None:
            raise ValueError("Movie ID not found")

        target_vector = self.item_embeddings[movie_idx]

        # Normalize embeddings
        item_norms = np.linalg.norm(self.item_embeddings, axis=1)
        target_norm = np.linalg.norm(target_vector)

        similarities = self.item_embeddings @ target_vector / (item_norms * target_norm + 1e-8)

        # Exclude the movie itself
        similarities[movie_idx] = -1

        top_k_idx = np.argsort(similarities)[-k:][::-1]

        movie_ids = [self.movie_map[str(i)] for i in top_k_idx]

        return movie_ids