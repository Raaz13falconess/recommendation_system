import numpy as np
import json
import faiss


class RecommenderEngine:

    def __init__(self, user_embedding_path, item_embedding_path, movie_map_path):

        self.user_embeddings = np.load(user_embedding_path)
        self.item_embeddings = np.load(item_embedding_path).astype("float32")

        with open(movie_map_path) as f:
            self.movie_map = json.load(f)

        # Build FAISS index
        dim = self.item_embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.item_embeddings)

        self.index.add(self.item_embeddings)

    def recommend(self, user_id, k=10):

        user_vector = self.user_embeddings[user_id].astype("float32").reshape(1, -1)

        faiss.normalize_L2(user_vector)

        scores, indices = self.index.search(user_vector, k)

        movie_ids = [self.movie_map[str(i)] for i in indices[0]]

        return movie_ids

    def similar_movies(self, movie_id, k=10):

        movie_idx = None
        for idx, mid in self.movie_map.items():
            if mid == movie_id:
                movie_idx = int(idx)
                break

        if movie_idx is None:
            raise ValueError("Movie ID not found")

        movie_vector = self.item_embeddings[movie_idx].reshape(1, -1)

        scores, indices = self.index.search(movie_vector, k + 1)

        indices = indices[0][1:]

        movie_ids = [self.movie_map[str(i)] for i in indices]

        return movie_ids