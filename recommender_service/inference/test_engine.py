from recommender_engine import RecommenderEngine

engine = RecommenderEngine(
    "embeddings/user_embeddings.npy",
    "embeddings/item_embeddings.npy"
)

print(engine.recommend(user_id=42))