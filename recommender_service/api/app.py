from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from recommender_service.inference.recommender_engine import RecommenderEngine
from recommender_service.inference.movie_metata import MovieMetadata


app = FastAPI(title="Movie Recommendation API")
metadata = MovieMetadata("../../Data/raw/ml-1m/movies.dat")


# Load recommender engine once at startup
engine = RecommenderEngine(
    "embeddings/user_embeddings.npy",
    "embeddings/item_embeddings.npy"
)


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[int]


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):

    movie_ids = engine.recommend(user_id, k)

    results = [
        {
            "movie_id": mid,
            "title": metadata.get_title(mid)
        }
        for mid in movie_ids
    ]

    return {
        "user_id": user_id,
        "recommendations": results
    }