from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from pathlib import Path

from recommender_service.inference.recommender_engine import RecommenderEngine
from recommender_service.inference.movie_metadata import MovieMetadata

BASE_DIR = Path(__file__).resolve().parents[2]

app = FastAPI(title="Movie Recommendation API")
metadata = MovieMetadata(BASE_DIR / "Data/raw/ml-1m/movies.dat")


# Load recommender engine once at startup
engine = RecommenderEngine(
    BASE_DIR / "embeddings/user_embeddings.npy",
    BASE_DIR / "embeddings/item_embeddings.npy",
    
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