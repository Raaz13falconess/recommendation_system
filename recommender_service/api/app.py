from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from recommender_service.inference.recommender_engine import RecommenderEngine


app = FastAPI(title="Movie Recommendation API")


# Load recommender engine once at startup
engine = RecommenderEngine(
    "embeddings/user_embeddings.npy",
    "embeddings/item_embeddings.npy"
)


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[int]


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, k: int = 10):

    recommendations = engine.recommend(user_id=user_id, k=k)

    return {
        "user_id": user_id,
        "recommendations": recommendations.tolist()
    }