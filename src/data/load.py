import pandas as pd
from pathlib import Path

def load_ratings(data_path: str) -> pd.DataFrame:
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"{data_path} not found")
    
    df = pd.read_csv(
        path, 
        sep=":",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    return df