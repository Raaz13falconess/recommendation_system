import pandas as pd
from pathlib import Path


def load_ratings(data_path):
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    print(f"Loading file: {data_path}")

    # Detect file type automatically
    if str(path).endswith(".dat"):
        df = pd.read_csv(
            data_path,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )
    else:
        df = pd.read_csv(path)

        # Normalize column names if needed
        df.columns = ["user_id", "movie_id", "rating", "timestamp"]

    print("Raw data sample:")
    print(df.head())

    print("Any NaN after loading?")
    print(df.isna().sum())

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    return df