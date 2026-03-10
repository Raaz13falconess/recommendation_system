import argparse
from pathlib import Path

from ml_pipeline.data.load import load_ratings
from ml_pipeline.data.preprocess import encode_ids
from ml_pipeline.data.split import time_based_split

import json

BASE_DIR = Path(__file__).resolve().parents[2]

def main(input_path: str, output_dir: str):
    
    print("Loading data...")
    df = load_ratings(input_path)

    print("Encoding IDs....")
    df, user_map, movie_map = encode_ids(df)

    movie_index_mapping = (
        df[["movie_idx", "movie_id"]]
        .drop_duplicates()
        .set_index("movie_idx")["movie_id"]
        .to_dict()
    )

    Path("embeddings").mkdir(exist_ok=True)

    with open("embeddings/movie_index_map.json", "w") as f:
        json.dump(movie_index_mapping, f)

    print("Splitting dataset....")
    train, val, test = time_based_split(df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Saving parquet files...")
    train.to_parquet(output_path / "train.parquet")
    val.to_parquet(output_path/ "val.parquet")
    test.to_parquet(output_path / "test.parquet")

    print("Done.....")


if __name__ == "__main__":
    input_path = Path(BASE_DIR / "Data/raw/ml-1m/ratings.dat")
    output_path = Path(BASE_DIR / "Data/processed")
    main(input_path, output_path)