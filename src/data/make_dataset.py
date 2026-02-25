import argparse
from pathlib import Path

from load import load_ratings
from preprocess import encode_ids
from split import time_based_split

def main(input_path: str, output_dir: str):
    
    print("Loading data...")
    df = load_ratings(input_path)

    print("Encoding IDs....")
    df, user_map, movie_map = encode_ids(df)

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
    main("Data/raw/ml-1m/ratings.dat", "Data/processed")