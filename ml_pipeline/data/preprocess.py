import pandas as pd 

def encode_ids(df: pd.DataFrame):
    user_mapping = {id_: idx for idx, id_ in enumerate(df["user_id"].unique())}
    movie_mapping = {id_: idx for idx, id_ in enumerate(df["movie_id"].unique())}

    df["user_idx"] = df["user_id"].map(user_mapping)
    df["movie_idx"] = df["movie_id"].map(movie_mapping)

    return df, user_mapping, movie_mapping
