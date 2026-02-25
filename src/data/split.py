import pandas as pd

def time_based_split(df: pd.DataFrame, train_size=0.8, val_size=0.1):
    df = df.sort_values("timestamp")

    n_total = len(df)
    n_train = int(n_total* train_size)
    n_val = int(n_total* val_size)

    train = df.iloc[:n_train]
    val = df.iloc[n_train: n_train + n_val]
    test = df.iloc[n_train + n_val:]

    return train, val, test

