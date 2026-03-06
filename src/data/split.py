import pandas as pd


def time_based_split(df, train_ratio=0.8, val_ratio=0.1):

    train_list = []
    val_list = []
    test_list = []

    for user_id, user_df in df.groupby("user_idx"):

        user_df = user_df.sort_values("timestamp")

        n = len(user_df)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_list.append(user_df.iloc[:train_end])
        val_list.append(user_df.iloc[train_end:val_end])
        test_list.append(user_df.iloc[val_end:])

    train = pd.concat(train_list)
    val = pd.concat(val_list)
    test = pd.concat(test_list)

    return train, val, test