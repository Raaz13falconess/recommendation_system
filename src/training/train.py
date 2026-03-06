import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

from src.models.matrix_factorization import MatrixFactorization


class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_idx, (users, items, ratings) in enumerate(dataloader):

        users = users.to(device)
        items = items.to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad()

        predictions = model(users, items)

        # DEBUG: check predictions
        if torch.isnan(predictions).any():
            print("NaN detected in predictions!")
            print("Users min/max:", users.min(), users.max())
            print("Items min/max:", items.min(), items.max())
            return float("nan")

        predictions = model(users, items)
        loss = criterion(predictions, ratings)

        if torch.isnan(loss):
            print("NaN detected in loss!")
            print("Predictions min/max:", predictions.min(), predictions.max())
            print("Ratings min/max:", ratings.min(), ratings.max())
            return float("nan")

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for users, items, ratings in dataloader:
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            predictions = model(users, items)
            predictions = torch.clamp(predictions, 0, 5)

            loss = criterion(predictions, ratings)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading parquet files...")
    train_df = pd.read_parquet("Data/processed/train.parquet")
    val_df = pd.read_parquet("Data/processed/val.parquet")

    # Debug data integrity
    print("Any NaN in train ratings?", train_df["rating"].isna().any())
    print("Train rating min/max:", train_df["rating"].min(), train_df["rating"].max())

    # IMPORTANT: compute embedding size using BOTH sets
    all_users = pd.concat([train_df["user_idx"], val_df["user_idx"]])
    all_items = pd.concat([train_df["movie_idx"], val_df["movie_idx"]])

    n_users = all_users.max() + 1
    n_items = all_items.max() + 1

    print("Embedding sizes:")
    print("Users:", n_users)
    print("Items:", n_items)

    train_dataset = RatingsDataset(train_df)
    val_dataset = RatingsDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)

    model = MatrixFactorization(n_users, n_items, embedding_dim=100).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    epochs = 20

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train RMSE: {math.sqrt(train_loss):.4f}")
        print(f"Val RMSE: {math.sqrt(val_loss):.4f}")
        print("-" * 40)


if __name__ == "__main__":
    main()