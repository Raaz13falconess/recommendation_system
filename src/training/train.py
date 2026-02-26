import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.matrix_factorization import MatrixFactorization


class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for users, items, ratings in tqdm(dataloader):
        users = users.to(device)
        items = items.to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad()
        predictions = model(users, items)

        loss = criterion(predictions, ratings)
        loss.backward()
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
            loss = criterion(predictions, ratings)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")

    n_users = train_df["user_idx"].nunique()
    n_items = train_df["movie_idx"].nunique()

    train_dataset = RatingsDataset(train_df)
    val_dataset = RatingsDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)

    model = MatrixFactorization(n_users, n_items, embedding_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    epochs = 5

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train MSE: {train_loss:.4f}")
        print(f"Val MSE: {val_loss:.4f}")
        print("-" * 30)


if __name__ == "__main__":
    main()