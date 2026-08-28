import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def read_merge(csv_path, parquet_path):
    rtf_output = pd.read_csv(csv_path)
    gw_map = pd.read_parquet(parquet_path)

    df = pd.merge(rtf_output, gw_map, on='object_id', how='inner')
    df = df.dropna()

    X = df[['rtf_score', 'gw_cred_level', 'gw_dp_dv']].values
    y = df['label'].values

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class GWDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(train_csv, train_parquet, val_csv, val_parquet, 
                    test_csv=None, test_parquet=None, batch_size=64):
    X_train, y_train = read_merge(train_csv, train_parquet)
    X_val, y_val = read_merge(val_csv, val_parquet)

    train_dataset = GWDataset(X_train, y_train)
    val_dataset = GWDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*4, shuffle=False)

    loaders = {'train': train_loader, 'val': val_loader}

    if test_csv and test_parquet:
        X_test, y_test = read_merge(test_csv, test_parquet)
        test_dataset = GWDataset(X_test, y_test)
        loaders['test'] = DataLoader(test_dataset, batch_size=batch_size*4, shuffle=False)

    return loaders