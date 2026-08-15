from itertools import groupby
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset



class LSTM_ZTF_dataset(Dataset):

    def __init__(self, parquet_path: str, feature_cols: list, label_col: str = "label"):
        df = pd.read_parquet(parquet_path)
        self.feature_cols = feature_cols

        df['target'] = (df[label_col] == 'Kilonovae').astype(int)

        grouped = df.groupby('group')

        self.sequences = [] 
        self.labels = []

        for group_id, group_df in grouped:
            features = ((group_df[self.feature_cols]).values).astype(np.float32)
            target = group_df['target'].iloc[0]

            if np.all(features[:, 0] == 0) and target == 1:
                continue

            if len(features) > 150:
                features = features[-150:]  # Cap background length

            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-6
            features = (features - mean) / std  # Normalize features
            # -----------------------

            features = torch.tensor(features, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.long)

            self.sequences.append(features)
            self.labels.append(target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        sequence = self.sequences[idx]
        label = self.labels[idx]

        return sequence, label

def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int64)

    padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    stacked_labels = torch.stack(labels)
    return padded_sequence, stacked_labels, lengths

feature_cols = ['flux', 'flux_err', 'time_day', 'fid']

train_dataset = LSTM_ZTF_dataset(parquet_path='data/processed/train.parquet', feature_cols=feature_cols)
val_dataset = LSTM_ZTF_dataset(parquet_path='data/processed/val.parquet', feature_cols=feature_cols)
test_dataset = LSTM_ZTF_dataset(parquet_path='data/processed/test.parquet', feature_cols=feature_cols)

train_loader = DataLoader(
    train_dataset,
    batch_size=32, 
    shuffle=True, 
    collate_fn=collate_fn
    )

val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, 
    collate_fn=collate_fn
    )

test_loader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False, 
    collate_fn=collate_fn
    )







        