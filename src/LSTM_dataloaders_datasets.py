from itertools import groupby
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset



class LSTM_ZTF_dataset(Dataset):
    def __init__(self, parquet_path: str, feature_cols: list, label_col: str = "label", max_len: int = 20, min_len: int = 3):
        df = pd.read_parquet(parquet_path)
        self.feature_cols = feature_cols
        df['target'] = (df[label_col] == 'Kilonovae').astype(int)
        grouped = df.groupby('group')

        self.sequences = [] 
        self.labels = []

        # 1. Find the true max length of Kilonovae
        kn_lengths = [len(g) for _, g in grouped if g['target'].iloc[0] == 1]
        if len(kn_lengths) > 0:
            dynamic_max = int(np.percentile(kn_lengths, 95))
            self.effective_max_len = min(dynamic_max, max_len)
        else:
            self.effective_max_len = max_len

        # 2. Process sequences
        for group_id, group_df in grouped:
            features = group_df[self.feature_cols].values.astype(np.float32)
            target = group_df['target'].iloc[0]

            if np.all(features[:, 0] == 0) and target == 1:
                continue

            if len(features) < min_len:
                continue

            # TRUNCATE LONG SEQUENCES TO MATCH KN LENGTHS
            if len(features) > self.effective_max_len:
                start_idx = np.random.randint(0, len(features) - self.effective_max_len)
                features = features[start_idx : start_idx + self.effective_max_len]

            # SAFE NORMALIZATION
            features[:, 2] = features[:, 2] - features[0, 2] # Relative time
            max_flux = np.max(np.abs(features[:, 0])) + 1e-6
            features[:, 0] = features[:, 0] / max_flux       # Scaled flux
            features[:, 1] = features[:, 1] / max_flux       # Scaled error

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)







        