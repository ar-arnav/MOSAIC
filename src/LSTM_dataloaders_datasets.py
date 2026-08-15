from typing import Sequence
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class LSTM_ZTF_dataset(Dataset):
    def __init__(self, parquet_path: str, label_col: str = "label", max_len: int = 20, min_len: int = 3):
        df = pd.read_parquet(parquet_path)
        df['target'] = (df[label_col] == 'Kilonovae').astype(int)
        grouped = df.groupby('group')

        self.sequences = [] 
        self.labels = []

        # 1. Find the true max length of Kilonovae to prevent length leakage
        kn_lengths = [len(g) for _, g in grouped if g['target'].iloc[0] == 1]
        if len(kn_lengths) > 0:
            dynamic_max = int(np.percentile(kn_lengths, 95))
            self.effective_max_len = min(dynamic_max, max_len)
        else:
            self.effective_max_len = max_len

        # 2. Process sequences
        for group_id, group_df in grouped:
            # Grabbing the raw ZTF columns
            mag = group_df['magpsf'].values.astype(np.float32)
            mag_err = group_df['sigmapsf'].values.astype(np.float32)
            time = group_df['time_day'].values.astype(np.float32)
            fid = group_df['fid'].values.astype(np.float32)
            target = group_df['target'].iloc[0]

            if len(mag) < min_len:
                continue

            # TRUNCATE LONG SEQUENCES TO MATCH KN LENGTHS
            if len(mag) > self.effective_max_len:
                start_idx = np.random.randint(0, len(mag) - self.effective_max_len)
                mag = mag[start_idx : start_idx + self.effective_max_len]
                mag_err = mag_err[start_idx : start_idx + self.effective_max_len]
                time = time[start_idx : start_idx + self.effective_max_len]
                fid = fid[start_idx : start_idx + self.effective_max_len]

            # --- REAL ZTF PHYSICS PREPROCESSING ---
            
            # 1. DELTA_MAG
            delta_mag = (mag - mag[0]) / 5.0  

            # 2. DELTA_T
            dt = np.diff(time, prepend=time[0]) / 30.0

            # 3. REL_ERR
            max_err = np.max(mag_err) + 1e-6
            rel_err = mag_err / max_err

            # 4. FILTER_ID
            fid_scaled = fid / 2.0

            # Stack into final shape (seq_len, 4)
            features = np.stack([delta_mag, dt, rel_err, fid_scaled], axis=1)

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


train_dataset = LSTM_ZTF_dataset(parquet_path='data/processed/train.parquet')
val_dataset = LSTM_ZTF_dataset(parquet_path='data/processed/val.parquet')
test_dataset = LSTM_ZTF_dataset(parquet_path='data/processed/test.parquet')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)