import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score

from LSTM_ZTF import LSTM_ZTF
from LSTM_dataloaders_datasets import train_loader, val_loader, test_loader

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
      super().__init__()
      self.gamma = gamma
      self.alpha = alpha
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):

      if targets.dim() == 1:
        targets = targets.unsqueeze(1)
      targets = targets.float()

      BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

      pt = torch.exp(-BCE_loss)

      alpha_t = targets * self.alpha + (1-targets) * (1-self.alpha)

      focal_loss = alpha_t * ((1-pt)**self.gamma) * BCE_loss

      return focal_loss.mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM_ZTF(
    input_size = 4,
    hidden_size = 64,
    num_layers = 2,
    output_size = 1,
    dropout = 0.2,
).to(device)

criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',       # Monitors validation loss drops
    factor=0.5,       # Halves LR when loss plateaus
    patience=3        # Waits 3 epochs of no improvement before dropping LR
)



def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels, lengths in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device).float()
        lengths = lengths.to("cpu")

        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        optimizer.zero_grad()

        logits = model(sequences, lengths)
        
        # FIX: Ensure logits is 2D to match labels (32, 1)
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)

        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
  
    return epoch_loss, epoch_acc


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_probas = []

    for sequences, labels, lengths in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device).float()
        lengths = lengths.to("cpu")

        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        logits = model(sequences, lengths)
        
        # FIX: Ensure logits is 2D to match labels (32, 1)
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)

        loss = criterion(logits, labels)

        total_loss += loss.item() * sequences.size(0)

        probas = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        targets = labels.squeeze(-1).cpu().numpy()
        all_probas.extend(probas)
        all_targets.extend(targets)

        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    val_loss = total_loss / total
    val_acc = correct / total

    precision, recall, _ = precision_recall_curve(all_targets, all_probas)
    val_pr_auc = auc(recall, precision)
    
    binary_preds = (np.array(all_probas) >= 0.5).astype(int)
    val_f1 = f1_score(all_targets, binary_preds, zero_division=0)
  
    return val_loss, val_acc, val_pr_auc, val_f1


if __name__ == "__main__":
    MAX_EPOCHS = 20
    PATIENCE = 5
    patience_counter = 0
    best_val_loss = float("inf")

    print("\nStarting Training Execution...")
    print("=" * 75)
    print(f"{'Epoch':^7} | {'Tr Loss':^10} | {'Tr Acc':^9} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val PR-AUC':^10} | {'Val F1':^8}")
    print("=" * 75)

    for epoch in range(1, MAX_EPOCHS + 1):
        # 1. Train 1 Epoch
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 2. Evaluate Validation Set
        val_loss, val_acc, val_pr_auc, val_f1 = eval_epoch(model, val_loader, criterion, device)

        # 3. Learning Rate Scheduler Step
        scheduler.step(val_loss)

        # 4. Print Status Row
        print(f"{epoch:^7d} | {tr_loss:^10.5f} | {tr_acc:^9.4f} | {val_loss:^10.5f} | {val_acc:^9.4f} | {val_pr_auc:^10.4f} | {val_f1:^8.4f}")

        # 5. Checkpoint & Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_kilonova_lstm.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("-" * 75)
                print(f"⏹️ Early stopping triggered! Validation loss did not improve for {PATIENCE} epochs.")
                break

    # =====================================================================
    # 5. UNSEEN TEST EVALUATION (FINAL STEP)
    # =====================================================================
    print("=" * 75)
    print("📦 Loading best saved model checkpoint for final Test set evaluation...")
    model.load_state_dict(torch.load("best_kilonova_lstm.pt"))

    test_loss, test_acc, test_pr_auc, test_f1 = eval_epoch(model, test_loader, criterion, device)

    print("-" * 75)
    print("🎯 FINAL UNSEEN TEST RESULTS:")
    print(f"   Test Loss:     {test_loss:.5f}")
    print(f"   Test Accuracy: {test_acc * 100:.2f}%")
    print(f"   Test PR-AUC:   {test_pr_auc:.4f}")
    print(f"   Test F1-Score: {test_f1:.4f}")
    print("=" * 75)
