import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc, 
    confusion_matrix, classification_report, precision_score, 
    recall_score, f1_score
)

# Import the Optical-Only model
from model_ablation import MOSAICOpticalOnly
from dataset import MOSAICDataset

# Style settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            rtf = batch['rtf'].to(device)
            spatial = batch['spatial'].to(device) # Ignored by model
            labels = batch['label'].to(device)
            
            outputs = model(rtf, spatial) 
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * rtf.size(0)
            
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_probs), np.array(all_preds), np.array(all_labels)

def generate_plots(history, y_true, y_probs, y_preds, title_suffix=""):
    os.makedirs("results/Fusion_MOSAIC/plots_optical_only", exist_ok=True)
    
    # 1. Training Curves
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (BCE)', color=color)
    ax1.plot(epochs, history['train_loss'], color=color, linestyle='-', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color=color, linestyle='--', linewidth=2, label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, history['val_acc'], color=color, linestyle='-', linewidth=2, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.0, 1.05)
    
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    plt.title(f'Optical-Only Model: Training Convergence{title_suffix}')
    plt.tight_layout()
    plt.savefig(f'results/Fusion_MOSAIC/plots_optical_only/training_curves{title_suffix}.png', dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Optical Only){title_suffix}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'results/Fusion_MOSAIC/plots_optical_only/roc_curve{title_suffix}.png', dpi=300)
    plt.close()

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Bkg', 'KN'], yticklabels=['Bkg', 'KN'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Optical Only){title_suffix}')
    plt.tight_layout()
    plt.savefig(f'results/Fusion_MOSAIC/plots_optical_only/confusion_matrix{title_suffix}.png', dpi=300)
    plt.close()

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Optical-Only Training on: {device}")
    
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    
    data_path = 'data/processed/full_pipeline'
    
    train_dataset = MOSAICDataset(split='train', data_dir=data_path)
    val_dataset = MOSAICDataset(split='val', data_dir=data_path)
    test_dataset = MOSAICDataset(split='test', data_dir=data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MOSAICOpticalOnly(rtf_dim=128, dropout_rate=0.3).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    os.makedirs("models", exist_ok=True)
    best_val_loss = float('inf')
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print("\n--- Beginning Optical-Only Training Loop ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            rtf = batch['rtf'].to(device)
            spatial = batch['spatial'].to(device) # Ignored by model
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(rtf, spatial)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * rtf.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        val_loss, _, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
        val_acc = (val_preds == val_labels).mean()
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/mosaic_optical_only_best.pt")
            
    print("\nTraining complete.")
    
    print("\nRunning evaluation on Independent Test Set...")
    model.load_state_dict(torch.load("models/mosaic_optical_only_best.pt"))
    test_loss, test_probs, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)
    
    test_acc = (test_preds == test_labels).mean()
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    roc_auc = roc_auc_score(test_labels, test_probs)
    
    print("==================================================")
    print("       OPTICAL-ONLY ABLATION RESULTS             ")
    print("==================================================")
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print(f"F1-Score      : {f1:.4f}")
    print(f"ROC-AUC        : {roc_auc:.4f}")
    print("==================================================")
    print(classification_report(test_labels, test_preds, target_names=['Background', 'Kilonova']))
    
    generate_plots(history, test_labels, test_probs, test_preds, title_suffix=" (Optical Only)")
    print("Optical plots exported to 'results/Fusion_MOSAIC/plots_optical_only/' directory.")

if __name__ == "__main__":
    train_pipeline()