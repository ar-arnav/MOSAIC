import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve, 
    auc, 
    confusion_matrix, 
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# Import your custom modules
from dataset import MOSAICDataset
from model import MOSAICFusionMLP

# Set style for publication-grade matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

def evaluate_model(model, dataloader, criterion, device):
    """Evaluates model loss and collects predictions and probabilities."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            rtf = batch['rtf'].to(device)
            spatial = batch['spatial'].to(device)
            labels = batch['label'].to(device)
            
            # Model output is already a probability (due to Sigmoid in model class)
            outputs = model(rtf, spatial) 
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * rtf.size(0)
            
            # No torch.sigmoid needed here because model has Sigmoid at the end
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int) # Binary classification
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_probs), np.array(all_preds), np.array(all_labels)

def generate_isef_visualizations(history, y_true, y_probs, y_preds):
    """Generates and saves publication-quality figures for IRIS/ISEF judging."""
    os.makedirs("results/Fusion_MOSAIC/plots", exist_ok=True)
    
    # --- 1. Training & Validation Loss/Accuracy Curves ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    color = 'tab:red'
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (BCE)', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(epochs, history['train_loss'], color=color, linestyle='-', linewidth=2, label='Train Loss')
    line2 = ax1.plot(epochs, history['val_loss'], color=color, linestyle='--', linewidth=2, label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color, fontsize=12, fontweight='bold')
    line3 = ax2.plot(epochs, history['val_acc'], color=color, linestyle='-', linewidth=2, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.0, 1.05)
    
    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True)
    
    plt.title('MOSAIC Fusion Model: Training Convergence & Validation Metrics', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('results/Fusion_MOSAIC/plots/training_curves.png', dpi=300)
    plt.close()
    print("Saved -> results/Fusion_MOSAIC/plots/training_curves.png")

    # --- 2. ROC Curve & AUC Score ---
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) - Test Set', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig('results/Fusion_MOSAIC/plots/roc_curve.png', dpi=300)
    plt.close()
    print("Saved -> results/Fusion_MOSAIC/plots/roc_curve.png")

    # --- 3. Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='forestgreen', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curve (Transient Imbalance Evaluation)', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower left", frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig('results/Fusion_MOSAIC/plots/precision_recall_curve.png', dpi=300)
    plt.close()
    print("Saved -> results/Fusion_MOSAIC/plots/precision_recall_curve.png")

    # --- 4. Confusion Matrix Heatmap ---
    cm = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Background', 'Kilonova'], 
                yticklabels=['Background', 'Kilonova'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.xlabel('Predicted Classification', fontsize=12, fontweight='bold')
    plt.ylabel('True Physical Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix (Test Split)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('results/Fusion_MOSAIC/plots/confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved -> results/Fusion_MOSAIC/plots/confusion_matrix.png")

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing MOSAIC Stream 3 Fusion Training on: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # --- UPDATED DATA PATH ---
    data_path = 'data/processed/full_pipeline'
    
    train_dataset = MOSAICDataset(split='train', data_dir=data_path)
    val_dataset = MOSAICDataset(split='val', data_dir=data_path)
    test_dataset = MOSAICDataset(split='test', data_dir=data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    model = MOSAICFusionMLP(rtf_dim=128, spatial_dim=6, dropout_rate=0.3).to(device)
    
    # --- UPDATED CRITERION ---
    # Using BCELoss because model has Sigmoid at the end (outputs probabilities)
    criterion = nn.BCELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    os.makedirs("models", exist_ok=True)
    best_val_loss = float('inf')
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\n--- Beginning Training Loop ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            rtf = batch['rtf'].to(device)
            spatial = batch['spatial'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(rtf, spatial)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * rtf.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        val_loss, val_probs, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
        val_acc = (val_preds == val_labels).mean()
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/mosaic_fusion_best.pt")
            
    print("\nTraining complete. Best model checkpoint saved to 'models/mosaic_fusion_best.pt'.")
    
    # --- RIGOROUS TEST SET EVALUATION ---
    print("\nRunning comprehensive evaluation on Independent Test Set...")
    model.load_state_dict(torch.load("models/mosaic_fusion_best.pt"))
    test_loss, test_probs, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)
    
    # Calculate performance metrics
    test_acc = (test_preds == test_labels).mean()
    precision = precision_score(test_labels, test_preds, zero_division=0)
    recall = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    roc_auc = roc_auc_score(test_labels, test_probs)
    
    print("==================================================")
    print("          FINAL ISEF EVALUATION REPORT             ")
    print("==================================================")
    print(f"Test Loss        : {test_loss:.4f}")
    print(f"Test Accuracy    : {test_acc * 100:.2f}%")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score         : {f1:.4f}")
    print(f"ROC-AUC Score    : {roc_auc:.4f}")
    print("==================================================")
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Background', 'Kilonova']))
    
    # Generate Plots for ISEF presentation/paper
    print("\nGenerating publication-quality plots...")
    generate_isef_visualizations(history, test_labels, test_probs, test_preds)
    print("All diagnostic visualizations successfully exported to 'results/Fusion_MOSAIC/plots' directory.")

if __name__ == "__main__":
    train_pipeline()