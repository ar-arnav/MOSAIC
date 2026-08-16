import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from LSTM_ZTF import LSTM_ZTF

# Set global styles for publication-quality figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Setup dedicated directory for results
RESULTS_DIR = os.path.expanduser("~/MOSAIC/results/LSTM_data")
os.makedirs(RESULTS_DIR, exist_ok=True)
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "test_predictions.csv")

def run_inference_only(parquet_path='data/processed/test.parquet', model_path='best_kilonova_lstm.pt', max_len=20, min_len=3):
    print("🚀 Running Inference (Model Evaluation)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_ZTF(input_size=2, hidden_size=64, num_layers=2, output_size=1, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    df = pd.read_parquet(parquet_path)
    if 'subclass' in df.columns:
        df['target'] = (df['subclass'] == 'Kilonovae').astype(int)
    else:
        df['target'] = df['label_class']

    grouped = df.groupby('group_id')
    kn_lengths = [len(g) for _, g in grouped if g['target'].iloc[0] == 1]
    effective_max_len = min(int(np.percentile(kn_lengths, 95)), max_len) if kn_lengths else max_len

    predictions = []

    print("Processing test set...")
    with torch.no_grad():
        for group_id, group_df in grouped:
            mag = group_df['magpsf'].values.astype(np.float32)
            time = group_df['time_day'].values.astype(np.float32)
            target = group_df['target'].iloc[0]

            if len(mag) < min_len:
                continue

            if len(mag) > effective_max_len:
                mag = mag[:effective_max_len]
                time = time[:effective_max_len]

            delta_mag = (mag - mag[0]) / 5.0  
            dt = np.diff(time, prepend=time[0]) / 30.0
            
            features = np.stack([delta_mag, dt], axis=1)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            length_tensor = torch.tensor([len(features)], dtype=torch.int64).to("cpu")

            logit = model(features_tensor, length_tensor)
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob >= 0.5 else 0

            true_label = 'Kilonova' if target == 1 else 'Imposter'
            pred_label = 'Kilonova' if pred == 1 else 'Imposter'

            if pred == 1 and target == 1: category = 'TP'
            elif pred == 0 and target == 0: category = 'TN'
            elif pred == 1 and target == 0: category = 'FP'
            else: category = 'FN'

            # Save ONLY the metrics and IDs (no heavy dataframes)
            predictions.append({
                'group_id': group_id,
                'true_label': true_label,
                'pred_label': pred_label,
                'category': category,
                'prob': prob,
                'Max_Delta_Mag': np.max(np.abs(delta_mag)),
                'Duration': time[-1] - time[0]
            })

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"✅ Inference complete. Saved to {PREDICTIONS_PATH}\n")
    return pred_df


def generate_figures_from_csv(parquet_path='test.parquet'):
    print("🎨 Generating figures from saved metrics...")
    
    # 1. Load the locked-in predictions
    if not os.path.exists(PREDICTIONS_PATH):
        print("ERROR: Predictions CSV not found. Run inference first!")
        return
        
    pred_df = pd.read_csv(PREDICTIONS_PATH)
    
    # 2. Load raw data ONLY to draw light curves when needed
    raw_df = pd.read_parquet(parquet_path)

    # ==========================================
    # FIGURE 1: PERFORMANCE OVERVIEW & PHYSICS
    # ==========================================
    print("Generating Figure 1: Model Performance & Feature Space...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cm_counts = [
        [len(pred_df[pred_df['category']=='TN']), len(pred_df[pred_df['category']=='FP'])], 
        [len(pred_df[pred_df['category']=='FN']), len(pred_df[pred_df['category']=='TP'])]
    ]
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Imposter', 'Predicted Kilonova'],
                yticklabels=['True Imposter', 'True Kilonova'], 
                ax=ax1, cbar=False, linewidths=2, linecolor='black')
    ax1.set_title('A. Confusion Matrix (Unseen Test Set)')

    kn_data = pred_df[pred_df['true_label'] == 'Kilonova']
    imp_data = pred_df[pred_df['true_label'] == 'Imposter']
    
    ax2.scatter(imp_data['Duration'], imp_data['Max_Delta_Mag'], c='gray', alpha=0.4, s=20, label='Imposters (All)')
    
    kn_correct = kn_data[kn_data['category']=='TP']
    kn_wrong = kn_data[kn_data['category']=='FN']
    ax2.scatter(kn_correct['Duration'], kn_correct['Max_Delta_Mag'], c='dodgerblue', edgecolors='black', s=50, label='Kilonovae (Correct)')
    ax2.scatter(kn_wrong['Duration'], kn_wrong['Max_Delta_Mag'], c='red', marker='X', s=100, label='Kilonovae (Missed)')
    
    ax2.set_xlabel('Event Duration (Days)')
    ax2.set_ylabel('Max |Δmag| (Normalized)')
    ax2.set_title('B. Physical Feature Space')
    ax2.legend()
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Figure1_Performance_and_Physics.png'))
    plt.close()

    # ==========================================
    # FIGURE 2: SUCCESSFUL DETECTIONS (True Positives)
    # ==========================================
    print("Generating Figure 2: Successful Kilonova Detections...")
    top_tps = pred_df[pred_df['category']=='TP'].sort_values(by='prob', ascending=False).head(6)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(top_tps.iterrows()):
        ax = axes[idx]
        grp_df = raw_df[raw_df['group_id'] == row['group_id']]
        
        ax.errorbar(grp_df['time_day'], grp_df['magpsf'], yerr=grp_df['sigmapsf'], 
                    fmt='o', color='dodgerblue', ecolor='gray', capsize=2, markersize=5)
        ax.invert_yaxis() 
        ax.set_title(f"Confidence: {row['prob']:.3f}", fontweight='bold')
        ax.set_xlabel('Time (Days)')
        if idx % 3 == 0:
            ax.set_ylabel('Magnitude (Brighter ↑)')
            
    fig.suptitle('Figure 2: Highest Confidence True Kilonova Detections', y=1.02, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Figure2_Successful_Detections.png'))
    plt.close()

    # ==========================================
    # FIGURE 3: EXPLANATORY ERROR ANALYSIS
    # ==========================================
    print("Generating Figure 3: Deep Error Analysis...")
    errors_to_plot = [
        ('False Positives (Imposters mistaken as Kilonovae)', 'FP', 'crimson'),
        ('False Negatives (Kilonovae missed)', 'FN', 'darkorange')
    ]

    for err_title, err_code, color in errors_to_plot:
        err_df = pred_df[pred_df['category'] == err_code]
        
        if err_code == 'FP':
            err_df = err_df.sort_values(by='prob', ascending=False).head(3)
        else:
            err_df = err_df.sort_values(by='prob', ascending=True).head(3)
            
        if err_df.empty:
            print(f"Skipping {err_title} - None found! (Amazing!)")
            continue
            
        n = len(err_df)
        fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
        if n == 1: axes = axes.reshape(2, 1) # Fix shape for single column
        
        for i, (_, row) in enumerate(err_df.iterrows()):
            grp_df = raw_df[raw_df['group_id'] == row['group_id']]
            time = grp_df['time_day'].values
            mag = grp_df['magpsf'].values
            delta_mag = (mag - mag[0]) / 5.0  # Replicate exact model input for bottom plot
            
            # Top Row: Raw Data
            ax1 = axes[0, i]
            ax1.errorbar(time, mag, yerr=grp_df['sigmapsf'], fmt='o', color=color, ecolor='gray', capsize=2, markersize=5)
            ax1.invert_yaxis()
            ax1.set_title(f"True: {row['true_label']}\nModel Prob: {row['prob']:.3f}")
            
            # Bottom Row: What the Model Sees
            ax2 = axes[1, i]
            ax2.plot(time, delta_mag, marker='s', color='black', linestyle='-', markersize=5)
            ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Model Input (Δmag)')
            ax2.set_xlabel('Time (Days)')
            
            subclass = grp_df['subclass'].iloc[0] if 'subclass' in grp_df.columns else 'Unknown'
            ax1.text(0.05, 0.05, f"Subclass: {subclass}", transform=ax1.transAxes, 
                     fontsize=9, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        axes[0,0].set_ylabel('Raw Magnitude (Brighter ↑)')
        axes[1,0].set_ylabel('Model Input: Δmag')
        
        fig.suptitle(f'Figure 3: Error Analysis - {err_title}', y=1.02, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        safe_filename = err_title.split('(')[0].strip().replace(" ", "_")
        plt.savefig(os.path.join(RESULTS_DIR, f'Figure3_{safe_filename}.png'))
        plt.close()

    print(f"\n✅ All publication-ready figures saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    # Update the path to point to your actual data location
    DATA_PATH = 'data/processed/test.parquet'
    
    # STEP 1: Run this once to generate the CSV. 
    # (Comment it out later if you just want to tweak graphs!)
    # run_inference_only(parquet_path=DATA_PATH)
    
    # STEP 2: Run this to generate graphs. You can run this 100 times safely.
    generate_figures_from_csv(parquet_path=DATA_PATH)