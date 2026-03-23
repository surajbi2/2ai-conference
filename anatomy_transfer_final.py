# ===============================
# CROSS-ANATOMY TRANSFER ANALYSIS
# ===============================

import os
import warnings
import random
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix, 
                             roc_curve, precision_recall_curve, average_precision_score)
from scipy.stats import ttest_rel, shapiro
from tqdm import tqdm
from datetime import datetime

torch.backends.cudnn.benchmark = True
sns.set_style("whitegrid")


# ===============================
# CONFIGURATION
# ===============================

class Config:
    """Experimental configuration"""
    
    # Paths - MODIFY THESE FOR YOUR SYSTEM
    BASE_PATH = r"C:\Users\Suraj\Documents\python\MURA-v1.1"
    TRAIN_ROOT = os.path.join(BASE_PATH, "train")
    VALID_ROOT = os.path.join(BASE_PATH, "valid")
    
    # Source and target anatomies
    SOURCE_ANATOMY = "XR_WRIST"
    TARGET_ANATOMIES = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER", "XR_FINGER"]
    
    # Multi-seed configuration
    RANDOM_SEEDS = [42, 123, 456, 789, 2024]
    
    # Training parameters
    EPOCHS_SOURCE = 10
    EPOCHS_FINETUNE = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    
    # Sample efficiency parameters
    SAMPLE_SIZES = [10, 25, 50, 75, 100, 150, 200, 500]
    SAMPLE_EFFICIENCY_RUNS = 3
    SAMPLE_EFFICIENCY_ANATOMIES = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER"]
    
    # Operating point thresholds
    THRESHOLDS = [0.3, 0.5, 0.7]


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===============================
# DATASET
# ===============================

class MURADataset(Dataset):
    """MURA dataset with anatomy-specific loading"""
    
    def __init__(self, root_dir, anatomy):
        self.samples = []
        anatomy_dir = os.path.join(root_dir, anatomy)

        if not os.path.exists(anatomy_dir):
            raise FileNotFoundError(f"Missing path: {anatomy_dir}")

        for patient in os.listdir(anatomy_dir):
            patient_path = os.path.join(anatomy_dir, patient)
            if not os.path.isdir(patient_path):
                continue

            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue

                study_lower = study.lower()
                if "positive" in study_lower:
                    label = 1.0
                elif "negative" in study_lower:
                    label = 0.0
                else:
                    continue

                for img in os.listdir(study_path):
                    if img.endswith(".png"):
                        self.samples.append(
                            (os.path.join(study_path, img), label)
                        )

        print(f"[INFO] {anatomy}: {len(self.samples)} images loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((224, 224), dtype=np.uint8)
        else:
            image = cv2.resize(image, (224, 224))

        image = image.astype(np.float32) / 255.0
        image = np.stack([image] * 3, axis=0)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32)
        )


# ===============================
# MODEL
# ===============================

from torchvision.models import densenet121, DenseNet121_Weights

def get_densenet121():
    """Get DenseNet121 with modified classifier"""
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


# ===============================
# TRAINING & EVALUATION
# ===============================

def train_epoch(model, loader, optimizer, device):
    """Single training epoch"""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for x, y in tqdm(loader, leave=False, desc="Training"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_detailed(model, loader, device):
    """
    Comprehensive evaluation with all metrics
    Returns: Dictionary with predictions, targets, and computed metrics
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc="Evaluating"):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs.tolist())
            all_targets.extend(y.numpy().flatten().tolist())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Compute metrics
    if len(np.unique(targets)) < 2:
        return {
            'preds': preds,
            'targets': targets,
            'auc': float('nan'),
            'accuracy': accuracy_score(targets, preds > 0.5),
            'sensitivity': float('nan'),
            'specificity': float('nan'),
        }

    # Binary predictions at 0.5 threshold
    preds_binary = (preds >= 0.5).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, preds_binary).ravel()
    
    # Metrics
    auc = roc_auc_score(targets, preds)
    accuracy = accuracy_score(targets, preds_binary)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'preds': preds,
        'targets': targets,
        'auc': auc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


# ===============================
# MULTI-SEED CROSS-ANATOMY EVALUATION
# ===============================

def cross_anatomy_evaluation(config, device, seed, output_dir):
    """
    Train on source, evaluate on all anatomies
    
    Args:
        config: Configuration object
        device: torch device
        seed: Random seed for this run
        output_dir: Directory to save results
    
    Returns:
        Dictionary of results per anatomy
    """
    print(f"\n{'='*60}")
    print(f"SEED {seed}: Training on {config.SOURCE_ANATOMY}")
    print(f"{'='*60}")
    
    set_seed(seed)
    
    # Load source data
    train_set = MURADataset(config.TRAIN_ROOT, config.SOURCE_ANATOMY)
    val_set = MURADataset(config.VALID_ROOT, config.SOURCE_ANATOMY)
    
    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, 
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    # Train model
    model = get_densenet121().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print(f"\nTraining for {config.EPOCHS_SOURCE} epochs...")
    for epoch in range(config.EPOCHS_SOURCE):
        loss = train_epoch(model, train_loader, optimizer, device)
        if (epoch + 1) % 2 == 0:
            result = evaluate_detailed(model, val_loader, device)
            print(f"Epoch {epoch+1:02d} | Loss {loss:.4f} | "
                  f"Source AUC {result['auc']:.4f}")
    
    # Save trained model
    model_path = os.path.join(output_dir, f'source_model_seed{seed}.pth')
    torch.save(model.state_dict(), model_path)
    
    # Evaluate on all anatomies
    results = {}
    all_anatomies = [config.SOURCE_ANATOMY] + config.TARGET_ANATOMIES
    
    for anatomy in all_anatomies:
        print(f"\nEvaluating on {anatomy}...")
        try:
            test_set = MURADataset(config.VALID_ROOT, anatomy)
            test_loader = DataLoader(
                test_set, batch_size=config.BATCH_SIZE, shuffle=False,
                num_workers=config.NUM_WORKERS, pin_memory=True
            )
            
            result = evaluate_detailed(model, test_loader, device)
            results[anatomy] = result
            
            print(f"  {anatomy}: AUC {result['auc']:.4f} | "
                  f"Acc {result['accuracy']:.4f} | "
                  f"Sens {result['sensitivity']:.4f} | "
                  f"Spec {result['specificity']:.4f}")
            
        except FileNotFoundError:
            print(f"  [WARNING] {anatomy} not found, skipping...")
            continue
    
    return model, results


# ===============================
# SAMPLE EFFICIENCY ANALYSIS
# ===============================

def sample_efficiency_experiment(base_model, config, device, output_dir):
    """
    Test how many target samples needed to recover performance
    
    Args:
        base_model: Trained source model
        config: Configuration object
        device: torch device
        output_dir: Directory to save results
    
    Returns:
        DataFrame with sample efficiency results
    """
    print(f"\n{'='*60}")
    print("SAMPLE EFFICIENCY ANALYSIS")
    print(f"{'='*60}")
    
    all_results = []
    
    for anatomy in config.SAMPLE_EFFICIENCY_ANATOMIES:
        print(f"\n{anatomy}:")
        
        try:
            # Load target data
            train_set = MURADataset(config.TRAIN_ROOT, anatomy)
            val_set = MURADataset(config.VALID_ROOT, anatomy)
            
            val_loader = DataLoader(
                val_set, batch_size=config.BATCH_SIZE, shuffle=False,
                num_workers=config.NUM_WORKERS, pin_memory=True
            )
            
            # Baseline: zero-shot (no fine-tuning)
            print("  Zero-shot (no fine-tuning)...")
            base_model.eval()
            zeroshot_result = evaluate_detailed(base_model, val_loader, device)
            
            all_results.append({
                'anatomy': anatomy,
                'n_samples': 0,
                'run': 0,
                'auc': zeroshot_result['auc'],
                'accuracy': zeroshot_result['accuracy'],
                'sensitivity': zeroshot_result['sensitivity'],
                'specificity': zeroshot_result['specificity'],
            })
            
            # Test different sample sizes
            for n_samples in config.SAMPLE_SIZES:
                if n_samples > len(train_set):
                    print(f"  Skipping {n_samples} (exceeds dataset size {len(train_set)})")
                    continue
                
                # Multiple runs per sample size
                for run_idx in range(config.SAMPLE_EFFICIENCY_RUNS):
                    print(f"  {n_samples} samples | Run {run_idx+1}/{config.SAMPLE_EFFICIENCY_RUNS}")
                    
                    # Create fresh model with source weights
                    model = get_densenet121().to(device)
                    model.load_state_dict(base_model.state_dict())
                    
                    # Random subset of training data
                    set_seed(config.RANDOM_SEEDS[0] + run_idx)  # Consistent but different per run
                    indices = np.random.choice(len(train_set), n_samples, replace=False)
                    subset = Subset(train_set, indices)
                    
                    train_loader = DataLoader(
                        subset, batch_size=config.BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=True
                    )
                    
                    # Fine-tune
                    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
                    for epoch in range(config.EPOCHS_FINETUNE):
                        train_epoch(model, train_loader, optimizer, device)
                    
                    # Evaluate
                    result = evaluate_detailed(model, val_loader, device)
                    
                    all_results.append({
                        'anatomy': anatomy,
                        'n_samples': n_samples,
                        'run': run_idx,
                        'auc': result['auc'],
                        'accuracy': result['accuracy'],
                        'sensitivity': result['sensitivity'],
                        'specificity': result['specificity'],
                    })
                    
                    print(f"    AUC: {result['auc']:.4f}")
        
        except FileNotFoundError:
            print(f"  [WARNING] {anatomy} not found in training set, skipping...")
            continue
    
    df = pd.DataFrame(all_results)
    
    # Save results
    efficiency_path = os.path.join(output_dir, 'sample_efficiency_results.csv')
    df.to_csv(efficiency_path, index=False)
    print(f"\n[SAVED] Sample efficiency: {efficiency_path}")
    
    return df


# ===============================
# STATISTICAL ANALYSIS
# ===============================

def statistical_analysis(multi_seed_results_df, config, output_dir):
    """
    Perform statistical tests comparing source to each target
    
    Args:
        multi_seed_results_df: DataFrame with all multi-seed results
        config: Configuration object
        output_dir: Directory to save results
    
    Returns:
        DataFrame with statistical test results
    """
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    results = []
    
    # Get source performance across seeds
    source_data = multi_seed_results_df[
        multi_seed_results_df['anatomy'] == config.SOURCE_ANATOMY
    ]
    source_aucs = source_data['auc'].values
    
    print(f"\nSource ({config.SOURCE_ANATOMY}):")
    print(f"  Mean AUC: {np.mean(source_aucs):.4f} ± {np.std(source_aucs):.4f}")
    print(f"  95% CI: [{np.mean(source_aucs) - 1.96*np.std(source_aucs)/np.sqrt(len(source_aucs)):.4f}, "
          f"{np.mean(source_aucs) + 1.96*np.std(source_aucs)/np.sqrt(len(source_aucs)):.4f}]")
    
    # Bonferroni correction
    n_comparisons = len(config.TARGET_ANATOMIES)
    bonferroni_alpha = 0.05 / n_comparisons
    
    print(f"\nBonferroni-corrected α = 0.05/{n_comparisons} = {bonferroni_alpha:.4f}")
    
    # Compare each target to source
    for target in config.TARGET_ANATOMIES:
        target_data = multi_seed_results_df[
            multi_seed_results_df['anatomy'] == target
        ]
        
        if len(target_data) == 0:
            continue
        
        target_aucs = target_data['auc'].values
        
        # Check if we have matching number of samples
        if len(source_aucs) != len(target_aucs):
            print(f"\n[WARNING] {target}: Mismatched sample sizes, skipping t-test")
            continue
        
        # Paired t-test
        t_stat, p_value = ttest_rel(source_aucs, target_aucs)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(source_aucs) - np.mean(target_aucs)
        pooled_std = np.sqrt((np.std(source_aucs)**2 + np.std(target_aucs)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # 95% CI for target
        ci_lower = np.mean(target_aucs) - 1.96*np.std(target_aucs)/np.sqrt(len(target_aucs))
        ci_upper = np.mean(target_aucs) + 1.96*np.std(target_aucs)/np.sqrt(len(target_aucs))
        
        significant = p_value < bonferroni_alpha
        
        results.append({
            'comparison': f'{config.SOURCE_ANATOMY} vs {target}',
            'source_mean': np.mean(source_aucs),
            'source_std': np.std(source_aucs),
            'target_mean': np.mean(target_aucs),
            'target_std': np.std(target_aucs),
            'target_ci_lower': ci_lower,
            'target_ci_upper': ci_upper,
            'auc_drop': mean_diff,
            'auc_drop_pct': (mean_diff / np.mean(source_aucs)) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'bonferroni_alpha': bonferroni_alpha,
            'significant': significant,
            'cohens_d': cohens_d,
        })
        
        print(f"\n{target}:")
        print(f"  Mean AUC: {np.mean(target_aucs):.4f} ± {np.std(target_aucs):.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Drop: {mean_diff:.4f} ({mean_diff/np.mean(source_aucs)*100:.1f}%)")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f} {'***' if significant else ''}")
        print(f"  Cohen's d: {cohens_d:.4f}")
    
    df = pd.DataFrame(results)
    
    # Save
    stats_path = os.path.join(output_dir, 'statistical_tests.csv')
    df.to_csv(stats_path, index=False)
    print(f"\n[SAVED] Statistical tests: {stats_path}")
    
    return df


# ===============================
# CLINICAL METRICS
# ===============================

def false_negative_analysis(multi_seed_results, output_dir):
    """
    Analyze false negatives (missed fractures)
    
    Args:
        multi_seed_results: Dictionary with {seed: {anatomy: result}}
        output_dir: Directory to save results
    
    Returns:
        DataFrame with FN analysis
    """
    print(f"\n{'='*60}")
    print("FALSE NEGATIVE ANALYSIS")
    print(f"{'='*60}")
    
    fn_results = []
    
    # Aggregate across seeds
    anatomy_results = {}
    
    for seed, results in multi_seed_results.items():
        for anatomy, result in results.items():
            if anatomy not in anatomy_results:
                anatomy_results[anatomy] = []
            anatomy_results[anatomy].append(result)
    
    # Compute FN metrics per anatomy
    for anatomy, results_list in anatomy_results.items():
        # Combine predictions across seeds
        all_preds = np.concatenate([r['preds'] for r in results_list])
        all_targets = np.concatenate([r['targets'] for r in results_list])
        
        # False negatives
        fn_mask = (all_targets == 1) & (all_preds < 0.5)
        fn_count = np.sum(fn_mask)
        total_positives = np.sum(all_targets == 1)
        fn_rate = (fn_count / total_positives * 100) if total_positives > 0 else 0
        
        # High-confidence false negatives (pred < 0.3)
        high_conf_fn = np.sum((all_targets == 1) & (all_preds < 0.3))
        high_conf_fn_rate = (high_conf_fn / fn_count * 100) if fn_count > 0 else 0
        
        fn_results.append({
            'anatomy': anatomy,
            'total_fractures': int(total_positives),
            'false_negatives': int(fn_count),
            'fn_rate': fn_rate,
            'high_conf_fn': int(high_conf_fn),
            'high_conf_fn_pct': high_conf_fn_rate,
        })
        
        print(f"\n{anatomy}:")
        print(f"  Total fractures: {int(total_positives)}")
        print(f"  False negatives: {int(fn_count)} ({fn_rate:.1f}%)")
        print(f"  High-confidence FN: {int(high_conf_fn)} ({high_conf_fn_rate:.1f}% of FN)")
    
    df = pd.DataFrame(fn_results)
    
    # Save
    fn_path = os.path.join(output_dir, 'false_negative_analysis.csv')
    df.to_csv(fn_path, index=False)
    print(f"\n[SAVED] FN analysis: {fn_path}")
    
    return df


def compute_operating_points(multi_seed_results, config, output_dir):
    """
    Compute sensitivity/specificity at different thresholds
    
    Args:
        multi_seed_results: Dictionary with {seed: {anatomy: result}}
        config: Configuration object
        output_dir: Directory to save results
    
    Returns:
        DataFrame with operating point metrics
    """
    print(f"\n{'='*60}")
    print("OPERATING POINT ANALYSIS")
    print(f"{'='*60}")
    
    op_results = []
    
    # Aggregate across seeds
    anatomy_results = {}
    for seed, results in multi_seed_results.items():
        for anatomy, result in results.items():
            if anatomy not in anatomy_results:
                anatomy_results[anatomy] = []
            anatomy_results[anatomy].append(result)
    
    # Compute metrics at each threshold
    for anatomy, results_list in anatomy_results.items():
        # Combine across seeds
        all_preds = np.concatenate([r['preds'] for r in results_list])
        all_targets = np.concatenate([r['targets'] for r in results_list])
        
        for threshold in config.THRESHOLDS:
            preds_binary = (all_preds >= threshold).astype(int)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(all_targets, preds_binary).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            op_results.append({
                'anatomy': anatomy,
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'fn_count': int(fn),
                'fp_count': int(fp),
            })
    
    df = pd.DataFrame(op_results)
    
    # Save
    op_path = os.path.join(output_dir, 'operating_points.csv')
    df.to_csv(op_path, index=False)
    print(f"\n[SAVED] Operating points: {op_path}")
    
    return df


# ===============================
# VISUALIZATION
# ===============================

def plot_roc_curves(multi_seed_results, output_dir):
    """
    Generate ROC curves for all anatomies
    
    Args:
        multi_seed_results: Dictionary with {seed: {anatomy: result}}
        output_dir: Directory to save plot
    """
    print("\n[INFO] Generating ROC curves...")
    
    # Aggregate results across seeds
    anatomy_results = {}
    for seed, results in multi_seed_results.items():
        for anatomy, result in results.items():
            if anatomy not in anatomy_results:
                anatomy_results[anatomy] = []
            anatomy_results[anatomy].append(result)
    
    # Create plot
    n_anatomies = len(anatomy_results)
    n_cols = 3
    n_rows = (n_anatomies + 1 + n_cols - 1) // n_cols  # +1 for combined plot
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_anatomies > 1 else [axes]
    
    # Individual ROC curves
    for idx, (anatomy, results_list) in enumerate(anatomy_results.items()):
        ax = axes[idx]
        
        # Combine predictions across seeds
        all_preds = np.concatenate([r['preds'] for r in results_list])
        all_targets = np.concatenate([r['targets'] for r in results_list])
        
        # Compute ROC
        fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
        roc_auc = roc_auc_score(all_targets, all_preds)
        
        # Plot
        ax.plot(fpr, tpr, linewidth=2.5, label=f'AUC = {roc_auc:.3f}', color='darkblue')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        # Optimal point (Youden's index)
        optimal_idx = np.argmax(tpr - fpr)
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
                label=f'Optimal (t={thresholds[optimal_idx]:.2f})')
        
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title(f'{anatomy}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
    
    # Combined plot
    if n_anatomies < len(axes):
        ax = axes[n_anatomies]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(anatomy_results)))
        
        for (anatomy, results_list), color in zip(anatomy_results.items(), colors):
            all_preds = np.concatenate([r['preds'] for r in results_list])
            all_targets = np.concatenate([r['targets'] for r in results_list])
            
            fpr, tpr, _ = roc_curve(all_targets, all_preds)
            roc_auc = roc_auc_score(all_targets, all_preds)
            
            ax.plot(fpr, tpr, linewidth=2.5, 
                   label=f'{anatomy} ({roc_auc:.3f})', color=color)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title('All Anatomies (Combined)', fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
    
    # Hide unused subplots
    for idx in range(n_anatomies + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('ROC Curves: Cross-Anatomy Transfer Performance', 
                 fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] ROC curves: {save_path}")


def plot_precision_recall_curves(multi_seed_results, output_dir):
    """
    Generate precision-recall curves for all anatomies
    
    Args:
        multi_seed_results: Dictionary with {seed: {anatomy: result}}
        output_dir: Directory to save plot
    """
    print("\n[INFO] Generating precision-recall curves...")
    
    # Aggregate results
    anatomy_results = {}
    for seed, results in multi_seed_results.items():
        for anatomy, result in results.items():
            if anatomy not in anatomy_results:
                anatomy_results[anatomy] = []
            anatomy_results[anatomy].append(result)
    
    # Create plot
    n_anatomies = len(anatomy_results)
    n_cols = 3
    n_rows = (n_anatomies + 1 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_anatomies > 1 else [axes]
    
    # Individual PR curves
    for idx, (anatomy, results_list) in enumerate(anatomy_results.items()):
        ax = axes[idx]
        
        all_preds = np.concatenate([r['preds'] for r in results_list])
        all_targets = np.concatenate([r['targets'] for r in results_list])
        
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(all_targets, all_preds)
        ap_score = average_precision_score(all_targets, all_preds)
        prevalence = np.mean(all_targets)
        
        # Plot
        ax.plot(recall, precision, linewidth=2.5, 
               label=f'AP = {ap_score:.3f}', color='darkgreen')
        ax.axhline(y=prevalence, color='r', linestyle='--', linewidth=1.5,
                  label=f'Baseline = {prevalence:.3f}')
        
        ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
        ax.set_title(f'{anatomy}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
    
    # Combined plot
    if n_anatomies < len(axes):
        ax = axes[n_anatomies]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(anatomy_results)))
        
        for (anatomy, results_list), color in zip(anatomy_results.items(), colors):
            all_preds = np.concatenate([r['preds'] for r in results_list])
            all_targets = np.concatenate([r['targets'] for r in results_list])
            
            precision, recall, _ = precision_recall_curve(all_targets, all_preds)
            ap_score = average_precision_score(all_targets, all_preds)
            
            ax.plot(recall, precision, linewidth=2.5,
                   label=f'{anatomy} ({ap_score:.3f})', color=color)
        
        ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
        ax.set_title('All Anatomies (Combined)', fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
    
    # Hide unused subplots
    for idx in range(n_anatomies + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Precision-Recall Curves: Cross-Anatomy Transfer Performance',
                 fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'precision_recall_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] PR curves: {save_path}")


def plot_auc_comparison(multi_seed_df, config, output_dir):
    """
    Bar chart comparing AUC across anatomies with error bars
    
    Args:
        multi_seed_df: DataFrame with multi-seed results
        config: Configuration object
        output_dir: Directory to save plot
    """
    print("\n[INFO] Generating AUC comparison plot...")
    
    # Compute summary statistics
    summary = multi_seed_df.groupby('anatomy')['auc'].agg(['mean', 'std']).reset_index()
    summary = summary.sort_values('mean', ascending=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color source differently
    colors = ['#2ecc71' if anat == config.SOURCE_ANATOMY else '#e74c3c' 
              for anat in summary['anatomy']]
    
    bars = ax.bar(summary['anatomy'], summary['mean'],
                  yerr=summary['std'], capsize=7,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, mean, std in zip(bars, summary['mean'], summary['std']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('AUC-ROC', fontsize=15, fontweight='bold')
    ax.set_xlabel('Anatomy', fontsize=15, fontweight='bold')
    ax.set_title('Cross-Anatomy Transfer Performance (Mean ± Std over 5 seeds)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2,
              label=f'Source ({config.SOURCE_ANATOMY})'),
        Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2,
              label='Target (Zero-Shot Transfer)')
    ]
    ax.legend(handles=legend_elements, fontsize=13, loc='lower left')
    
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'auc_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] AUC comparison: {save_path}")


def plot_false_negative_analysis(fn_df, config, output_dir):
    """
    Visualize false negative rates across anatomies
    
    Args:
        fn_df: DataFrame with FN analysis
        config: Configuration object
        output_dir: Directory to save plot
    """
    print("\n[INFO] Generating false negative analysis plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: FN rates
    ax = axes[0]
    
    colors = ['#2ecc71' if anat == config.SOURCE_ANATOMY else '#e74c3c'
              for anat in fn_df['anatomy']]
    
    bars = ax.bar(fn_df['anatomy'], fn_df['fn_rate'],
                  color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add source baseline line
    source_fn = fn_df[fn_df['anatomy'] == config.SOURCE_ANATOMY]['fn_rate'].values[0]
    ax.axhline(y=source_fn, color='green', linestyle='--', linewidth=2.5,
              label=f'Source ({config.SOURCE_ANATOMY}): {source_fn:.1f}%', alpha=0.8)
    
    ax.set_ylabel('False Negative Rate (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Anatomy', fontsize=14, fontweight='bold')
    ax.set_title('Missed Fracture Rate by Anatomy', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, fn_df['fn_rate']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Breakdown by confidence
    ax = axes[1]
    
    x = np.arange(len(fn_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fn_df['false_negatives'],
                   width, label='All False Negatives',
                   color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, fn_df['high_conf_fn'],
                   width, label='High-Confidence FN (pred<0.3)',
                   color='#c0392b', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_xlabel('Anatomy', fontsize=14, fontweight='bold')
    ax.set_title('False Negative Breakdown', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(fn_df['anatomy'], rotation=30, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Clinical Impact: False Negative Analysis',
                 fontsize=17, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'false_negative_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] FN analysis plot: {save_path}")


def plot_sample_efficiency(efficiency_df, config, output_dir):
    """
    Plot sample efficiency curves for multiple anatomies
    
    Args:
        efficiency_df: DataFrame with sample efficiency results
        config: Configuration object
        output_dir: Directory to save plot
    """
    print("\n[INFO] Generating sample efficiency plot...")
    
    # Source performance for reference
    source_auc = 0.879  # Hard-coded for now, should be computed
    
    anatomies = efficiency_df['anatomy'].unique()
    n_anatomies = len(anatomies)
    
    fig, axes = plt.subplots(1, n_anatomies, figsize=(6*n_anatomies, 5))
    if n_anatomies == 1:
        axes = [axes]
    
    colors_map = {'XR_ELBOW': '#3498db', 'XR_HAND': '#e74c3c', 'XR_SHOULDER': '#9b59b6'}
    
    for idx, anatomy in enumerate(anatomies):
        ax = axes[idx]
        
        # Get data for this anatomy
        data = efficiency_df[efficiency_df['anatomy'] == anatomy]
        
        # Compute mean and std per sample size
        grouped = data.groupby('n_samples')['auc'].agg(['mean', 'std']).reset_index()
        
        # Plot with error bars
        color = colors_map.get(anatomy, '#34495e')
        ax.errorbar(grouped['n_samples'], grouped['mean'], yerr=grouped['std'],
                   marker='o', linewidth=2.5, markersize=10, capsize=5,
                   color=color, label='Fine-tuned Model', alpha=0.9)
        
        # Source performance line
        ax.axhline(y=source_auc, color='green', linestyle='--', linewidth=2.5,
                  label=f'Source Performance ({source_auc:.3f})', alpha=0.8)
        
        # 90% and 95% recovery lines
        ax.axhline(y=source_auc*0.90, color='orange', linestyle=':', linewidth=2,
                  label='90% Recovery', alpha=0.6)
        ax.axhline(y=source_auc*0.95, color='red', linestyle=':', linewidth=2,
                  label='95% Recovery', alpha=0.6)
        
        ax.set_xlabel('Number of Target Samples', fontsize=13, fontweight='bold')
        ax.set_ylabel('AUC-ROC', fontsize=13, fontweight='bold')
        ax.set_title(f'{anatomy}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_ylim([0.65, 0.92])
        
        # Format x-axis
        ax.set_xscale('log')
        ax.set_xticks([10, 25, 50, 100, 200, 500])
        ax.set_xticklabels(['10', '25', '50', '100', '200', '500'])
    
    plt.suptitle('Sample Efficiency: Fine-Tuning on Target Anatomies',
                 fontsize=17, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'sample_efficiency.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Sample efficiency: {save_path}")


# ===============================
# MAIN PIPELINE
# ===============================

def main():
    # Initialize
    print("\n" + "="*70)
    print("CROSS-ANATOMY TRANSFER LEARNING - PRODUCTION PIPELINE")
    print("Conference-Ready Analysis")
    print("="*70)
    
    # Check GPU
    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
    device = torch.device("cuda")
    print(f"\n[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Configuration
    config = Config()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_conference_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Output directory: {output_dir}")
    
    # ================================================================
    # PHASE 1: MULTI-SEED CROSS-ANATOMY EVALUATION
    # ================================================================
    print("\n" + "="*70)
    print("PHASE 1: MULTI-SEED CROSS-ANATOMY EVALUATION")
    print("="*70)
    
    multi_seed_results = {}  # {seed: {anatomy: result}}
    multi_seed_data = []  # For DataFrame
    
    for seed_idx, seed in enumerate(config.RANDOM_SEEDS):
        print(f"\n[PROGRESS] Seed {seed_idx+1}/{len(config.RANDOM_SEEDS)}")
        
        # Train and evaluate
        model, results = cross_anatomy_evaluation(config, device, seed, output_dir)
        multi_seed_results[seed] = results
        
        # Store for DataFrame
        for anatomy, result in results.items():
            multi_seed_data.append({
                'seed': seed,
                'anatomy': anatomy,
                'auc': result['auc'],
                'accuracy': result['accuracy'],
                'sensitivity': result['sensitivity'],
                'specificity': result['specificity'],
                'ppv': result['ppv'],
                'npv': result['npv'],
                'tp': result['tp'],
                'tn': result['tn'],
                'fp': result['fp'],
                'fn': result['fn'],
            })
    
    # Save aggregated results
    multi_seed_df = pd.DataFrame(multi_seed_data)
    multi_seed_path = os.path.join(output_dir, 'multi_seed_results.csv')
    multi_seed_df.to_csv(multi_seed_path, index=False)
    print(f"\n[SAVED] Multi-seed results: {multi_seed_path}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY: Cross-Anatomy Performance")
    print("="*70)
    summary = multi_seed_df.groupby('anatomy')['auc'].agg(['mean', 'std', 'min', 'max'])
    print(summary.to_string())
    
    # ================================================================
    # PHASE 2: STATISTICAL ANALYSIS
    # ================================================================
    stats_df = statistical_analysis(multi_seed_df, config, output_dir)
    
    # ================================================================
    # PHASE 3: CLINICAL METRICS
    # ================================================================
    print("\n" + "="*70)
    print("PHASE 3: CLINICAL IMPACT ANALYSIS")
    print("="*70)
    
    # False negative analysis
    fn_df = false_negative_analysis(multi_seed_results, output_dir)
    
    # Operating points
    op_df = compute_operating_points(multi_seed_results, config, output_dir)
    
    # ================================================================
    # PHASE 4: SAMPLE EFFICIENCY
    # ================================================================
    print("\n" + "="*70)
    print("PHASE 4: SAMPLE EFFICIENCY ANALYSIS")
    print("="*70)
    
    # Use first seed's model as base
    first_seed = config.RANDOM_SEEDS[0]
    base_model_path = os.path.join(output_dir, f'source_model_seed{first_seed}.pth')
    base_model = get_densenet121().to(device)
    base_model.load_state_dict(torch.load(base_model_path))
    
    efficiency_df = sample_efficiency_experiment(base_model, config, device, output_dir)
    
    # ================================================================
    # PHASE 5: VISUALIZATION
    # ================================================================
    print("\n" + "="*70)
    print("PHASE 5: GENERATING PUBLICATION FIGURES")
    print("="*70)
    
    plot_auc_comparison(multi_seed_df, config, output_dir)
    plot_roc_curves(multi_seed_results, output_dir)
    plot_precision_recall_curves(multi_seed_results, output_dir)
    plot_false_negative_analysis(fn_df, config, output_dir)
    plot_sample_efficiency(efficiency_df, config, output_dir)
    
    # ================================================================
    # PHASE 6: FINAL SUMMARY
    # ================================================================
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\n[OUTPUT] All results saved to: {output_dir}")
    print("\n[FILES CREATED]")
    print("  Tables:")
    print(f"    - multi_seed_results.csv")
    print(f"    - statistical_tests.csv")
    print(f"    - false_negative_analysis.csv")
    print(f"    - operating_points.csv")
    print(f"    - sample_efficiency_results.csv")
    print("\n  Figures:")
    print(f"    - auc_comparison.png")
    print(f"    - roc_curves.png")
    print(f"    - precision_recall_curves.png")
    print(f"    - false_negative_analysis.png")
    print(f"    - sample_efficiency.png")
    
    # Executive summary
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)
    
    source_perf = multi_seed_df[multi_seed_df['anatomy'] == config.SOURCE_ANATOMY]['auc']
    target_perf = multi_seed_df[multi_seed_df['anatomy'] != config.SOURCE_ANATOMY]['auc']
    
    print(f"\nSource ({config.SOURCE_ANATOMY}):")
    print(f"  Mean AUC: {source_perf.mean():.4f} ± {source_perf.std():.4f}")
    
    print(f"\nTarget (Average across {len(config.TARGET_ANATOMIES)} anatomies):")
    print(f"  Mean AUC: {target_perf.mean():.4f} ± {target_perf.std():.4f}")
    
    drop = source_perf.mean() - target_perf.mean()
    drop_pct = (drop / source_perf.mean()) * 100
    print(f"\nPerformance Drop:")
    print(f"  Absolute: {drop:.4f} AUC points")
    print(f"  Relative: {drop_pct:.1f}%")
    
    # Clinical impact
    source_fn = fn_df[fn_df['anatomy'] == config.SOURCE_ANATOMY]['fn_rate'].values[0]
    target_fn = fn_df[fn_df['anatomy'] != config.SOURCE_ANATOMY]['fn_rate'].mean()
    fn_increase = target_fn - source_fn
    
    print(f"\nFalse Negative Rate:")
    print(f"  Source: {source_fn:.1f}%")
    print(f"  Target (avg): {target_fn:.1f}%")
    print(f"  Increase: +{fn_increase:.1f} percentage points")
    
    print("\n" + "="*70)
    print("READY FOR CONFERENCE SUBMISSION")
    print("="*70)
    print("\nNext steps:")
    print("1. Review all CSV files for accuracy")
    print("2. Examine all PNG figures for quality")
    print("3. Begin writing paper using these results")
    print("4. See publication_strategy.md for timeline")
    print("\n" + "="*70)


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
