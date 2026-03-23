# ===============================
# CROSS-ANATOMY TRANSFER ANALYSIS
# Complete diagnostic pipeline for fracture detection generalization
# ===============================

import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from tqdm import tqdm
import json
from datetime import datetime

torch.backends.cudnn.benchmark = True
sns.set_style("whitegrid")


# ===============================
# 1. DATASET
# ===============================

class MURADataset(Dataset):
    """MURA dataset with anatomy-specific loading"""
    
    def __init__(self, root_dir, anatomy, return_confidence=False):
        self.samples = []
        self.return_confidence = return_confidence
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
# 2. MODEL WITH FEATURE EXTRACTION
# ===============================

from torchvision.models import densenet121, DenseNet121_Weights

def get_densenet121():
    """Get DenseNet121 with modified classifier"""
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


class FeatureExtractor(nn.Module):
    """Extract features from penultimate layer"""
    
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class LayerwiseModel(nn.Module):
    """Model that allows freezing specific layers"""
    
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.classifier = base_model.classifier
        
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)
    
    def freeze_until_layer(self, layer_idx):
        """Freeze all layers up to layer_idx"""
        for i, (name, param) in enumerate(self.features.named_parameters()):
            if i < layer_idx:
                param.requires_grad = False
            else:
                param.requires_grad = True


# ===============================
# 3. TRAINING & EVALUATION
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
    Comprehensive evaluation with confidence scores
    Returns: AUC, Accuracy, predictions, targets, confidence scores
    """
    model.eval()
    preds, targets, confidences = [], [], []

    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc="Evaluating"):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            preds.extend(probs.tolist())
            targets.extend(y.numpy().flatten().tolist())
            # Confidence = distance from decision boundary (0.5)
            confidences.extend(np.abs(probs - 0.5).tolist())

    preds = np.array(preds)
    targets = np.array(targets)
    confidences = np.array(confidences)

    if len(np.unique(targets)) < 2:
        return {
            'auc': float('nan'),
            'acc': accuracy_score(targets, preds > 0.5),
            'preds': preds,
            'targets': targets,
            'confidences': confidences
        }

    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, preds > 0.5)
    
    return {
        'auc': auc,
        'acc': acc,
        'preds': preds,
        'targets': targets,
        'confidences': confidences
    }


# ===============================
# 4. FEATURE SPACE ANALYSIS
# ===============================

def extract_features(feature_model, loader, device, anatomy_name):
    """Extract features with labels"""
    feats, labels, fracture_labels = [], [], []
    
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc=f"Extracting {anatomy_name}"):
            x = x.to(device)
            f = feature_model(x).detach().cpu().numpy()
            feats.append(f)
            fracture_labels.append(y.numpy().flatten())
    
    return np.vstack(feats), np.concatenate(fracture_labels)


def compute_feature_distances(features_dict):
    """
    Compute inter-anatomy vs intra-anatomy feature distances
    
    Returns: DataFrame with distance statistics
    """
    results = []
    
    anatomy_names = list(features_dict.keys())
    
    for i, anat1 in enumerate(anatomy_names):
        feat1 = features_dict[anat1]['features']
        
        # Intra-anatomy distance
        intra_dist = cdist(feat1, feat1, metric='euclidean')
        intra_mean = np.mean(intra_dist[np.triu_indices_from(intra_dist, k=1)])
        
        for j, anat2 in enumerate(anatomy_names):
            if i >= j:
                continue
                
            feat2 = features_dict[anat2]['features']
            
            # Inter-anatomy distance
            inter_dist = cdist(feat1, feat2, metric='euclidean')
            inter_mean = np.mean(inter_dist)
            
            results.append({
                'anatomy_pair': f"{anat1} ↔ {anat2}",
                'intra_distance': intra_mean,
                'inter_distance': inter_mean,
                'distance_ratio': inter_mean / intra_mean
            })
    
    return pd.DataFrame(results)


def plot_feature_space(features_dict, save_path="feature_space.png"):
    """PCA visualization of feature space"""
    
    # Combine all features
    all_features = []
    all_labels = []
    all_fractures = []
    
    for anat_name, data in features_dict.items():
        all_features.append(data['features'])
        all_labels.extend([anat_name] * len(data['features']))
        all_fractures.append(data['fracture_labels'])
    
    X = np.vstack(all_features)
    y_anatomy = np.array(all_labels)
    y_fracture = np.concatenate(all_fractures)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # By anatomy
    ax = axes[0]
    for anat in features_dict.keys():
        mask = y_anatomy == anat
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  alpha=0.6, label=anat, s=30)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Feature Space: Colored by Anatomy")
    ax.legend()
    
    # By fracture status
    ax = axes[1]
    for frac_status, color, label in [(0, 'blue', 'No Fracture'), 
                                       (1, 'red', 'Fracture')]:
        mask = y_fracture == frac_status
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  alpha=0.4, label=label, c=color, s=30)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Feature Space: Colored by Fracture Status")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Feature space visualization: {save_path}")


# ===============================
# 5. LAYER-WISE TRANSFER ANALYSIS
# ===============================

def layerwise_transfer_experiment(base_model, train_loader, target_loader, 
                                   device, n_samples=100):
    """
    Test transfer performance when freezing different layer depths
    
    Returns: DataFrame with performance vs layer depth
    """
    results = []
    
    # Get total number of parameter groups
    total_params = len(list(base_model.features.parameters()))
    layer_configs = [0, total_params // 4, total_params // 2, 
                    3 * total_params // 4, total_params]
    
    print("\n[INFO] Layer-wise Transfer Analysis")
    print(f"Total parameter groups: {total_params}")
    
    for freeze_depth in layer_configs:
        print(f"\n  Testing freeze depth: {freeze_depth}/{total_params}")
        
        # Create fresh model
        model = LayerwiseModel(base_model)
        model = model.to(device)
        model.freeze_until_layer(freeze_depth)
        
        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        # Fine-tune on limited target data
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                                     model.parameters()), lr=1e-4)
        
        # Train for 3 epochs on limited data
        limited_loader = get_limited_loader(train_loader, n_samples)
        
        for epoch in range(3):
            train_epoch(model, limited_loader, optimizer, device)
        
        # Evaluate
        eval_result = evaluate_detailed(model, target_loader, device)
        
        results.append({
            'frozen_layers': freeze_depth,
            'frozen_percentage': freeze_depth / total_params * 100,
            'trainable_params': trainable,
            'total_params': total,
            'auc': eval_result['auc'],
            'accuracy': eval_result['acc']
        })
        
        print(f"    AUC: {eval_result['auc']:.3f}, Acc: {eval_result['acc']:.3f}")
    
    return pd.DataFrame(results)


def get_limited_loader(loader, n_samples):
    """Create a dataloader with limited samples"""
    from torch.utils.data import Subset
    
    dataset = loader.dataset
    indices = np.random.choice(len(dataset), 
                              min(n_samples, len(dataset)), 
                              replace=False)
    subset = Subset(dataset, indices)
    
    return DataLoader(subset, batch_size=loader.batch_size, 
                     shuffle=True, num_workers=0)


# ===============================
# 6. CLINICAL IMPLICATIONS
# ===============================

def analyze_false_negatives(results_dict, save_path="false_negatives.png"):
    """
    Analyze false negatives (missed fractures) across anatomies
    Critical for clinical deployment
    """
    fn_results = []
    
    for anatomy, result in results_dict.items():
        preds = result['preds']
        targets = result['targets']
        confidences = result['confidences']
        
        # Get false negatives
        fn_mask = (targets == 1) & (preds < 0.5)
        fn_count = np.sum(fn_mask)
        fn_rate = fn_count / np.sum(targets == 1) * 100
        
        # High-confidence false negatives (most dangerous)
        high_conf_fn = np.sum((targets == 1) & (preds < 0.3))
        
        fn_results.append({
            'anatomy': anatomy,
            'total_fractures': int(np.sum(targets == 1)),
            'false_negatives': int(fn_count),
            'fn_rate': fn_rate,
            'high_conf_fn': int(high_conf_fn),
            'avg_confidence': np.mean(confidences[fn_mask]) if fn_count > 0 else 0
        })
    
    df = pd.DataFrame(fn_results)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # False negative rates
    ax = axes[0]
    bars = ax.bar(df['anatomy'], df['fn_rate'], color='crimson', alpha=0.7)
    ax.set_ylabel("False Negative Rate (%)")
    ax.set_title("Missed Fracture Rate by Anatomy")
    ax.axhline(y=df[df['anatomy'] == 'XR_WRIST']['fn_rate'].values[0], 
              color='green', linestyle='--', label='Source (WRIST)')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.legend()
    
    # Absolute counts
    ax = axes[1]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['false_negatives'], width, 
          label='All FN', color='orange', alpha=0.7)
    ax.bar(x + width/2, df['high_conf_fn'], width, 
          label='High-Conf FN', color='red', alpha=0.7)
    ax.set_ylabel("Count")
    ax.set_title("False Negative Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(df['anatomy'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] False negative analysis: {save_path}")
    return df


def sample_efficiency_curve(base_model, train_loader, target_loader, 
                            device, target_anatomy):
    """
    How many target samples needed to recover performance?
    Critical for deployment planning
    """
    sample_sizes = [10, 25, 50, 100, 200, 500]
    results = []
    
    print(f"\n[INFO] Sample Efficiency for {target_anatomy}")
    
    for n_samples in sample_sizes:
        if n_samples > len(train_loader.dataset):
            break
            
        print(f"  Testing with {n_samples} samples...")
        
        # Create model
        model = get_densenet121().to(device)
        model.load_state_dict(base_model.state_dict())
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Fine-tune on limited data
        limited_loader = get_limited_loader(train_loader, n_samples)
        
        for epoch in range(5):
            train_epoch(model, limited_loader, optimizer, device)
        
        # Evaluate
        eval_result = evaluate_detailed(model, target_loader, device)
        
        results.append({
            'n_samples': n_samples,
            'auc': eval_result['auc'],
            'accuracy': eval_result['acc']
        })
        
        print(f"    AUC: {eval_result['auc']:.3f}")
    
    return pd.DataFrame(results)


# ===============================
# 7. MAIN EXECUTION
# ===============================

def main():
    # Check GPU
    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
    
    # Paths
    BASE_PATH = r"C:\Users\Suraj\Documents\python\MURA-v1.1"
    TRAIN_ROOT = os.path.join(BASE_PATH, "train")
    VALID_ROOT = os.path.join(BASE_PATH, "valid")
    
    # Source anatomy
    SOURCE_ANATOMY = "XR_WRIST"
    
    # Target anatomies for transfer
    TARGET_ANATOMIES = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER", "XR_FINGER"]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Saving results to: {output_dir}")
    
    # ===========================
    # PHASE 1: TRAIN ON SOURCE
    # ===========================
    print("\n" + "="*50)
    print("PHASE 1: TRAINING ON SOURCE ANATOMY")
    print("="*50)
    
    train_set = MURADataset(TRAIN_ROOT, SOURCE_ANATOMY)
    val_set = MURADataset(VALID_ROOT, SOURCE_ANATOMY)
    
    train_loader = DataLoader(
        train_set, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    model = get_densenet121().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training
    EPOCHS = 10
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, device)
        eval_result = evaluate_detailed(model, val_loader, device)
        print(f"Epoch {epoch+1:02d} | Loss {loss:.4f} | {SOURCE_ANATOMY} AUC {eval_result['auc']:.4f}")
    
    # Save trained model
    model_path = os.path.join(output_dir, "source_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n[SAVED] Source model: {model_path}")
    
    # ===========================
    # PHASE 2: CROSS-ANATOMY EVALUATION
    # ===========================
    print("\n" + "="*50)
    print("PHASE 2: ZERO-SHOT CROSS-ANATOMY TRANSFER")
    print("="*50)
    
    # Evaluate source
    source_result = evaluate_detailed(model, val_loader, device)
    
    # Store all results
    transfer_results = {SOURCE_ANATOMY: source_result}
    cross_loaders = {SOURCE_ANATOMY: val_loader}
    
    # Evaluate each target anatomy
    for target_anat in TARGET_ANATOMIES:
        print(f"\nEvaluating on {target_anat}...")
        try:
            target_set = MURADataset(VALID_ROOT, target_anat)
            target_loader = DataLoader(
                target_set, batch_size=16, shuffle=False,
                num_workers=4, pin_memory=True
            )
            
            result = evaluate_detailed(model, target_loader, device)
            transfer_results[target_anat] = result
            cross_loaders[target_anat] = target_loader
            
            print(f"  {target_anat} AUC: {result['auc']:.4f} | Acc: {result['acc']:.4f}")
            
        except FileNotFoundError:
            print(f"  [WARNING] {target_anat} not found, skipping...")
            continue
    
    # Create results summary
    summary_data = []
    for anat, result in transfer_results.items():
        summary_data.append({
            'Anatomy': anat,
            'AUC': result['auc'],
            'Accuracy': result['acc'],
            'Transfer Type': 'Source' if anat == SOURCE_ANATOMY else 'Target'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "transfer_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVED] Transfer results: {summary_path}")
    print("\n" + summary_df.to_string(index=False))
    
    # ===========================
    # PHASE 3: FEATURE SPACE ANALYSIS
    # ===========================
    print("\n" + "="*50)
    print("PHASE 3: FEATURE SPACE ANALYSIS")
    print("="*50)
    
    feature_model = FeatureExtractor(model).to(device).eval()
    
    features_dict = {}
    for anat_name, loader in cross_loaders.items():
        feats, frac_labels = extract_features(feature_model, loader, device, anat_name)
        features_dict[anat_name] = {
            'features': feats,
            'fracture_labels': frac_labels
        }
    
    # Compute distances
    distance_df = compute_feature_distances(features_dict)
    distance_path = os.path.join(output_dir, "feature_distances.csv")
    distance_df.to_csv(distance_path, index=False)
    print(f"\n[SAVED] Feature distances: {distance_path}")
    print("\n" + distance_df.to_string(index=False))
    
    # Plot feature space
    plot_path = os.path.join(output_dir, "feature_space.png")
    plot_feature_space(features_dict, plot_path)
    
    # ===========================
    # PHASE 4: LAYER-WISE TRANSFER
    # ===========================
    print("\n" + "="*50)
    print("PHASE 4: LAYER-WISE TRANSFER ANALYSIS")
    print("="*50)
    
    # Run on first target anatomy
    first_target = list(cross_loaders.keys())[1]  # Skip source
    target_loader = cross_loaders[first_target]
    
    # Get training data for target (for fine-tuning)
    try:
        target_train = MURADataset(TRAIN_ROOT, first_target)
        target_train_loader = DataLoader(
            target_train, batch_size=16, shuffle=True,
            num_workers=0, pin_memory=True
        )
        
        layerwise_df = layerwise_transfer_experiment(
            model, target_train_loader, target_loader, device, n_samples=100
        )
        
        layerwise_path = os.path.join(output_dir, "layerwise_transfer.csv")
        layerwise_df.to_csv(layerwise_path, index=False)
        print(f"\n[SAVED] Layerwise results: {layerwise_path}")
        
    except FileNotFoundError:
        print(f"[WARNING] No training data for {first_target}, skipping layerwise analysis")
    
    # ===========================
    # PHASE 5: CLINICAL IMPLICATIONS
    # ===========================
    print("\n" + "="*50)
    print("PHASE 5: CLINICAL IMPACT ANALYSIS")
    print("="*50)
    
    # False negative analysis
    fn_path = os.path.join(output_dir, "false_negatives.png")
    fn_df = analyze_false_negatives(transfer_results, fn_path)
    fn_csv_path = os.path.join(output_dir, "false_negative_analysis.csv")
    fn_df.to_csv(fn_csv_path, index=False)
    print(f"\n[SAVED] FN analysis: {fn_csv_path}")
    print("\n" + fn_df.to_string(index=False))
    
    # Sample efficiency curve
    print("\n[INFO] Running sample efficiency analysis...")
    try:
        efficiency_df = sample_efficiency_curve(
            model, target_train_loader, target_loader, device, first_target
        )
        
        efficiency_path = os.path.join(output_dir, "sample_efficiency.csv")
        efficiency_df.to_csv(efficiency_path, index=False)
        
        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(efficiency_df['n_samples'], efficiency_df['auc'], 
                marker='o', linewidth=2, markersize=8)
        plt.axhline(y=source_result['auc'], color='green', 
                   linestyle='--', label='Source Performance')
        plt.xlabel("Number of Target Samples")
        plt.ylabel("AUC")
        plt.title(f"Sample Efficiency: {first_target}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        eff_plot_path = os.path.join(output_dir, "sample_efficiency.png")
        plt.savefig(eff_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVED] Sample efficiency: {efficiency_path}")
        print("\n" + efficiency_df.to_string(index=False))
        
    except Exception as e:
        print(f"[WARNING] Sample efficiency failed: {e}")
    
    # ===========================
    # FINAL SUMMARY
    # ===========================
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
    print(f"\nAll results saved to: {output_dir}")
    print("\nKey Files:")
    print(f"  - transfer_results.csv: Cross-anatomy performance")
    print(f"  - feature_distances.csv: Inter/intra anatomy distances")
    print(f"  - feature_space.png: PCA visualization")
    print(f"  - false_negative_analysis.csv: Clinical impact metrics")
    print(f"  - sample_efficiency.csv: Fine-tuning data requirements")
    
    # Generate executive summary
    print("\n" + "="*50)
    print("EXECUTIVE SUMMARY")
    print("="*50)
    
    source_auc = transfer_results[SOURCE_ANATOMY]['auc']
    target_aucs = [r['auc'] for a, r in transfer_results.items() if a != SOURCE_ANATOMY]
    avg_target_auc = np.mean(target_aucs)
    
    print(f"\nSource ({SOURCE_ANATOMY}) AUC: {source_auc:.3f}")
    print(f"Average Target AUC: {avg_target_auc:.3f}")
    print(f"Performance Drop: {(source_auc - avg_target_auc):.3f} ({(source_auc - avg_target_auc)/source_auc*100:.1f}%)")
    
    avg_fn_increase = fn_df[fn_df['anatomy'] != SOURCE_ANATOMY]['fn_rate'].mean() - \
                      fn_df[fn_df['anatomy'] == SOURCE_ANATOMY]['fn_rate'].values[0]
    print(f"\nAverage False Negative Rate Increase: {avg_fn_increase:.1f}%")
    
    print("\n" + "="*50)


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
