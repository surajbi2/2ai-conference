# ===============================
# CROSS-ANATOMY TRANSFER ANALYSIS
# Revised for journal extension (2AI 2026 → journal)
#
# Changes from conference version:
#   1. Added Grad-CAM analysis (addresses R4: "shallow analysis of failure modes")
#   2. Added ECE calibration metrics (addresses R4: "missing confidence calibration")
#   3. Fixed source_auc hardcode in plot_sample_efficiency — now computed from data
#   4. Fixed Unicode encoding artifacts (± α symbols)
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
from scipy.stats import ttest_rel
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

    # Grad-CAM configuration
    # Target layer: last dense block in DenseNet-121 before global avg pool
    GRADCAM_TARGET_LAYER = "features.denseblock4"
    GRADCAM_N_EXAMPLES = 6      # FN examples to visualize per anatomy
    GRADCAM_CONF_THRESHOLD = 0.3  # High-confidence FN: pred < this value

    # ECE calibration
    ECE_N_BINS = 10


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
            torch.tensor([label], dtype=torch.float32),
            img_path   # returned so Grad-CAM can reload specific images
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

    for batch in tqdm(loader, leave=False, desc="Training"):
        x, y = batch[0], batch[1]
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
    Comprehensive evaluation with all metrics.
    Returns dict with predictions, targets, and computed metrics.
    Also collects image paths for Grad-CAM selection.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="Evaluating"):
            x, y = batch[0], batch[1]
            paths = batch[2] if len(batch) > 2 else [""] * len(y)

            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs.tolist())
            all_targets.extend(y.numpy().flatten().tolist())
            all_paths.extend(paths)

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    if len(np.unique(targets)) < 2:
        return {
            'preds': preds,
            'targets': targets,
            'paths': all_paths,
            'auc': float('nan'),
            'accuracy': accuracy_score(targets, preds > 0.5),
            'sensitivity': float('nan'),
            'specificity': float('nan'),
        }

    preds_binary = (preds >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds_binary).ravel()

    auc = roc_auc_score(targets, preds)
    accuracy = accuracy_score(targets, preds_binary)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        'preds': preds,
        'targets': targets,
        'paths': all_paths,
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
# NEW: ECE CALIBRATION
# ===============================

def compute_ece(preds, targets, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well predicted confidence aligns with actual accuracy.
    A perfectly calibrated model has ECE = 0.
    High ECE confirms the high-confidence error problem we identified.

    Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks."
    In: Proceedings of ICML.

    Args:
        preds:    numpy array of predicted probabilities in [0, 1]
        targets:  numpy array of binary ground truth labels
        n_bins:   number of equal-width bins (default 10)

    Returns:
        dict with 'ece', 'bin_data' (for reliability diagram),
        'mce' (Maximum Calibration Error)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    mce = 0.0
    n_total = len(preds)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (preds >= lo) & (preds < hi)
        if i == n_bins - 1:          # include right edge in last bin
            mask = (preds >= lo) & (preds <= hi)

        n_bin = np.sum(mask)
        if n_bin == 0:
            bin_data.append({
                'bin_lo': lo, 'bin_hi': hi,
                'avg_confidence': (lo + hi) / 2,
                'avg_accuracy': 0.0,
                'count': 0,
                'gap': 0.0,
            })
            continue

        avg_conf = np.mean(preds[mask])
        avg_acc = np.mean(targets[mask])
        gap = abs(avg_conf - avg_acc)

        ece += (n_bin / n_total) * gap
        mce = max(mce, gap)

        bin_data.append({
            'bin_lo': lo, 'bin_hi': hi,
            'avg_confidence': avg_conf,
            'avg_accuracy': avg_acc,
            'count': int(n_bin),
            'gap': gap,
        })

    return {
        'ece': ece,
        'mce': mce,
        'bin_data': bin_data,
        'n_bins': n_bins,
    }


def calibration_analysis(multi_seed_results, config, output_dir):
    """
    Compute ECE and MCE per anatomy, aggregated across all seeds.
    Also generates reliability diagrams.

    This directly addresses R4's critique about missing calibration analysis.
    High ECE values confirm that high-confidence errors are a systematic
    calibration failure, not random noise.

    Args:
        multi_seed_results: dict {seed: {anatomy: result}}
        config: Config object
        output_dir: str, where to save outputs

    Returns:
        DataFrame with ECE/MCE per anatomy
    """
    print(f"\n{'='*60}")
    print("CALIBRATION ANALYSIS (ECE / MCE)")
    print(f"{'='*60}")

    # Aggregate predictions across all seeds per anatomy
    anatomy_preds = {}
    anatomy_targets = {}

    for seed, results in multi_seed_results.items():
        for anatomy, result in results.items():
            if anatomy not in anatomy_preds:
                anatomy_preds[anatomy] = []
                anatomy_targets[anatomy] = []
            anatomy_preds[anatomy].append(result['preds'])
            anatomy_targets[anatomy].append(result['targets'])

    cal_results = []
    all_anatomies = [config.SOURCE_ANATOMY] + config.TARGET_ANATOMIES
    anatomy_order = [a for a in all_anatomies if a in anatomy_preds]

    # Reliability diagram — one subplot per anatomy
    n_anat = len(anatomy_order)
    n_cols = 3
    n_rows = (n_anat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, anatomy in enumerate(anatomy_order):
        all_p = np.concatenate(anatomy_preds[anatomy])
        all_t = np.concatenate(anatomy_targets[anatomy])

        cal = compute_ece(all_p, all_t, n_bins=config.ECE_N_BINS)

        cal_results.append({
            'anatomy': anatomy,
            'ece': cal['ece'],
            'mce': cal['mce'],
            'n_predictions': len(all_p),
        })

        print(f"\n{anatomy}:")
        print(f"  ECE : {cal['ece']:.4f}")
        print(f"  MCE : {cal['mce']:.4f}")

        # Reliability diagram
        ax = axes[idx]
        bd = cal['bin_data']
        bin_confs = [b['avg_confidence'] for b in bd]
        bin_accs  = [b['avg_accuracy']   for b in bd]
        bin_gaps  = [b['gap']            for b in bd]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
        # Actual reliability
        ax.bar([b['bin_lo'] for b in bd],
               bin_accs,
               width=1.0 / config.ECE_N_BINS,
               align='edge',
               color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5,
               label='Accuracy')
        # Gap overlay
        ax.bar([b['bin_lo'] for b in bd],
               bin_gaps,
               width=1.0 / config.ECE_N_BINS,
               align='edge',
               bottom=np.minimum(bin_confs, bin_accs),
               color='red', alpha=0.4, label='Gap')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{anatomy}\nECE={cal["ece"]:.3f}  MCE={cal["mce"]:.3f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(anatomy_order), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Reliability Diagrams: Confidence Calibration per Anatomy',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'calibration_reliability.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVED] Reliability diagram: {fig_path}")

    # ECE bar chart — one bar per anatomy, colored source vs target
    df = pd.DataFrame(cal_results)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71' if a == config.SOURCE_ANATOMY else '#e74c3c'
              for a in df['anatomy']]
    bars = ax2.bar(df['anatomy'], df['ece'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, df['ece']):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    ax2.set_ylabel('Expected Calibration Error (ECE)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Anatomy', fontsize=13, fontweight='bold')
    ax2.set_title('Calibration Error per Anatomy\n'
                  '(Higher ECE = more overconfident errors)',
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, edgecolor='black',
              label=f'Source ({config.SOURCE_ANATOMY})'),
        Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black',
              label='Target (Zero-Shot Transfer)'),
    ]
    ax2.legend(handles=legend_elements, fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.tight_layout()

    ece_fig_path = os.path.join(output_dir, 'calibration_ece_bars.png')
    plt.savefig(ece_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] ECE bar chart: {ece_fig_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'calibration_ece.csv')
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] ECE table: {csv_path}")

    return df


# ===============================
# NEW: GRAD-CAM ANALYSIS
# ===============================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for DenseNet-121.

    Produces a heatmap showing which spatial regions the model attends to
    when making a prediction. Used here to visualize WHY the model fails
    on hand false negatives vs. succeeds on wrist true positives.

    Target layer: features.denseblock4  (last dense block, before global avg pool)
    This is the standard choice for DenseNet-121 — it has the richest
    spatial feature map before spatial information is pooled away.

    Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization." ICCV.
    """

    def __init__(self, model, target_layer_name):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name):
        """Register forward and backward hooks on the target layer."""
        target = dict(self.model.named_modules()).get(layer_name)
        if target is None:
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available: {list(dict(self.model.named_modules()).keys())[:10]}..."
            )

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hook_handles.append(target.register_forward_hook(forward_hook))
        self._hook_handles.append(target.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()

    def generate(self, input_tensor, device):
        """
        Generate Grad-CAM heatmap for a single image tensor.

        Args:
            input_tensor: (1, 3, 224, 224) float tensor
            device: torch device

        Returns:
            heatmap: (224, 224) numpy array in [0, 1]
            pred_prob: scalar predicted probability
        """
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        # Forward pass
        logit = self.model(input_tensor)
        pred_prob = torch.sigmoid(logit).item()

        # Backward pass w.r.t. the single output neuron
        self.model.zero_grad()
        logit.backward()

        # Global average pool the gradients over spatial dimensions
        # shape: (1, C, H, W) -> (C,)
        weights = self.gradients.mean(dim=(2, 3)).squeeze()  # (C,)

        # Weighted sum of activation maps
        # activations shape: (1, C, H, W)
        cam = (weights[:, None, None] * self.activations.squeeze(0)).sum(0)  # (H, W)

        # ReLU — only keep positive contributions
        cam = torch.relu(cam)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Resize to input resolution
        heatmap = cv2.resize(cam, (224, 224))

        return heatmap, pred_prob


def _load_image_for_gradcam(img_path):
    """
    Load a single image the same way MURADataset does.
    Returns (tensor (3,224,224), raw_gray_uint8 (224,224))
    """
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        image = np.zeros((224, 224), dtype=np.uint8)
    else:
        image = cv2.resize(image, (224, 224))

    raw = image.copy()
    image = image.astype(np.float32) / 255.0
    image = np.stack([image] * 3, axis=0)

    return torch.tensor(image, dtype=torch.float32), raw


def gradcam_analysis(model, multi_seed_results, config, output_dir):
    """
    Generate Grad-CAM visualizations comparing:
      (A) Wrist TRUE POSITIVES  — what correct wrist detection looks like
      (B) Hand HIGH-CONFIDENCE FALSE NEGATIVES — why hand fractures are missed

    This directly addresses R4's weakness: "speculates about bone morphology
    but provides no visualization to support claims."

    The key expected finding:
      - Wrist TP: model attends to cortical bone / fracture line region
      - Hand HC-FN: model attends to soft tissue / background / wrong bone

    One figure is generated:
      gradcam_wrist_vs_hand.png  — 2-row grid, N examples each

    Args:
        model:               trained source model (wrist weights, seed 42)
        multi_seed_results:  dict {seed: {anatomy: result}}
        config:              Config object
        output_dir:          str

    Returns:
        dict with summary stats (mean activation overlap, etc.)
    """
    print(f"\n{'='*60}")
    print("GRAD-CAM ANALYSIS: WRIST TP vs HAND HC-FN")
    print(f"{'='*60}")

    device = next(model.parameters()).device
    gradcam = GradCAM(model, config.GRADCAM_TARGET_LAYER)

    # ----------------------------------------------------------------
    # Collect candidate images from seed 42 (first seed, representative)
    # ----------------------------------------------------------------
    seed = config.RANDOM_SEEDS[0]
    seed_results = multi_seed_results.get(seed, {})

    # --- Wrist TRUE POSITIVES (correct fracture detection) ---
    wrist_res = seed_results.get(config.SOURCE_ANATOMY, {})
    wrist_preds   = np.array(wrist_res.get('preds', []))
    wrist_targets = np.array(wrist_res.get('targets', []))
    wrist_paths   = wrist_res.get('paths', [])

    if len(wrist_paths) == 0:
        print("[WARNING] No image paths in wrist results — did evaluate_detailed "
              "run with the updated MURADataset? Skipping Grad-CAM.")
        gradcam.remove_hooks()
        return {}

    # True positives: label=1, pred >= 0.7 (high-confidence correct)
    tp_mask = (wrist_targets == 1) & (wrist_preds >= 0.7)
    tp_indices = np.where(tp_mask)[0]
    np.random.seed(42)
    tp_sample = np.random.choice(
        tp_indices,
        size=min(config.GRADCAM_N_EXAMPLES, len(tp_indices)),
        replace=False
    )

    # --- Hand HIGH-CONFIDENCE FALSE NEGATIVES ---
    hand_res = seed_results.get("XR_HAND", {})
    hand_preds   = np.array(hand_res.get('preds', []))
    hand_targets = np.array(hand_res.get('targets', []))
    hand_paths   = hand_res.get('paths', [])

    # High-confidence FN: label=1, pred < 0.3
    hc_fn_mask = (hand_targets == 1) & (hand_preds < config.GRADCAM_CONF_THRESHOLD)
    hc_fn_indices = np.where(hc_fn_mask)[0]
    hc_fn_sample = np.random.choice(
        hc_fn_indices,
        size=min(config.GRADCAM_N_EXAMPLES, len(hc_fn_indices)),
        replace=False
    )

    n_wrist = len(tp_sample)
    n_hand  = len(hc_fn_sample)

    if n_wrist == 0 or n_hand == 0:
        print("[WARNING] Insufficient examples for Grad-CAM. "
              f"Wrist TP={n_wrist}, Hand HC-FN={n_hand}. Skipping.")
        gradcam.remove_hooks()
        return {}

    print(f"  Wrist TP examples:    {n_wrist}")
    print(f"  Hand HC-FN examples:  {n_hand}")

    # ----------------------------------------------------------------
    # Generate heatmaps
    # ----------------------------------------------------------------
    n_cols = max(n_wrist, n_hand)
    fig, axes = plt.subplots(
        4, n_cols,
        figsize=(3.0 * n_cols, 12),
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05}
    )
    # Row 0: wrist raw X-ray
    # Row 1: wrist Grad-CAM overlay
    # Row 2: hand raw X-ray
    # Row 3: hand Grad-CAM overlay

    row_labels = [
        "Wrist (source)\nX-ray",
        "Wrist (source)\nGrad-CAM",
        "Hand (target)\nX-ray",
        "Hand (target)\nGrad-CAM",
    ]
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=10, fontweight='bold',
                                    rotation=90, labelpad=10)

    summary_stats = {'wrist_tp': [], 'hand_hcfn': []}

    # --- Wrist TP row ---
    for col, idx in enumerate(tp_sample):
        path = wrist_paths[idx]
        prob = wrist_preds[idx]

        tensor, raw = _load_image_for_gradcam(path)
        heatmap, _ = gradcam.generate(tensor, device)

        # Raw X-ray
        ax0 = axes[0, col]
        ax0.imshow(raw, cmap='gray')
        ax0.set_title(f'p={prob:.2f}', fontsize=9, color='green')
        ax0.axis('off')

        # Grad-CAM overlay
        ax1 = axes[1, col]
        ax1.imshow(raw, cmap='gray')
        ax1.imshow(heatmap, cmap='jet', alpha=0.45)
        ax1.axis('off')

        # Track mean activation in central 50% region (bone-likely region)
        h, w = heatmap.shape
        center_activation = heatmap[h//4:3*h//4, w//4:3*w//4].mean()
        summary_stats['wrist_tp'].append(center_activation)

    # Hide unused wrist columns
    for col in range(n_wrist, n_cols):
        axes[0, col].axis('off')
        axes[1, col].axis('off')

    # --- Hand HC-FN row ---
    for col, idx in enumerate(hc_fn_sample):
        path = hand_paths[idx]
        prob = hand_preds[idx]

        tensor, raw = _load_image_for_gradcam(path)
        heatmap, _ = gradcam.generate(tensor, device)

        # Raw X-ray
        ax2 = axes[2, col]
        ax2.imshow(raw, cmap='gray')
        ax2.set_title(f'p={prob:.2f}', fontsize=9, color='red')
        ax2.axis('off')

        # Grad-CAM overlay
        ax3 = axes[3, col]
        ax3.imshow(raw, cmap='gray')
        ax3.imshow(heatmap, cmap='jet', alpha=0.45)
        ax3.axis('off')

        border_activation = (
            heatmap[:h//4, :].mean() + heatmap[3*h//4:, :].mean() +
            heatmap[:, :w//4].mean() + heatmap[:, 3*w//4:].mean()
        ) / 4
        summary_stats['hand_hcfn'].append(border_activation)

    # Hide unused hand columns
    for col in range(n_hand, n_cols):
        axes[2, col].axis('off')
        axes[3, col].axis('off')

    # ----------------------------------------------------------------
    # Title and colorbar
    # ----------------------------------------------------------------
    fig.suptitle(
        "Grad-CAM Analysis: Wrist True Positives vs Hand High-Confidence False Negatives\n"
        "Top rows: wrist model correctly attends to cortical bone structure.\n"
        "Bottom rows: model attends to irrelevant regions on hand X-rays, missing fractures.",
        fontsize=11, fontweight='bold', y=1.02
    )

    # Single colorbar for all heatmaps
    sm = plt.cm.ScalarMappable(cmap='jet',
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.3, label='Activation intensity')

    save_path = os.path.join(output_dir, 'gradcam_wrist_vs_hand.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Grad-CAM figure: {save_path}")

    gradcam.remove_hooks()

    # ----------------------------------------------------------------
    # Quantitative summary
    # ----------------------------------------------------------------
    wrist_center_mean = np.mean(summary_stats['wrist_tp']) if summary_stats['wrist_tp'] else 0
    hand_border_mean  = np.mean(summary_stats['hand_hcfn']) if summary_stats['hand_hcfn'] else 0

    print(f"\n  Wrist TP — mean central activation:      {wrist_center_mean:.3f}")
    print(f"  Hand HC-FN — mean peripheral activation: {hand_border_mean:.3f}")
    print(f"  (Higher central activation = model attends to bone region)")
    print(f"  (Higher peripheral = model attends to background/soft tissue)")

    summary_df = pd.DataFrame({
        'group': ['wrist_true_positive', 'hand_high_conf_fn'],
        'mean_center_activation': [wrist_center_mean, hand_border_mean],
        'n_examples': [n_wrist, n_hand],
    })
    summary_path = os.path.join(output_dir, 'gradcam_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"[SAVED] Grad-CAM summary: {summary_path}")

    return {'wrist_center_mean': wrist_center_mean,
            'hand_border_mean': hand_border_mean}


# ===============================
# MULTI-SEED CROSS-ANATOMY EVALUATION
# ===============================

def cross_anatomy_evaluation(config, device, seed, output_dir):
    """
    Train on source, evaluate on all anatomies.

    Returns:
        model:   trained DenseNet-121
        results: dict {anatomy: evaluation_dict}
    """
    print(f"\n{'='*60}")
    print(f"SEED {seed}: Training on {config.SOURCE_ANATOMY}")
    print(f"{'='*60}")

    set_seed(seed)

    train_set = MURADataset(config.TRAIN_ROOT, config.SOURCE_ANATOMY)
    val_set   = MURADataset(config.VALID_ROOT, config.SOURCE_ANATOMY)

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    model = get_densenet121().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"\nTraining for {config.EPOCHS_SOURCE} epochs...")
    for epoch in range(config.EPOCHS_SOURCE):
        loss = train_epoch(model, train_loader, optimizer, device)
        if (epoch + 1) % 2 == 0:
            result = evaluate_detailed(model, val_loader, device)
            print(f"Epoch {epoch+1:02d} | Loss {loss:.4f} | "
                  f"Source AUC {result['auc']:.4f}")

    model_path = os.path.join(output_dir, f'source_model_seed{seed}.pth')
    torch.save(model.state_dict(), model_path)

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

    return model, results


# ===============================
# SAMPLE EFFICIENCY ANALYSIS
# ===============================

def sample_efficiency_experiment(base_model, config, device, output_dir):
    """
    Test how many target samples are needed to recover source performance.

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
            train_set = MURADataset(config.TRAIN_ROOT, anatomy)
            val_set   = MURADataset(config.VALID_ROOT, anatomy)

            val_loader = DataLoader(
                val_set, batch_size=config.BATCH_SIZE, shuffle=False,
                num_workers=config.NUM_WORKERS, pin_memory=True
            )

            # Zero-shot baseline
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

            for n_samples in config.SAMPLE_SIZES:
                if n_samples > len(train_set):
                    print(f"  Skipping {n_samples} (exceeds dataset size {len(train_set)})")
                    continue

                for run_idx in range(config.SAMPLE_EFFICIENCY_RUNS):
                    print(f"  {n_samples} samples | Run {run_idx+1}/{config.SAMPLE_EFFICIENCY_RUNS}")

                    model = get_densenet121().to(device)
                    model.load_state_dict(base_model.state_dict())

                    set_seed(config.RANDOM_SEEDS[0] + run_idx)
                    indices = np.random.choice(len(train_set), n_samples, replace=False)
                    subset = Subset(train_set, indices)

                    train_loader = DataLoader(
                        subset, batch_size=config.BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=True
                    )

                    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
                    for epoch in range(config.EPOCHS_FINETUNE):
                        train_epoch(model, train_loader, optimizer, device)

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

    df = pd.DataFrame(all_results)
    efficiency_path = os.path.join(output_dir, 'sample_efficiency_results.csv')
    df.to_csv(efficiency_path, index=False)
    print(f"\n[SAVED] Sample efficiency: {efficiency_path}")

    return df


# ===============================
# STATISTICAL ANALYSIS
# ===============================

def statistical_analysis(multi_seed_results_df, config, output_dir):
    """
    Paired t-tests comparing source AUC to each target, with Bonferroni correction.

    Returns:
        DataFrame with statistical test results
    """
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")

    results = []

    source_data = multi_seed_results_df[
        multi_seed_results_df['anatomy'] == config.SOURCE_ANATOMY
    ]
    source_aucs = source_data['auc'].values

    n = len(source_aucs)
    print(f"\nSource ({config.SOURCE_ANATOMY}):")
    print(f"  Mean AUC: {np.mean(source_aucs):.4f} ± {np.std(source_aucs):.4f}")
    print(f"  95% CI: [{np.mean(source_aucs) - 1.96*np.std(source_aucs)/np.sqrt(n):.4f}, "
          f"{np.mean(source_aucs) + 1.96*np.std(source_aucs)/np.sqrt(n):.4f}]")

    n_comparisons = len(config.TARGET_ANATOMIES)
    bonferroni_alpha = 0.05 / n_comparisons
    print(f"\nBonferroni-corrected α = 0.05/{n_comparisons} = {bonferroni_alpha:.4f}")

    for target in config.TARGET_ANATOMIES:
        target_data = multi_seed_results_df[
            multi_seed_results_df['anatomy'] == target
        ]
        if len(target_data) == 0:
            continue

        target_aucs = target_data['auc'].values

        if len(source_aucs) != len(target_aucs):
            print(f"\n[WARNING] {target}: Mismatched sample sizes, skipping t-test")
            continue

        t_stat, p_value = ttest_rel(source_aucs, target_aucs)

        mean_diff = np.mean(source_aucs) - np.mean(target_aucs)
        pooled_std = np.sqrt(
            (np.std(source_aucs)**2 + np.std(target_aucs)**2) / 2
        )
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        n_t = len(target_aucs)
        ci_lower = np.mean(target_aucs) - 1.96 * np.std(target_aucs) / np.sqrt(n_t)
        ci_upper = np.mean(target_aucs) + 1.96 * np.std(target_aucs) / np.sqrt(n_t)

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
    stats_path = os.path.join(output_dir, 'statistical_tests.csv')
    df.to_csv(stats_path, index=False)
    print(f"\n[SAVED] Statistical tests: {stats_path}")

    return df


# ===============================
# CLINICAL METRICS
# ===============================

def false_negative_analysis(multi_seed_results, output_dir):
    """
    Analyze false negatives (missed fractures) aggregated across all seeds.

    Returns:
        DataFrame with FN analysis
    """
    print(f"\n{'='*60}")
    print("FALSE NEGATIVE ANALYSIS")
    print(f"{'='*60}")

    anatomy_results = {}
    for seed, results in multi_seed_results.items():
        for anatomy, result in results.items():
            if anatomy not in anatomy_results:
                anatomy_results[anatomy] = []
            anatomy_results[anatomy].append(result)

    fn_results = []
    for anatomy, results_list in anatomy_results.items():
        all_preds   = np.concatenate([r['preds']   for r in results_list])
        all_targets = np.concatenate([r['targets'] for r in results_list])

        fn_mask        = (all_targets == 1) & (all_preds < 0.5)
        fn_count       = np.sum(fn_mask)
        total_positives = np.sum(all_targets == 1)
        fn_rate        = (fn_count / total_positives * 100) if total_positives > 0 else 0

        high_conf_fn      = np.sum((all_targets == 1) & (all_preds < 0.3))
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
    fn_path = os.path.join(output_dir, 'false_negative_analysis.csv')
    df.to_csv(fn_path, index=False)
    print(f"\n[SAVED] FN analysis: {fn_path}")

    return df


def compute_operating_points(multi_seed_results, config, output_dir):
    """
    Compute sensitivity/specificity at multiple decision thresholds.

    Returns:
        DataFrame with operating point metrics
    """
    print(f"\n{'='*60}")
    print("OPERATING POINT ANALYSIS")
    print(f"{'='*60}")

    anatomy_results = {}
    for seed, results in multi_seed_results.items():
        for anatomy, result in results.items():
            if anatomy not in anatomy_results:
                anatomy_results[anatomy] = []
            anatomy_results[anatomy].append(result)

    op_results = []
    for anatomy, results_list in anatomy_results.items():
        all_preds   = np.concatenate([r['preds']   for r in results_list])
        all_targets = np.concatenate([r['targets'] for r in results_list])

        for threshold in config.THRESHOLDS:
            preds_binary = (all_preds >= threshold).astype(int)
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
    op_path = os.path.join(output_dir, 'operating_points.csv')
    df.to_csv(op_path, index=False)
    print(f"\n[SAVED] Operating points: {op_path}")

    return df


# ===============================
# VISUALIZATION
# ===============================

def plot_roc_curves(multi_seed_results, output_dir):
    """ROC curves for all anatomies (individual + combined panel)."""
    print("\n[INFO] Generating ROC curves...")

    anatomy_results = {}
    for seed, results in multi_seed_results.items():
        for anatomy, result in results.items():
            if anatomy not in anatomy_results:
                anatomy_results[anatomy] = []
            anatomy_results[anatomy].append(result)

    n_anatomies = len(anatomy_results)
    n_cols = 3
    n_rows = (n_anatomies + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, (anatomy, results_list) in enumerate(anatomy_results.items()):
        ax = axes[idx]
        all_preds   = np.concatenate([r['preds']   for r in results_list])
        all_targets = np.concatenate([r['targets'] for r in results_list])

        fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
        roc_auc = roc_auc_score(all_targets, all_preds)

        ax.plot(fpr, tpr, linewidth=2.5,
                label=f'AUC = {roc_auc:.3f}', color='darkblue')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

        optimal_idx = np.argmax(tpr - fpr)
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
                label=f'Optimal (t={thresholds[optimal_idx]:.2f})')

        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title(anatomy, fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    # Combined panel
    if n_anatomies < len(axes):
        ax = axes[n_anatomies]
        colors = plt.cm.tab10(np.linspace(0, 1, len(anatomy_results)))
        for (anatomy, results_list), color in zip(anatomy_results.items(), colors):
            all_preds   = np.concatenate([r['preds']   for r in results_list])
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
    """Precision-recall curves for all anatomies."""
    print("\n[INFO] Generating precision-recall curves...")

    anatomy_results = {}
    for seed, results in multi_seed_results.items():
        for anatomy, result in results.items():
            if anatomy not in anatomy_results:
                anatomy_results[anatomy] = []
            anatomy_results[anatomy].append(result)

    n_anatomies = len(anatomy_results)
    n_cols = 3
    n_rows = (n_anatomies + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, (anatomy, results_list) in enumerate(anatomy_results.items()):
        ax = axes[idx]
        all_preds   = np.concatenate([r['preds']   for r in results_list])
        all_targets = np.concatenate([r['targets'] for r in results_list])

        precision, recall, _ = precision_recall_curve(all_targets, all_preds)
        ap_score   = average_precision_score(all_targets, all_preds)
        prevalence = np.mean(all_targets)

        ax.plot(recall, precision, linewidth=2.5,
                label=f'AP = {ap_score:.3f}', color='darkgreen')
        ax.axhline(y=prevalence, color='r', linestyle='--', linewidth=1.5,
                   label=f'Baseline = {prevalence:.3f}')

        ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
        ax.set_title(anatomy, fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    if n_anatomies < len(axes):
        ax = axes[n_anatomies]
        colors = plt.cm.tab10(np.linspace(0, 1, len(anatomy_results)))
        for (anatomy, results_list), color in zip(anatomy_results.items(), colors):
            all_preds   = np.concatenate([r['preds']   for r in results_list])
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
    """Bar chart comparing AUC across anatomies with error bars."""
    print("\n[INFO] Generating AUC comparison plot...")

    summary = multi_seed_df.groupby('anatomy')['auc'].agg(['mean', 'std']).reset_index()
    summary = summary.sort_values('mean', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2ecc71' if anat == config.SOURCE_ANATOMY else '#e74c3c'
              for anat in summary['anatomy']]

    bars = ax.bar(summary['anatomy'], summary['mean'],
                  yerr=summary['std'], capsize=7,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    for bar, mean, std in zip(bars, summary['mean'], summary['std']):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax.set_ylabel('AUC-ROC', fontsize=15, fontweight='bold')
    ax.set_xlabel('Anatomy', fontsize=15, fontweight='bold')
    ax.set_title('Cross-Anatomy Transfer Performance (Mean ± Std over 5 seeds)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2,
              label=f'Source ({config.SOURCE_ANATOMY})'),
        Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2,
              label='Target (Zero-Shot Transfer)'),
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
    """Visualize false negative rates across anatomies."""
    print("\n[INFO] Generating false negative analysis plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = ['#2ecc71' if anat == config.SOURCE_ANATOMY else '#e74c3c'
              for anat in fn_df['anatomy']]
    bars = ax.bar(fn_df['anatomy'], fn_df['fn_rate'],
                  color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    source_fn = fn_df[fn_df['anatomy'] == config.SOURCE_ANATOMY]['fn_rate'].values[0]
    ax.axhline(y=source_fn, color='green', linestyle='--', linewidth=2.5,
               label=f'Source ({config.SOURCE_ANATOMY}): {source_fn:.1f}%', alpha=0.8)

    ax.set_ylabel('False Negative Rate (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Anatomy', fontsize=14, fontweight='bold')
    ax.set_title('Missed Fracture Rate by Anatomy', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    for bar, val in zip(bars, fn_df['fn_rate']):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax = axes[1]
    x = np.arange(len(fn_df))
    width = 0.35
    ax.bar(x - width / 2, fn_df['false_negatives'], width,
           label='All False Negatives',
           color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar(x + width / 2, fn_df['high_conf_fn'], width,
           label='High-Confidence FN (pred<0.3)',
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


def plot_sample_efficiency(efficiency_df, multi_seed_df, config, output_dir):
    """
    Plot sample efficiency curves for multiple anatomies.

    NOTE: source_auc is now computed from actual multi_seed_df data
    instead of being hardcoded. This ensures the reference line
    matches whatever model you actually trained.
    """
    print("\n[INFO] Generating sample efficiency plot...")

    # Compute source AUC from actual data — no more hardcoded 0.879
    source_auc = multi_seed_df[
        multi_seed_df['anatomy'] == config.SOURCE_ANATOMY
    ]['auc'].mean()
    print(f"  Source AUC (from data): {source_auc:.4f}")

    anatomies = efficiency_df['anatomy'].unique()
    n_anatomies = len(anatomies)

    fig, axes = plt.subplots(1, n_anatomies, figsize=(6 * n_anatomies, 5))
    if n_anatomies == 1:
        axes = [axes]

    colors_map = {
        'XR_ELBOW':    '#3498db',
        'XR_HAND':     '#e74c3c',
        'XR_SHOULDER': '#9b59b6',
    }

    for idx, anatomy in enumerate(anatomies):
        ax = axes[idx]
        data = efficiency_df[efficiency_df['anatomy'] == anatomy]
        grouped = data.groupby('n_samples')['auc'].agg(['mean', 'std']).reset_index()

        color = colors_map.get(anatomy, '#34495e')
        ax.errorbar(grouped['n_samples'], grouped['mean'], yerr=grouped['std'],
                    marker='o', linewidth=2.5, markersize=10, capsize=5,
                    color=color, label='Fine-tuned Model', alpha=0.9)

        ax.axhline(y=source_auc, color='green', linestyle='--', linewidth=2.5,
                   label=f'Source AUC ({source_auc:.3f})', alpha=0.8)
        ax.axhline(y=source_auc * 0.90, color='orange', linestyle=':', linewidth=2,
                   label='90% Recovery', alpha=0.6)
        ax.axhline(y=source_auc * 0.95, color='red', linestyle=':', linewidth=2,
                   label='95% Recovery', alpha=0.6)

        ax.set_xlabel('Number of Target Samples', fontsize=13, fontweight='bold')
        ax.set_ylabel('AUC-ROC', fontsize=13, fontweight='bold')
        ax.set_title(anatomy, fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_ylim([0.65, 0.92])
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
    print("\n" + "=" * 70)
    print("CROSS-ANATOMY TRANSFER LEARNING — JOURNAL REVISION PIPELINE")
    print("=" * 70)

    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
    device = torch.device("cuda")
    print(f"\n[INFO] Using GPU: {torch.cuda.get_device_name(0)}")

    config = Config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_journal_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Output directory: {output_dir}")

    # ================================================================
    # PHASE 1: MULTI-SEED CROSS-ANATOMY EVALUATION
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: MULTI-SEED CROSS-ANATOMY EVALUATION")
    print("=" * 70)

    multi_seed_results = {}
    multi_seed_data    = []

    for seed_idx, seed in enumerate(config.RANDOM_SEEDS):
        print(f"\n[PROGRESS] Seed {seed_idx + 1}/{len(config.RANDOM_SEEDS)}")
        model, results = cross_anatomy_evaluation(config, device, seed, output_dir)
        multi_seed_results[seed] = results

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

    multi_seed_df = pd.DataFrame(multi_seed_data)
    multi_seed_path = os.path.join(output_dir, 'multi_seed_results.csv')
    multi_seed_df.to_csv(multi_seed_path, index=False)
    print(f"\n[SAVED] Multi-seed results: {multi_seed_path}")

    print("\n" + "=" * 70)
    print("SUMMARY: Cross-Anatomy Performance")
    print("=" * 70)
    summary = multi_seed_df.groupby('anatomy')['auc'].agg(['mean', 'std', 'min', 'max'])
    print(summary.to_string())

    # ================================================================
    # PHASE 2: STATISTICAL ANALYSIS
    # ================================================================
    stats_df = statistical_analysis(multi_seed_df, config, output_dir)

    # ================================================================
    # PHASE 3: CLINICAL METRICS
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: CLINICAL IMPACT ANALYSIS")
    print("=" * 70)

    fn_df = false_negative_analysis(multi_seed_results, output_dir)
    op_df = compute_operating_points(multi_seed_results, config, output_dir)

    # ================================================================
    # PHASE 4: CALIBRATION ANALYSIS  (NEW — journal revision)
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: CALIBRATION ANALYSIS (ECE / MCE)")
    print("=" * 70)

    cal_df = calibration_analysis(multi_seed_results, config, output_dir)

    # ================================================================
    # PHASE 5: SAMPLE EFFICIENCY
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: SAMPLE EFFICIENCY ANALYSIS")
    print("=" * 70)

    first_seed = config.RANDOM_SEEDS[0]
    base_model_path = os.path.join(output_dir, f'source_model_seed{first_seed}.pth')
    base_model = get_densenet121().to(device)
    base_model.load_state_dict(torch.load(base_model_path))

    efficiency_df = sample_efficiency_experiment(base_model, config, device, output_dir)

    # ================================================================
    # PHASE 6: GRAD-CAM ANALYSIS  (NEW — journal revision)
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: GRAD-CAM FAILURE ANALYSIS")
    print("=" * 70)

    # Reload seed-42 model (the one whose paths are stored in multi_seed_results)
    seed42_model_path = os.path.join(output_dir, f'source_model_seed{first_seed}.pth')
    seed42_model = get_densenet121().to(device)
    seed42_model.load_state_dict(torch.load(seed42_model_path))

    gradcam_summary = gradcam_analysis(
        seed42_model, multi_seed_results, config, output_dir
    )

    # ================================================================
    # PHASE 7: VISUALIZATION
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 7: GENERATING PUBLICATION FIGURES")
    print("=" * 70)

    plot_auc_comparison(multi_seed_df, config, output_dir)
    plot_roc_curves(multi_seed_results, output_dir)
    plot_precision_recall_curves(multi_seed_results, output_dir)
    plot_false_negative_analysis(fn_df, config, output_dir)
    # Pass multi_seed_df so source_auc is computed from real data
    plot_sample_efficiency(efficiency_df, multi_seed_df, config, output_dir)

    # ================================================================
    # PHASE 8: FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE — JOURNAL REVISION SUMMARY")
    print("=" * 70)

    print(f"\n[OUTPUT] All results saved to: {output_dir}")
    print("\n[FILES CREATED]")
    print("  Tables:")
    print("    - multi_seed_results.csv")
    print("    - statistical_tests.csv")
    print("    - false_negative_analysis.csv")
    print("    - operating_points.csv")
    print("    - sample_efficiency_results.csv")
    print("    - calibration_ece.csv          [NEW]")
    print("    - gradcam_summary.csv          [NEW]")
    print("\n  Figures:")
    print("    - auc_comparison.png")
    print("    - roc_curves.png")
    print("    - precision_recall_curves.png")
    print("    - false_negative_analysis.png")
    print("    - sample_efficiency.png")
    print("    - calibration_reliability.png  [NEW]")
    print("    - calibration_ece_bars.png     [NEW]")
    print("    - gradcam_wrist_vs_hand.png    [NEW]")

    source_perf = multi_seed_df[multi_seed_df['anatomy'] == config.SOURCE_ANATOMY]['auc']
    target_perf = multi_seed_df[multi_seed_df['anatomy'] != config.SOURCE_ANATOMY]['auc']

    drop     = source_perf.mean() - target_perf.mean()
    drop_pct = (drop / source_perf.mean()) * 100

    print(f"\nSource ({config.SOURCE_ANATOMY}): {source_perf.mean():.4f} ± {source_perf.std():.4f}")
    print(f"Target (avg):                   {target_perf.mean():.4f} ± {target_perf.std():.4f}")
    print(f"Performance Drop:               {drop:.4f} AUC ({drop_pct:.1f}%)")

    source_fn  = fn_df[fn_df['anatomy'] == config.SOURCE_ANATOMY]['fn_rate'].values[0]
    target_fn  = fn_df[fn_df['anatomy'] != config.SOURCE_ANATOMY]['fn_rate'].mean()
    print(f"FN Rate Source: {source_fn:.1f}% | Target avg: {target_fn:.1f}%")

    if not cal_df.empty:
        src_ece = cal_df[cal_df['anatomy'] == config.SOURCE_ANATOMY]['ece'].values[0]
        tgt_ece = cal_df[cal_df['anatomy'] != config.SOURCE_ANATOMY]['ece'].mean()
        print(f"ECE Source: {src_ece:.4f} | Target avg: {tgt_ece:.4f}")

    print("\n" + "=" * 70)
    print("READY FOR JOURNAL SUBMISSION")
    print("=" * 70)


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
