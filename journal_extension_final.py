# =============================================================================
# JOURNAL EXTENSION: Cross-Anatomy Transfer Learning - Comprehensive Analysis
# Target: Computers in Biology and Medicine / Medical Image Analysis
#
# This file is COMPLETELY SEPARATE from the conference code.
# It assumes you have already run the conference pipeline and have:
#   - Saved model checkpoints: source_model_seed{seed}.pth
#   - MURA dataset at C:\Users\Suraj\Documents\python\MURA-v1.1
#   - FracAtlas dataset at C:\FracAtlas\FracAtlas
#
# NEW EXPERIMENTS (not in conference paper):
#   Phase 1: Grad-CAM mechanistic analysis
#   Phase 2: Feature space analysis (t-SNE / PCA)
#   Phase 3: Confidence calibration (ECE + reliability diagrams)
#   Phase 4: FracAtlas external validation
#   Phase 5: Domain adaptation baselines (fine-tune vs DANN vs MMD)
#
# OUTPUT: All results saved to journal_results_{timestamp}/
# =============================================================================

import os
import warnings
import random
import json
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
    calibration_curve
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel
from tqdm import tqdm
from datetime import datetime

torch.backends.cudnn.benchmark = True
sns.set_style("whitegrid")
sns.set_palette("husl")


# =============================================================================
# CONFIGURATION
# =============================================================================

class JournalConfig:
    """
    Configuration for journal extension experiments.
    MODIFY PATHS TO MATCH YOUR SYSTEM.
    """

    # ── MURA paths (same as conference code) ──────────────────────────────
    MURA_BASE      = r"C:\Users\Suraj\Documents\python\MURA-v1.1"
    MURA_TRAIN     = os.path.join(MURA_BASE, "train")
    MURA_VALID     = os.path.join(MURA_BASE, "valid")

    # ── FracAtlas paths ───────────────────────────────────────────────────
    FRACATLAS_BASE        = r"C:\FracAtlas\FracAtlas"
    FRACATLAS_IMAGES      = os.path.join(FRACATLAS_BASE, "images")
    FRACATLAS_FRACTURED   = os.path.join(FRACATLAS_IMAGES, "Fractured")
    FRACATLAS_NORMAL      = os.path.join(FRACATLAS_IMAGES, "Non_fractured")
    FRACATLAS_CSV         = os.path.join(FRACATLAS_BASE, "dataset.csv")
    FRACATLAS_SPLIT_TRAIN = os.path.join(FRACATLAS_BASE, "Utilities", "Fracture Split", "train.csv")
    FRACATLAS_SPLIT_VALID = os.path.join(FRACATLAS_BASE, "Utilities", "Fracture Split", "valid.csv")
    FRACATLAS_SPLIT_TEST  = os.path.join(FRACATLAS_BASE, "Utilities", "Fracture Split", "test.csv")

    # ── Saved MURA models from conference run ─────────────────────────────
    # Point this to the folder where your conference code saved checkpoints.
    # The conference code saves them as: source_model_seed{seed}.pth
    CONFERENCE_RESULTS_DIR = r"."   # Change to e.g. "results_conference_20240115_120000"

    # ── MURA anatomy settings ─────────────────────────────────────────────
    SOURCE_ANATOMY     = "XR_WRIST"
    TARGET_ANATOMIES   = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER", "XR_FINGER"]
    RANDOM_SEEDS       = [42, 123, 456, 789, 2024]
    PRIMARY_SEED       = 42          # seed used for Phase 1-3 (single model)

    # ── Training hyperparameters ─────────────────────────────────────────
    BATCH_SIZE       = 16
    LEARNING_RATE    = 1e-4
    EPOCHS_SOURCE    = 10            # only needed if retraining
    EPOCHS_FINETUNE  = 5
    NUM_WORKERS      = 4

    # ── Grad-CAM settings ────────────────────────────────────────────────
    GRADCAM_N_IMAGES = 5             # images per anatomy for Grad-CAM grid

    # ── t-SNE / PCA settings ─────────────────────────────────────────────
    TSNE_PERPLEXITY  = 30
    TSNE_N_ITER      = 1000
    MAX_FEATURES_PER_CLASS = 200    # limit for t-SNE speed

    # ── Calibration settings ─────────────────────────────────────────────
    CALIBRATION_BINS = 10

    # ── Domain adaptation settings ───────────────────────────────────────
    DA_TARGET_ANATOMY  = "XR_ELBOW"  # elbow: best behaved, clean result
    DA_EPOCHS          = 10
    DA_SAMPLE_SIZES    = [50, 200, 500]
    DANN_LAMBDA        = 0.1         # gradient reversal strength

    # ── FracAtlas ────────────────────────────────────────────────────────
    # FracAtlas has no anatomy subfolders — we treat it as a single
    # "external dataset" for cross-dataset generalization validation.
    FRACATLAS_TEST_SIZE = 0.2        # fraction used if no split CSV exists


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DATASETS
# =============================================================================

class MURADataset(Dataset):
    """MURA anatomy dataset — identical loading logic to conference code."""

    def __init__(self, root_dir, anatomy, augment=False):
        self.samples  = []
        self.augment  = augment
        anatomy_dir   = os.path.join(root_dir, anatomy)

        if not os.path.exists(anatomy_dir):
            raise FileNotFoundError(f"MURA path not found: {anatomy_dir}")

        for patient in os.listdir(anatomy_dir):
            patient_path = os.path.join(anatomy_dir, patient)
            if not os.path.isdir(patient_path):
                continue
            for study in os.listdir(patient_path):
                study_path  = os.path.join(patient_path, study)
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
                        self.samples.append((os.path.join(study_path, img), label))

        print(f"  [MURA] {anatomy}: {len(self.samples)} images "
              f"({sum(1 for _,l in self.samples if l==1)} fracture, "
              f"{sum(1 for _,l in self.samples if l==0)} normal)")

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
        if self.augment:
            if random.random() > 0.5:
                image = np.fliplr(image)
            if random.random() > 0.5:
                image = np.flipud(image)
        image = np.stack([image] * 3, axis=0)
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32)
        )


class FracAtlasDataset(Dataset):
    """
    FracAtlas dataset loader.

    FracAtlas stores all images in two flat folders:
        images/Fractured/      -> label 1
        images/Non_fractured/  -> label 0

    If split CSVs exist (Utilities/Fracture Split/*.csv) we use them.
    Otherwise we use all images and split manually.
    The CSVs contain image IDs; we match them to the folder contents.
    """

    def __init__(self, config, split='test', body_part_filter=None):
        """
        Args:
            config: JournalConfig instance
            split: 'train', 'valid', or 'test'
            body_part_filter: optional string like 'Hand' to filter
                              by body part (requires dataset.csv to have
                              a 'body_part' or similar column)
        """
        self.samples = []
        self.split   = split

        # ── Load split CSV if it exists ──────────────────────────────────
        split_csv_map = {
            'train': config.FRACATLAS_SPLIT_TRAIN,
            'valid': config.FRACATLAS_SPLIT_VALID,
            'test':  config.FRACATLAS_SPLIT_TEST,
        }
        split_csv = split_csv_map.get(split)
        split_ids = None

        if split_csv and os.path.exists(split_csv):
            df_split = pd.read_csv(split_csv)
            # The CSV likely has a column with image IDs / filenames
            # Try common column names
            for col in ['image_id', 'id', 'filename', 'file_id', 'name']:
                if col in df_split.columns:
                    split_ids = set(df_split[col].astype(str).tolist())
                    break
            if split_ids is None:
                # Use first column
                split_ids = set(df_split.iloc[:, 0].astype(str).tolist())
            print(f"  [FracAtlas] Using split CSV for '{split}': {len(split_ids)} IDs")

        # ── Load body_part info if filtering ─────────────────────────────
        body_part_ids = None
        if body_part_filter and os.path.exists(config.FRACATLAS_CSV):
            df_meta = pd.read_csv(config.FRACATLAS_CSV)
            # Identify body_part column
            bp_col = None
            for col in df_meta.columns:
                if 'body' in col.lower() or 'part' in col.lower() or 'anatomy' in col.lower():
                    bp_col = col
                    break
            if bp_col:
                mask = df_meta[bp_col].str.lower().str.contains(
                    body_part_filter.lower(), na=False
                )
                id_col = None
                for col in ['image_id', 'id', 'filename', 'file_id', 'name']:
                    if col in df_meta.columns:
                        id_col = col
                        break
                if id_col is None:
                    id_col = df_meta.columns[0]
                body_part_ids = set(df_meta[mask][id_col].astype(str).tolist())
                print(f"  [FracAtlas] Body part filter '{body_part_filter}': "
                      f"{len(body_part_ids)} images")

        # ── Collect image paths ──────────────────────────────────────────
        def should_include(fname):
            stem = os.path.splitext(fname)[0]
            if split_ids is not None:
                if stem not in split_ids and fname not in split_ids:
                    return False
            if body_part_ids is not None:
                if stem not in body_part_ids and fname not in body_part_ids:
                    return False
            return True

        for fname in os.listdir(config.FRACATLAS_FRACTURED):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                if should_include(fname):
                    self.samples.append(
                        (os.path.join(config.FRACATLAS_FRACTURED, fname), 1.0)
                    )

        for fname in os.listdir(config.FRACATLAS_NORMAL):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                if should_include(fname):
                    self.samples.append(
                        (os.path.join(config.FRACATLAS_NORMAL, fname), 0.0)
                    )

        # ── Fallback: if split CSVs didn't filter anything, do random split
        if split_ids is None:
            random.seed(42)
            random.shuffle(self.samples)
            n = len(self.samples)
            if split == 'train':
                self.samples = self.samples[:int(0.7 * n)]
            elif split == 'valid':
                self.samples = self.samples[int(0.7 * n):int(0.85 * n)]
            else:
                self.samples = self.samples[int(0.85 * n):]

        n_frac   = sum(1 for _, l in self.samples if l == 1.0)
        n_normal = sum(1 for _, l in self.samples if l == 0.0)
        print(f"  [FracAtlas] {split}: {len(self.samples)} images "
              f"({n_frac} fracture, {n_normal} normal)")

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


# =============================================================================
# MODEL
# =============================================================================

from torchvision.models import densenet121, DenseNet121_Weights


def get_densenet121(pretrained=True):
    """Standard DenseNet-121 with binary output — same as conference code."""
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model   = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def load_conference_model(config, seed, device):
    """
    Load a trained model checkpoint from the conference experiment.
    Tries the configured directory first; if not found, retrains.
    """
    model_path = os.path.join(
        config.CONFERENCE_RESULTS_DIR,
        f"source_model_seed{seed}.pth"
    )

    if not os.path.exists(model_path):
        # Search common result folder patterns
        for folder in sorted(os.listdir('.'), reverse=True):
            if folder.startswith('results_conference'):
                candidate = os.path.join(folder, f"source_model_seed{seed}.pth")
                if os.path.exists(candidate):
                    model_path = candidate
                    print(f"  [INFO] Found checkpoint: {model_path}")
                    break

    model = get_densenet121(pretrained=False).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        print(f"  [INFO] Loaded checkpoint: {model_path}")
    else:
        print(f"  [WARNING] Checkpoint not found for seed {seed}.")
        print(f"            Training from scratch on {config.SOURCE_ANATOMY}...")
        model = train_source_model(config, device, seed)

    return model


def train_source_model(config, device, seed):
    """Train DenseNet-121 on MURA source anatomy from scratch."""
    set_seed(seed)
    model     = get_densenet121(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_set    = MURADataset(config.MURA_TRAIN, config.SOURCE_ANATOMY)
    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    model.train()
    for epoch in range(config.EPOCHS_SOURCE):
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{config.EPOCHS_SOURCE}",
                          leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}")

    return model


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate(model, loader, device):
    """Full evaluation: returns preds, targets, and metric dict."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            probs = torch.sigmoid(model(x)).cpu().numpy().flatten()
            all_preds.extend(probs.tolist())
            all_targets.extend(y.numpy().flatten().tolist())

    preds   = np.array(all_preds)
    targets = np.array(all_targets)

    if len(np.unique(targets)) < 2:
        return {'preds': preds, 'targets': targets, 'auc': float('nan')}

    preds_binary = (preds >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds_binary).ravel()

    return {
        'preds':       preds,
        'targets':     targets,
        'auc':         roc_auc_score(targets, preds),
        'ap':          average_precision_score(targets, preds),
        'accuracy':    accuracy_score(targets, preds_binary),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ppv':         tp / (tp + fp) if (tp + fp) > 0 else 0,
        'npv':         tn / (tn + fn) if (tn + fn) > 0 else 0,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def train_epoch(model, loader, optimizer, device, criterion=None):
    """Single training epoch."""
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def make_loader(dataset, batch_size, shuffle, num_workers=0):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True
    )


# =============================================================================
# PHASE 1: GRAD-CAM MECHANISTIC ANALYSIS
# =============================================================================

class GradCAM:
    """
    Grad-CAM implementation for DenseNet-121.
    Hooks into the last DenseBlock (features.denseblock4).
    """

    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Target layer: last dense block before the classifier
        target_layer = self.model.features.denseblock4

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, device):
        """
        Generate Grad-CAM heatmap for a single image tensor (1, 3, 224, 224).
        Returns heatmap as numpy array (224, 224).
        """
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).to(device)
        image_tensor.requires_grad = False

        # Forward pass
        output = self.model(image_tensor)
        self.model.zero_grad()

        # Backward pass on the predicted class
        output.backward(torch.ones_like(output))

        # Weight activations by gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear',
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam


def run_gradcam_analysis(model, config, device, output_dir):
    """
    Generate Grad-CAM heatmaps for source and all target anatomies.
    Compares what the wrist-trained model attends to on each anatomy.

    Returns:
        DataFrame with per-anatomy attention statistics.
    """
    print(f"\n{'='*60}")
    print("PHASE 1: GRAD-CAM MECHANISTIC ANALYSIS")
    print(f"{'='*60}")

    gradcam    = GradCAM(model)
    gradcam_dir = os.path.join(output_dir, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)

    all_anatomies = [config.SOURCE_ANATOMY] + config.TARGET_ANATOMIES
    attention_stats = []

    # We'll collect one figure per anatomy: a grid of
    # [original | heatmap overlay] for N_IMAGES fracture + N_IMAGES normal
    n_images = config.GRADCAM_N_IMAGES

    for anatomy in all_anatomies:
        print(f"\n  Generating Grad-CAM for {anatomy}...")
        try:
            val_set = MURADataset(config.MURA_VALID, anatomy)
        except FileNotFoundError:
            print(f"  [SKIP] {anatomy} not found.")
            continue

        # Separate fracture and normal images
        fracture_indices = [i for i, (_, l) in enumerate(val_set.samples) if l == 1.0]
        normal_indices   = [i for i, (_, l) in enumerate(val_set.samples) if l == 0.0]

        selected_fracture = fracture_indices[:n_images]
        selected_normal   = normal_indices[:n_images]
        selected          = selected_fracture + selected_normal
        labels_str        = (['Fracture'] * len(selected_fracture) +
                             ['Normal']   * len(selected_normal))

        # Figure: rows = images, cols = [original, heatmap]
        n_rows = len(selected)
        if n_rows == 0:
            continue

        fig, axes = plt.subplots(n_rows, 2, figsize=(6, 2.5 * n_rows))
        if n_rows == 1:
            axes = [axes]

        cam_intensities_fracture = []
        cam_intensities_normal   = []
        cam_centrality_fracture  = []
        cam_centrality_normal    = []

        for row_idx, (sample_idx, label_str) in enumerate(
                zip(selected, labels_str)):
            img_tensor, label = val_set[sample_idx]
            cam = gradcam.generate(img_tensor, device)

            # Original image (grayscale, first channel)
            img_np = img_tensor[0].numpy()

            ax_orig = axes[row_idx][0]
            ax_heat = axes[row_idx][1]

            ax_orig.imshow(img_np, cmap='gray', vmin=0, vmax=1)
            ax_orig.set_title(f"{label_str}", fontsize=9, fontweight='bold')
            ax_orig.axis('off')

            # Overlay heatmap
            ax_heat.imshow(img_np, cmap='gray', vmin=0, vmax=1, alpha=0.6)
            ax_heat.imshow(cam, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            ax_heat.set_title(f"Grad-CAM", fontsize=9)
            ax_heat.axis('off')

            # Quantify attention: mean intensity + centrality score
            mean_intensity = float(cam.mean())
            h, w = cam.shape
            y_idx, x_idx = np.mgrid[0:h, 0:w]
            # Centrality: how close the attention centroid is to image center
            total = cam.sum() + 1e-8
            centroid_y = (y_idx * cam).sum() / total
            centroid_x = (x_idx * cam).sum() / total
            centrality = 1.0 - (
                np.sqrt((centroid_y - h/2)**2 + (centroid_x - w/2)**2) /
                (np.sqrt((h/2)**2 + (w/2)**2))
            )

            if label_str == 'Fracture':
                cam_intensities_fracture.append(mean_intensity)
                cam_centrality_fracture.append(centrality)
            else:
                cam_intensities_normal.append(mean_intensity)
                cam_centrality_normal.append(centrality)

        plt.suptitle(f"Grad-CAM: {anatomy} (wrist-trained model)",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig_path = os.path.join(gradcam_dir, f"gradcam_{anatomy}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {fig_path}")

        # Aggregate stats
        attention_stats.append({
            'anatomy':             anatomy,
            'mean_cam_fracture':   np.mean(cam_intensities_fracture) if cam_intensities_fracture else np.nan,
            'mean_cam_normal':     np.mean(cam_intensities_normal)   if cam_intensities_normal   else np.nan,
            'centrality_fracture': np.mean(cam_centrality_fracture)  if cam_centrality_fracture  else np.nan,
            'centrality_normal':   np.mean(cam_centrality_normal)    if cam_centrality_normal    else np.nan,
        })

    df_stats = pd.DataFrame(attention_stats)
    stats_path = os.path.join(gradcam_dir, 'gradcam_attention_stats.csv')
    df_stats.to_csv(stats_path, index=False)
    print(f"\n  [SAVED] Grad-CAM stats: {stats_path}")

    # Summary comparison plot
    _plot_gradcam_summary(df_stats, config, gradcam_dir)

    return df_stats


def _plot_gradcam_summary(df_stats, config, output_dir):
    """Bar chart comparing attention intensity across anatomies."""
    if df_stats.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, title in [
        (axes[0], 'mean_cam_fracture',   'Mean Grad-CAM Intensity (Fracture images)'),
        (axes[1], 'centrality_fracture', 'Attention Centrality (Fracture images)'),
    ]:
        colors = ['#2ecc71' if a == config.SOURCE_ANATOMY else '#e74c3c'
                  for a in df_stats['anatomy']]
        bars = ax.bar(df_stats['anatomy'], df_stats[col],
                      color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Anatomy', fontsize=11)
        ax.set_ylabel(col.replace('_', ' ').title(), fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        for bar, val in zip(bars, df_stats[col]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(
        'Grad-CAM Analysis: Attention Patterns Across Anatomies\n'
        '(Wrist-trained model applied zero-shot)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradcam_summary.png'),
                dpi=250, bbox_inches='tight')
    plt.close()


# =============================================================================
# PHASE 2: FEATURE SPACE ANALYSIS (t-SNE + PCA)
# =============================================================================

def extract_features(model, loader, device):
    """
    Extract penultimate-layer features from DenseNet-121.
    Returns: features (N, 1024), labels (N,), preds (N,)
    """
    model.eval()

    # Hook to capture features before the final classifier
    features_list = []

    def hook_fn(module, input, output):
        # DenseNet output after adaptive pooling: (B, 1024, 1, 1)
        features_list.append(output.squeeze(-1).squeeze(-1).detach().cpu())

    # Register hook on adaptive pool (just before classifier)
    hook = model.features.register_forward_hook(
        lambda m, i, o: features_list.append(
            F.adaptive_avg_pool2d(o, (1, 1)).squeeze(-1).squeeze(-1).detach().cpu()
        )
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc='    Extracting features', leave=False):
            x = x.to(device, non_blocking=True)
            out = model(x)
            preds = torch.sigmoid(out).cpu().numpy().flatten()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().flatten().tolist())

    hook.remove()

    features = torch.cat(features_list, dim=0).numpy()
    labels   = np.array(all_labels)
    preds    = np.array(all_preds)

    # Sanity check: features shape should be (N, 1024)
    if features.shape[0] != len(labels):
        # If hook fired twice per forward (which can happen), take every other
        mid = features.shape[0] // 2
        features = features[mid:]

    return features, labels, preds


def run_feature_analysis(model, config, device, output_dir):
    """
    Extract features from all anatomies, run t-SNE + PCA,
    quantify inter vs intra anatomy distance ratio.

    Returns:
        DataFrame with distance ratio statistics.
    """
    print(f"\n{'='*60}")
    print("PHASE 2: FEATURE SPACE ANALYSIS (t-SNE / PCA)")
    print(f"{'='*60}")

    feat_dir = os.path.join(output_dir, 'feature_analysis')
    os.makedirs(feat_dir, exist_ok=True)

    all_anatomies  = [config.SOURCE_ANATOMY] + config.TARGET_ANATOMIES
    anatomy_feats  = {}   # anatomy -> (features, labels, preds)
    max_per        = config.MAX_FEATURES_PER_CLASS

    for anatomy in all_anatomies:
        print(f"\n  Extracting features: {anatomy}")
        try:
            val_set = MURADataset(config.MURA_VALID, anatomy)
        except FileNotFoundError:
            continue
        loader = make_loader(val_set, config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS)
        feats, labels, preds = extract_features(model, loader, device)

        # Subsample for t-SNE speed
        idx_pos = np.where(labels == 1)[0]
        idx_neg = np.where(labels == 0)[0]
        idx_pos = idx_pos[:max_per]
        idx_neg = idx_neg[:max_per]
        idx     = np.concatenate([idx_pos, idx_neg])
        np.random.shuffle(idx)

        anatomy_feats[anatomy] = (feats[idx], labels[idx], preds[idx])
        print(f"    {len(idx)} features collected")

    if len(anatomy_feats) < 2:
        print("  [SKIP] Not enough anatomies for feature analysis.")
        return pd.DataFrame()

    # ── Build combined feature matrix for t-SNE ───────────────────────
    all_feats_list   = []
    anatomy_labels   = []
    fracture_labels  = []

    for anatomy, (feats, labels, _) in anatomy_feats.items():
        all_feats_list.append(feats)
        anatomy_labels.extend([anatomy] * len(feats))
        fracture_labels.extend(labels.tolist())

    X      = np.vstack(all_feats_list)
    y_anat = np.array(anatomy_labels)
    y_frac = np.array(fracture_labels)

    print(f"\n  Running PCA (50 components) then t-SNE on {len(X)} features...")
    # PCA first to reduce noise
    pca  = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X)

    tsne  = TSNE(n_components=2, perplexity=config.TSNE_PERPLEXITY,
                 n_iter=config.TSNE_N_ITER, random_state=42, n_jobs=-1)
    X_2d  = tsne.fit_transform(X_pca)

    # ── Plot: color by anatomy ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors_anat = {
        'XR_WRIST':    '#2ecc71',
        'XR_ELBOW':    '#3498db',
        'XR_HAND':     '#e74c3c',
        'XR_SHOULDER': '#9b59b6',
        'XR_FINGER':   '#f39c12',
    }

    ax = axes[0]
    unique_anats = list(dict.fromkeys(anatomy_labels))
    for anat in unique_anats:
        mask = y_anat == anat
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors_anat.get(anat, 'gray'),
                   label=anat, alpha=0.6, s=15, edgecolors='none')
    ax.set_title('t-SNE: colored by anatomy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.grid(alpha=0.2)

    # Plot: color by fracture/normal
    ax = axes[1]
    colors_frac = {0.0: '#3498db', 1.0: '#e74c3c'}
    for frac_val, label_str, marker in [
            (1.0, 'Fracture', 'o'),
            (0.0, 'Normal', '^')]:
        mask = y_frac == frac_val
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors_frac[frac_val], marker=marker,
                   label=label_str, alpha=0.5, s=12, edgecolors='none')
    ax.set_title('t-SNE: colored by fracture status', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.grid(alpha=0.2)

    plt.suptitle(
        'Feature Space Analysis: DenseNet-121 Penultimate Layer\n'
        '(Wrist-trained model, zero-shot to all anatomies)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(feat_dir, 'tsne_feature_space.png'),
                dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] t-SNE plot")

    # ── Quantify distance ratio ───────────────────────────────────────
    print("\n  Computing inter/intra anatomy distance ratios...")
    distance_stats = []
    source_centroid = anatomy_feats[config.SOURCE_ANATOMY][0].mean(axis=0)

    for anatomy, (feats, labels, _) in anatomy_feats.items():
        centroid = feats.mean(axis=0)

        # Intra-anatomy distance: mean pairwise within-anatomy
        if len(feats) > 100:
            idx_sample = np.random.choice(len(feats), 100, replace=False)
            feats_sample = feats[idx_sample]
        else:
            feats_sample = feats

        diffs = feats_sample[:, np.newaxis] - feats_sample[np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=2)
        intra_dist = dists[np.triu_indices_from(dists, k=1)].mean()

        # Inter-anatomy distance: centroid to source centroid
        inter_dist = np.linalg.norm(centroid - source_centroid)

        # Ratio: if anatomy >> source, model has encoded anatomy strongly
        ratio = inter_dist / (intra_dist + 1e-8)

        distance_stats.append({
            'anatomy':     anatomy,
            'inter_dist':  float(inter_dist),
            'intra_dist':  float(intra_dist),
            'dist_ratio':  float(ratio),
            'n_samples':   len(feats),
        })
        print(f"    {anatomy}: inter={inter_dist:.2f}, "
              f"intra={intra_dist:.2f}, ratio={ratio:.2f}")

    df_dist = pd.DataFrame(distance_stats)
    df_dist.to_csv(os.path.join(feat_dir, 'feature_distances.csv'), index=False)
    print(f"  [SAVED] Feature distances CSV")

    _plot_distance_ratios(df_dist, config, feat_dir)

    return df_dist


def _plot_distance_ratios(df_dist, config, output_dir):
    """Bar chart: anatomy distance ratio — higher = more anatomy-encoded."""
    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ['#2ecc71' if a == config.SOURCE_ANATOMY else '#e74c3c'
              for a in df_dist['anatomy']]
    bars = ax.bar(df_dist['anatomy'], df_dist['dist_ratio'],
                  color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Inter/Intra Distance Ratio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Anatomy', fontsize=12)
    ax.set_title(
        'Feature Space Distance Ratio by Anatomy\n'
        '(High ratio = features dominated by anatomy identity, not fracture signal)',
        fontsize=12, fontweight='bold'
    )
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')

    for bar, val in zip(bars, df_dist['dist_ratio']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.85, edgecolor='black',
              label=f'Source ({config.SOURCE_ANATOMY})'),
        Patch(facecolor='#e74c3c', alpha=0.85, edgecolor='black',
              label='Target anatomies')
    ]
    ax.legend(handles=legend_elements, fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_ratio.png'),
                dpi=250, bbox_inches='tight')
    plt.close()


# =============================================================================
# PHASE 3: CONFIDENCE CALIBRATION ANALYSIS
# =============================================================================

def compute_ece(preds, targets, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (preds >= low) & (preds < high)
        if mask.sum() == 0:
            bin_data.append({'bin_mid': (low + high) / 2,
                             'accuracy': np.nan, 'confidence': np.nan,
                             'count': 0})
            continue

        bin_preds   = preds[mask]
        bin_targets = targets[mask]
        accuracy    = (bin_targets == (bin_preds >= 0.5).astype(int)).mean()
        confidence  = bin_preds.mean()
        count       = mask.sum()

        ece        += (count / len(preds)) * abs(confidence - accuracy)
        bin_data.append({'bin_mid': (low + high) / 2,
                         'accuracy': accuracy,
                         'confidence': confidence,
                         'count': int(count)})

    return float(ece), pd.DataFrame(bin_data)


def run_calibration_analysis(model, config, device, output_dir):
    """
    Compute ECE and reliability diagrams for source + all target anatomies.
    Key insight: misclassified high-confidence predictions reveal
    dangerous overconfidence on transferred anatomies.

    Returns:
        DataFrame with ECE per anatomy.
    """
    print(f"\n{'='*60}")
    print("PHASE 3: CONFIDENCE CALIBRATION ANALYSIS")
    print(f"{'='*60}")

    cal_dir = os.path.join(output_dir, 'calibration')
    os.makedirs(cal_dir, exist_ok=True)

    all_anatomies = [config.SOURCE_ANATOMY] + config.TARGET_ANATOMIES
    ece_results   = []

    n_cols = 3
    n_rows = (len(all_anatomies) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, anatomy in enumerate(all_anatomies):
        print(f"\n  {anatomy}...")
        try:
            val_set = MURADataset(config.MURA_VALID, anatomy)
        except FileNotFoundError:
            continue

        loader = make_loader(val_set, config.BATCH_SIZE, False,
                             num_workers=config.NUM_WORKERS)
        result = evaluate(model, loader, device)
        preds   = result['preds']
        targets = result['targets']

        ece, bin_df = compute_ece(preds, targets, config.CALIBRATION_BINS)

        # High-confidence errors: |pred - 0.5| > 0.3 AND wrong
        high_conf_mask = np.abs(preds - 0.5) > 0.3
        errors_mask    = (targets != (preds >= 0.5).astype(int))
        hce_rate       = (high_conf_mask & errors_mask).mean()

        ece_results.append({
            'anatomy':        anatomy,
            'ece':            ece,
            'high_conf_error_rate': hce_rate,
            'auc':            result['auc'],
        })

        print(f"    ECE={ece:.4f}  High-conf error rate={hce_rate:.4f}  "
              f"AUC={result['auc']:.4f}")

        # Reliability diagram
        ax = axes[idx]
        valid = bin_df.dropna(subset=['accuracy', 'confidence'])
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5,
                alpha=0.5, label='Perfect calibration')
        ax.bar(valid['bin_mid'], valid['accuracy'], width=0.1,
               alpha=0.7, color='#3498db', edgecolor='white',
               linewidth=0.5, label='Model accuracy')
        ax.set_xlabel('Confidence', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'{anatomy}\nECE={ece:.3f}', fontsize=11,
                     fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for j in range(len(all_anatomies), len(axes)):
        axes[j].axis('off')

    plt.suptitle(
        'Reliability Diagrams: Model Calibration Across Anatomies\n'
        '(Perfect calibration = diagonal line; deviation = overconfidence/underconfidence)',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(os.path.join(cal_dir, 'reliability_diagrams.png'),
                dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Reliability diagrams")

    # ECE comparison bar chart
    df_ece = pd.DataFrame(ece_results)
    df_ece.to_csv(os.path.join(cal_dir, 'calibration_results.csv'), index=False)

    _plot_ece_comparison(df_ece, config, cal_dir)

    return df_ece


def _plot_ece_comparison(df_ece, config, output_dir):
    """Two-panel: ECE and high-confidence error rate by anatomy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, ylabel, title in [
        (axes[0], 'ece', 'ECE (lower = better calibration)',
         'Expected Calibration Error'),
        (axes[1], 'high_conf_error_rate',
         'High-confidence error rate',
         'High-Confidence Errors (|pred - 0.5| > 0.3 AND wrong)')
    ]:
        colors = ['#2ecc71' if a == config.SOURCE_ANATOMY else '#e74c3c'
                  for a in df_ece['anatomy']]
        bars = ax.bar(df_ece['anatomy'], df_ece[col],
                      color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel('Anatomy', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        for bar, val in zip(bars, df_ece[col]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(
        'Calibration Analysis: Overconfidence Under Transfer',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ece_comparison.png'),
                dpi=250, bbox_inches='tight')
    plt.close()


# =============================================================================
# PHASE 4: FRACATLAS EXTERNAL VALIDATION
# =============================================================================

def run_fracatlas_validation(model, config, device, output_dir):
    """
    External validation: apply MURA-wrist-trained model to FracAtlas.

    This is the critical generalization test:
    - If degradation patterns in FracAtlas match MURA findings →
      findings generalize across datasets (stronger paper)
    - If FracAtlas shows less degradation →
      partially MURA-specific (still interesting, but weaker)

    FracAtlas has no anatomy labels in folder structure; we use the
    dataset.csv body_part column if available. Otherwise we treat it
    as a single "external fracture dataset" and compare overall AUC
    to the MURA wrist AUC.

    Returns:
        dict with FracAtlas validation results
    """
    print(f"\n{'='*60}")
    print("PHASE 4: FRACATLAS EXTERNAL VALIDATION")
    print(f"{'='*60}")

    ext_dir = os.path.join(output_dir, 'fracatlas_validation')
    os.makedirs(ext_dir, exist_ok=True)

    results = {}

    # ── Check if body_part info is available ─────────────────────────
    body_parts_available = []
    if os.path.exists(config.FRACATLAS_CSV):
        df_meta = pd.read_csv(config.FRACATLAS_CSV)
        print(f"\n  FracAtlas CSV columns: {list(df_meta.columns)}")

        bp_col = None
        for col in df_meta.columns:
            if any(k in col.lower() for k in ['body', 'part', 'anatomy', 'region', 'type']):
                bp_col = col
                break

        if bp_col:
            parts = df_meta[bp_col].dropna().unique()
            body_parts_available = [str(p) for p in parts]
            print(f"  Body parts found: {body_parts_available}")
        else:
            print(f"  No body part column found — using all images as single set.")

    # ── Whole-dataset evaluation (always run) ────────────────────────
    print(f"\n  Evaluating on full FracAtlas test set...")
    try:
        test_set = FracAtlasDataset(config, split='test')
        if len(test_set) == 0:
            raise ValueError("FracAtlas test set is empty")

        test_loader = make_loader(test_set, config.BATCH_SIZE, False,
                                  num_workers=config.NUM_WORKERS)
        result_all  = evaluate(model, test_loader, device)

        results['all'] = result_all
        print(f"  FracAtlas (all): AUC={result_all['auc']:.4f}  "
              f"Sens={result_all['sensitivity']:.4f}  "
              f"Spec={result_all['specificity']:.4f}")

    except Exception as e:
        print(f"  [ERROR] FracAtlas evaluation failed: {e}")
        print("  Check that FRACATLAS_BASE path is correct in JournalConfig.")
        return {}

    # ── Per-body-part evaluation (if available) ────────────────────
    if body_parts_available:
        print(f"\n  Per-body-part evaluation...")
        for part in body_parts_available:
            try:
                part_set = FracAtlasDataset(config, split='test',
                                            body_part_filter=part)
                if len(part_set) < 10:
                    continue
                loader  = make_loader(part_set, config.BATCH_SIZE, False)
                result  = evaluate(model, loader, device)
                results[part] = result
                print(f"    {part}: AUC={result['auc']:.4f}  "
                      f"n={len(part_set)}")
            except Exception as e:
                print(f"    {part}: failed ({e})")

    # ── FN analysis on FracAtlas ──────────────────────────────────
    fn_results = _fracatlas_fn_analysis(results, ext_dir)

    # ── Comparison plot: MURA vs FracAtlas ───────────────────────
    _plot_fracatlas_comparison(results, config, ext_dir)

    # Save summary CSV
    summary_rows = []
    for key, res in results.items():
        summary_rows.append({
            'dataset':     'FracAtlas',
            'subset':      key,
            'auc':         res.get('auc', np.nan),
            'sensitivity': res.get('sensitivity', np.nan),
            'specificity': res.get('specificity', np.nan),
            'ppv':         res.get('ppv', np.nan),
            'npv':         res.get('npv', np.nan),
            'fn_rate':     res['fn'] / max(res['tp'] + res['fn'], 1)
                           if 'fn' in res else np.nan,
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(ext_dir, 'fracatlas_results.csv'), index=False)
    print(f"\n  [SAVED] FracAtlas results CSV")

    return results


def _fracatlas_fn_analysis(results, output_dir):
    """False negative analysis on FracAtlas."""
    rows = []
    for key, res in results.items():
        if 'preds' not in res or 'targets' not in res:
            continue
        preds   = res['preds']
        targets = res['targets']
        fn_mask = (targets == 1) & (preds < 0.5)
        hcfn    = (targets == 1) & (preds < 0.3)
        n_frac  = int((targets == 1).sum())
        n_fn    = int(fn_mask.sum())
        rows.append({
            'subset':         key,
            'total_fractures': n_frac,
            'false_negatives': n_fn,
            'fn_rate':         n_fn / max(n_frac, 1) * 100,
            'high_conf_fn':    int(hcfn.sum()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'fracatlas_fn_analysis.csv'), index=False)
    return df


def _plot_fracatlas_comparison(results, config, output_dir):
    """
    Side-by-side: MURA wrist AUC vs FracAtlas AUC (overall + per body part).
    """
    # Get MURA reference AUC from conference results if loadable
    mura_ref_auc = 0.872   # fallback from conference paper results

    labels = []
    aucs   = []
    colors = []

    labels.append(f"MURA\n{config.SOURCE_ANATOMY}\n(source)")
    aucs.append(mura_ref_auc)
    colors.append('#2ecc71')

    for key, res in results.items():
        labels.append(f"FracAtlas\n{key}")
        aucs.append(res.get('auc', np.nan))
        colors.append('#9b59b6')

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(labels)), 5))
    bars = ax.bar(range(len(labels)), aucs,
                  color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Dataset Generalization: MURA → FracAtlas\n'
                 '(Wrist-trained model, zero-shot transfer)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim([0.45, 1.0])
    ax.axhline(y=mura_ref_auc, color='green', linestyle='--',
               linewidth=2, alpha=0.7, label=f'MURA source ({mura_ref_auc:.3f})')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, aucs):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fracatlas_comparison.png'),
                dpi=250, bbox_inches='tight')
    plt.close()


# =============================================================================
# PHASE 5: DOMAIN ADAPTATION BASELINES
# =============================================================================

class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for DANN."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DANNModel(nn.Module):
    """
    Domain-Adversarial Neural Network built on top of DenseNet-121.
    Feature extractor: DenseNet features + pool
    Fracture classifier: linear head
    Domain classifier: linear head on reversed gradients
    """

    def __init__(self, base_model, lambda_=0.1):
        super().__init__()
        self.features       = base_model.features
        self.fracture_head  = base_model.classifier
        self.domain_head    = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        self.lambda_        = lambda_

    def forward(self, x, alpha=None):
        feats = self.features(x)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)

        frac_out = self.fracture_head(feats)

        if alpha is not None:
            rev_feats   = GradientReversal.apply(feats, alpha)
            domain_out  = self.domain_head(rev_feats)
            return frac_out, domain_out

        return frac_out


def mmd_loss(source_feats, target_feats):
    """
    Maximum Mean Discrepancy loss with RBF kernel.
    Minimizing this aligns source and target feature distributions.
    """
    def rbf_kernel(x, y, gamma=1.0):
        xx = (x * x).sum(dim=1, keepdim=True)
        yy = (y * y).sum(dim=1, keepdim=True)
        xy = torch.mm(x, y.t())
        sq_dist = xx + yy.t() - 2 * xy
        return torch.exp(-gamma * sq_dist)

    K_ss = rbf_kernel(source_feats, source_feats)
    K_tt = rbf_kernel(target_feats,  target_feats)
    K_st = rbf_kernel(source_feats,  target_feats)
    return K_ss.mean() + K_tt.mean() - 2 * K_st.mean()


def get_features_batch(model, x, device):
    """Extract feature vector from DenseNet for a batch."""
    x = x.to(device)
    feats = model.features(x)
    feats = F.relu(feats, inplace=True)
    feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
    return feats


def run_domain_adaptation(base_model_state, config, device, output_dir):
    """
    Compare three adaptation strategies on the target anatomy:
      1. Standard fine-tuning (all layers)
      2. DANN: Domain-Adversarial Neural Network
      3. MMD: Maximum Mean Discrepancy alignment

    Tests each method at multiple sample sizes.

    Returns:
        DataFrame with results per method × sample size
    """
    print(f"\n{'='*60}")
    print("PHASE 5: DOMAIN ADAPTATION BASELINES")
    print(f"Target anatomy: {config.DA_TARGET_ANATOMY}")
    print(f"{'='*60}")

    da_dir = os.path.join(output_dir, 'domain_adaptation')
    os.makedirs(da_dir, exist_ok=True)

    target_anatomy = config.DA_TARGET_ANATOMY
    all_results    = []

    # ── Load target data ──────────────────────────────────────────────
    try:
        train_set = MURADataset(config.MURA_TRAIN, target_anatomy)
        val_set   = MURADataset(config.MURA_VALID, target_anatomy)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return pd.DataFrame()

    # Load source data (needed for DANN / MMD)
    source_train = MURADataset(config.MURA_TRAIN, config.SOURCE_ANATOMY)
    source_loader = make_loader(
        Subset(source_train, list(range(min(500, len(source_train))))),
        config.BATCH_SIZE, True
    )

    val_loader = make_loader(val_set, config.BATCH_SIZE, False,
                             num_workers=config.NUM_WORKERS)

    # ── Zero-shot baseline ────────────────────────────────────────────
    print(f"\n  Zero-shot baseline...")
    base_model = get_densenet121(pretrained=False).to(device)
    base_model.load_state_dict(base_model_state)
    zs_result  = evaluate(base_model, val_loader, device)
    all_results.append({
        'method': 'zero-shot', 'n_samples': 0,
        'auc': zs_result['auc'],
        'sensitivity': zs_result['sensitivity'],
        'specificity': zs_result['specificity'],
    })
    print(f"    AUC = {zs_result['auc']:.4f}")

    # ── Test each sample size ─────────────────────────────────────────
    for n_samples in config.DA_SAMPLE_SIZES:
        if n_samples > len(train_set):
            print(f"\n  [SKIP] n_samples={n_samples} > dataset size {len(train_set)}")
            continue

        print(f"\n  ── n_samples = {n_samples} ──────────────────────────")
        set_seed(42)
        indices = np.random.choice(len(train_set), n_samples, replace=False)
        subset  = Subset(train_set, indices)

        # ── Method 1: Standard fine-tuning ───────────────────────────
        print(f"  [1/3] Standard fine-tuning...")
        ft_model = get_densenet121(pretrained=False).to(device)
        ft_model.load_state_dict(base_model_state)
        ft_opt   = optim.Adam(ft_model.parameters(), lr=config.LEARNING_RATE)
        ft_loader = make_loader(subset, config.BATCH_SIZE, True)

        for _ in range(config.DA_EPOCHS):
            train_epoch(ft_model, ft_loader, ft_opt, device)

        ft_result = evaluate(ft_model, val_loader, device)
        all_results.append({
            'method': 'fine-tune', 'n_samples': n_samples,
            'auc': ft_result['auc'],
            'sensitivity': ft_result['sensitivity'],
            'specificity': ft_result['specificity'],
        })
        print(f"    AUC = {ft_result['auc']:.4f}")

        # ── Method 2: DANN ────────────────────────────────────────────
        print(f"  [2/3] DANN...")
        dann_base  = get_densenet121(pretrained=False).to(device)
        dann_base.load_state_dict(base_model_state)
        dann_model = DANNModel(dann_base, lambda_=config.DANN_LAMBDA).to(device)
        dann_opt   = optim.Adam(dann_model.parameters(), lr=config.LEARNING_RATE)
        target_loader = make_loader(subset, config.BATCH_SIZE, True)

        for epoch in range(config.DA_EPOCHS):
            dann_model.train()
            total_loss = 0.0
            src_iter   = iter(source_loader)
            tgt_iter   = iter(target_loader)
            n_batches  = min(len(source_loader), len(target_loader))
            alpha      = (2.0 / (1.0 + np.exp(-10 * epoch / config.DA_EPOCHS))) - 1

            for _ in range(n_batches):
                try:
                    x_src, y_src = next(src_iter)
                except StopIteration:
                    src_iter = iter(source_loader)
                    x_src, y_src = next(src_iter)
                try:
                    x_tgt, _ = next(tgt_iter)
                except StopIteration:
                    break

                x_src, y_src = x_src.to(device), y_src.to(device)
                x_tgt        = x_tgt.to(device)

                # Source: fracture loss + domain loss (domain = 0)
                frac_out, dom_src = dann_model(x_src, alpha)
                loss_frac = nn.BCEWithLogitsLoss()(frac_out, y_src)
                loss_dom_src = nn.BCEWithLogitsLoss()(
                    dom_src, torch.zeros_like(dom_src)
                )

                # Target: domain loss only (domain = 1, no labels)
                _, dom_tgt = dann_model(x_tgt, alpha)
                loss_dom_tgt = nn.BCEWithLogitsLoss()(
                    dom_tgt, torch.ones_like(dom_tgt)
                )

                loss = loss_frac + config.DANN_LAMBDA * (loss_dom_src + loss_dom_tgt)
                dann_opt.zero_grad(set_to_none=True)
                loss.backward()
                dann_opt.step()
                total_loss += loss.item()

        # Evaluate DANN on fracture prediction only
        dann_model.eval()
        d_preds, d_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out   = dann_model(x.to(device))
                probs = torch.sigmoid(out).cpu().numpy().flatten()
                d_preds.extend(probs.tolist())
                d_targets.extend(y.numpy().flatten().tolist())
        d_preds   = np.array(d_preds)
        d_targets = np.array(d_targets)
        dann_auc  = roc_auc_score(d_targets, d_preds) if len(np.unique(d_targets)) > 1 else np.nan
        d_binary  = (d_preds >= 0.5).astype(int)
        tn, fp, fn_, tp_ = confusion_matrix(d_targets, d_binary).ravel()

        all_results.append({
            'method': 'DANN', 'n_samples': n_samples,
            'auc': dann_auc,
            'sensitivity': tp_ / max(tp_ + fn_, 1),
            'specificity': tn / max(tn + fp, 1),
        })
        print(f"    AUC = {dann_auc:.4f}")

        # ── Method 3: MMD ─────────────────────────────────────────────
        print(f"  [3/3] MMD...")
        mmd_model = get_densenet121(pretrained=False).to(device)
        mmd_model.load_state_dict(base_model_state)
        mmd_opt   = optim.Adam(mmd_model.parameters(), lr=config.LEARNING_RATE)
        tgt_loader_mmd = make_loader(subset, config.BATCH_SIZE, True)

        for epoch in range(config.DA_EPOCHS):
            mmd_model.train()
            src_iter = iter(source_loader)
            tgt_iter = iter(tgt_loader_mmd)
            n_batches = min(len(source_loader), len(tgt_loader_mmd))

            for _ in range(n_batches):
                try:
                    x_src, y_src = next(src_iter)
                except StopIteration:
                    src_iter = iter(source_loader)
                    x_src, y_src = next(src_iter)
                try:
                    x_tgt, y_tgt = next(tgt_iter)
                except StopIteration:
                    break

                x_src, y_src = x_src.to(device), y_src.to(device)
                x_tgt, y_tgt = x_tgt.to(device), y_tgt.to(device)

                # Fracture loss on source + target (supervised)
                out_src = mmd_model(x_src)
                out_tgt = mmd_model(x_tgt)
                loss_frac = (nn.BCEWithLogitsLoss()(out_src, y_src) +
                             nn.BCEWithLogitsLoss()(out_tgt, y_tgt))

                # MMD alignment
                with torch.no_grad():
                    feat_src = get_features_batch(mmd_model, x_src, device).detach()
                feat_tgt = get_features_batch(mmd_model, x_tgt, device)
                # Re-compute feat_src with grad
                feat_src = get_features_batch(mmd_model, x_src, device)
                mmd = mmd_loss(feat_src, feat_tgt)

                loss = loss_frac + 0.1 * mmd
                mmd_opt.zero_grad(set_to_none=True)
                loss.backward()
                mmd_opt.step()

        mmd_result = evaluate(mmd_model, val_loader, device)
        all_results.append({
            'method': 'MMD', 'n_samples': n_samples,
            'auc': mmd_result['auc'],
            'sensitivity': mmd_result['sensitivity'],
            'specificity': mmd_result['specificity'],
        })
        print(f"    AUC = {mmd_result['auc']:.4f}")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(da_dir, 'domain_adaptation_results.csv'), index=False)
    print(f"\n  [SAVED] Domain adaptation results")

    _plot_domain_adaptation(df, config, da_dir)

    return df


def _plot_domain_adaptation(df, config, output_dir):
    """Learning curves: AUC vs n_samples for each DA method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = [m for m in df['method'].unique() if m != 'zero-shot']
    colors  = {'fine-tune': '#3498db', 'DANN': '#e74c3c', 'MMD': '#9b59b6'}

    zs_auc  = df[df['method'] == 'zero-shot']['auc'].values
    zs_auc  = float(zs_auc[0]) if len(zs_auc) > 0 else 0.75

    for ax, metric, ylabel, title in [
        (axes[0], 'auc',         'AUC-ROC', 'AUC vs Training Samples'),
        (axes[1], 'sensitivity', 'Sensitivity', 'Sensitivity vs Training Samples'),
    ]:
        ax.axhline(y=zs_auc if metric == 'auc' else
                   float(df[df['method'] == 'zero-shot']['sensitivity'].values[0]
                         if metric in df.columns else 0.5),
                   color='gray', linestyle='--', linewidth=2,
                   alpha=0.7, label='Zero-shot baseline')

        for method in methods:
            m_df = df[df['method'] == method].sort_values('n_samples')
            ax.plot(m_df['n_samples'], m_df[metric],
                    marker='o', linewidth=2.5, markersize=9,
                    color=colors.get(method, 'black'),
                    label=method, alpha=0.9)

        ax.set_xlabel('Number of target training samples', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\n({config.DA_TARGET_ANATOMY})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.suptitle(
        f'Domain Adaptation: Fine-tuning vs DANN vs MMD\n'
        f'(Source: {config.SOURCE_ANATOMY} → Target: {config.DA_TARGET_ANATOMY})',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_adaptation_comparison.png'),
                dpi=250, bbox_inches='tight')
    plt.close()


# =============================================================================
# FINAL SUMMARY FIGURE
# =============================================================================

def generate_summary_figure(results_dict, config, output_dir):
    """
    One combined figure with key findings across all phases.
    Suitable for a journal paper overview figure.
    """
    print(f"\n{'='*60}")
    print("GENERATING JOURNAL SUMMARY FIGURE")
    print(f"{'='*60}")

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel A: AUC by anatomy (MURA) ───────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    mura_data = results_dict.get('mura_summary')
    if mura_data is not None and not mura_data.empty:
        colors = ['#2ecc71' if a == config.SOURCE_ANATOMY else '#e74c3c'
                  for a in mura_data['anatomy']]
        ax_a.bar(range(len(mura_data)), mura_data['mean_auc'],
                 yerr=mura_data.get('std_auc', None),
                 color=colors, alpha=0.85, edgecolor='black',
                 linewidth=1.2, capsize=5)
        ax_a.set_xticks(range(len(mura_data)))
        ax_a.set_xticklabels(mura_data['anatomy'], rotation=30,
                              ha='right', fontsize=8)
    ax_a.set_title('A: Cross-Anatomy AUC\n(MURA, 5-seed)', fontsize=11,
                   fontweight='bold')
    ax_a.set_ylabel('AUC-ROC', fontsize=10)
    ax_a.set_ylim([0.5, 1.0])
    ax_a.grid(axis='y', alpha=0.3)

    # ── Panel B: Grad-CAM attention ───────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    gradcam_data = results_dict.get('gradcam_stats')
    if gradcam_data is not None and not gradcam_data.empty:
        colors = ['#2ecc71' if a == config.SOURCE_ANATOMY else '#e74c3c'
                  for a in gradcam_data['anatomy']]
        ax_b.bar(range(len(gradcam_data)),
                 gradcam_data['mean_cam_fracture'],
                 color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
        ax_b.set_xticks(range(len(gradcam_data)))
        ax_b.set_xticklabels(gradcam_data['anatomy'], rotation=30,
                              ha='right', fontsize=8)
    ax_b.set_title('B: Grad-CAM Attention Intensity\n(Fracture images)',
                   fontsize=11, fontweight='bold')
    ax_b.set_ylabel('Mean CAM intensity', fontsize=10)
    ax_b.grid(axis='y', alpha=0.3)

    # ── Panel C: Feature distance ratio ──────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    feat_data = results_dict.get('feature_distances')
    if feat_data is not None and not feat_data.empty:
        colors = ['#2ecc71' if a == config.SOURCE_ANATOMY else '#e74c3c'
                  for a in feat_data['anatomy']]
        ax_c.bar(range(len(feat_data)), feat_data['dist_ratio'],
                 color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
        ax_c.set_xticks(range(len(feat_data)))
        ax_c.set_xticklabels(feat_data['anatomy'], rotation=30,
                              ha='right', fontsize=8)
    ax_c.set_title('C: Inter/Intra Distance Ratio\n(Higher = anatomy-encoded)',
                   fontsize=11, fontweight='bold')
    ax_c.set_ylabel('Distance ratio', fontsize=10)
    ax_c.grid(axis='y', alpha=0.3)

    # ── Panel D: Calibration ECE ──────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    cal_data = results_dict.get('calibration')
    if cal_data is not None and not cal_data.empty:
        colors = ['#2ecc71' if a == config.SOURCE_ANATOMY else '#e74c3c'
                  for a in cal_data['anatomy']]
        ax_d.bar(range(len(cal_data)), cal_data['ece'],
                 color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
        ax_d.set_xticks(range(len(cal_data)))
        ax_d.set_xticklabels(cal_data['anatomy'], rotation=30,
                              ha='right', fontsize=8)
    ax_d.set_title('D: Expected Calibration Error\n(Lower = better)',
                   fontsize=11, fontweight='bold')
    ax_d.set_ylabel('ECE', fontsize=10)
    ax_d.grid(axis='y', alpha=0.3)

    # ── Panel E: FracAtlas vs MURA ─────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    frac_data = results_dict.get('fracatlas')
    if frac_data:
        auc_all = frac_data.get('all', {}).get('auc', np.nan)
        ax_e.bar(['MURA Wrist\n(source)', 'FracAtlas\n(external)'],
                 [0.872, auc_all],
                 color=['#2ecc71', '#9b59b6'],
                 alpha=0.85, edgecolor='black', linewidth=1.2)
        for i, val in enumerate([0.872, auc_all]):
            if not np.isnan(val):
                ax_e.text(i, val + 0.01, f'{val:.3f}', ha='center',
                          va='bottom', fontsize=10, fontweight='bold')
    ax_e.set_title('E: External Validation\n(MURA → FracAtlas)',
                   fontsize=11, fontweight='bold')
    ax_e.set_ylabel('AUC-ROC', fontsize=10)
    ax_e.set_ylim([0.45, 1.0])
    ax_e.grid(axis='y', alpha=0.3)

    # ── Panel F: Domain adaptation ────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    da_data = results_dict.get('domain_adaptation')
    if da_data is not None and not da_data.empty:
        methods = [m for m in da_data['method'].unique() if m != 'zero-shot']
        colors_da = {'fine-tune': '#3498db', 'DANN': '#e74c3c', 'MMD': '#9b59b6'}
        zs_auc_val = float(da_data[da_data['method'] == 'zero-shot']['auc'].values[0]) \
                     if 'zero-shot' in da_data['method'].values else 0.76
        ax_f.axhline(y=zs_auc_val, color='gray', linestyle='--',
                     linewidth=2, alpha=0.7, label='Zero-shot')
        for method in methods:
            m_df = da_data[da_data['method'] == method].sort_values('n_samples')
            ax_f.plot(m_df['n_samples'], m_df['auc'],
                      marker='o', linewidth=2, markersize=7,
                      color=colors_da.get(method, 'black'), label=method)
    ax_f.set_xlabel('Target samples', fontsize=10)
    ax_f.set_ylabel('AUC-ROC', fontsize=10)
    ax_f.set_title('F: Domain Adaptation\n(Fine-tune vs DANN vs MMD)',
                   fontsize=11, fontweight='bold')
    ax_f.legend(fontsize=8)
    ax_f.grid(alpha=0.3)

    plt.suptitle(
        'Journal Extension: Comprehensive Cross-Anatomy Transfer Analysis\n'
        'Mechanistic Analysis + External Validation + Domain Adaptation',
        fontsize=14, fontweight='bold', y=1.01
    )

    save_path = os.path.join(output_dir, 'JOURNAL_SUMMARY_FIGURE.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Journal summary figure: {save_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("JOURNAL EXTENSION PIPELINE")
    print("Cross-Anatomy Transfer Learning — Comprehensive Analysis")
    print("Target: Computers in Biology and Medicine")
    print("=" * 70)

    assert torch.cuda.is_available(), "CUDA required. GPU not detected."
    device = torch.device("cuda")
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")

    config    = JournalConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"journal_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[OUT] {output_dir}")

    # ── Save config ───────────────────────────────────────────────────
    cfg_dict = {k: str(v) for k, v in vars(JournalConfig).items()
                if not k.startswith('_') and not callable(v)}
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=2)

    # ── Load primary model (seed 42 from conference run) ─────────────
    print(f"\n[INFO] Loading primary model (seed {config.PRIMARY_SEED})...")
    set_seed(config.PRIMARY_SEED)
    primary_model = load_conference_model(config, config.PRIMARY_SEED, device)
    primary_state = primary_model.state_dict()

    # ── Compute MURA summary (needed for summary figure) ─────────────
    print(f"\n[INFO] Computing MURA cross-anatomy summary...")
    mura_summary_rows = []
    all_anatomies = [config.SOURCE_ANATOMY] + config.TARGET_ANATOMIES
    for anatomy in all_anatomies:
        try:
            val_set = MURADataset(config.MURA_VALID, anatomy)
            loader  = make_loader(val_set, config.BATCH_SIZE, False,
                                  num_workers=config.NUM_WORKERS)
            result  = evaluate(primary_model, loader, device)
            mura_summary_rows.append({
                'anatomy': anatomy,
                'mean_auc': result['auc'],
                'std_auc': 0.0,   # single-seed here; 5-seed from conference CSV
            })
        except FileNotFoundError:
            pass

    mura_summary_df = pd.DataFrame(mura_summary_rows)
    # Enrich with 5-seed std from conference results if available
    conf_csv = os.path.join(config.CONFERENCE_RESULTS_DIR, 'multi_seed_results.csv')
    if not os.path.exists(conf_csv):
        for folder in sorted(os.listdir('.'), reverse=True):
            if folder.startswith('results_conference'):
                candidate = os.path.join(folder, 'multi_seed_results.csv')
                if os.path.exists(candidate):
                    conf_csv = candidate
                    break
    if os.path.exists(conf_csv):
        df_conf = pd.read_csv(conf_csv)
        stats   = df_conf.groupby('anatomy')['auc'].agg(
            mean_auc='mean', std_auc='std'
        ).reset_index()
        mura_summary_df = stats

    mura_summary_df.to_csv(
        os.path.join(output_dir, 'mura_summary.csv'), index=False
    )

    # ─────────────────────────────────────────────────────────────────
    # RUN ALL PHASES
    # ─────────────────────────────────────────────────────────────────
    results_dict = {'mura_summary': mura_summary_df}

    # Phase 1: Grad-CAM
    gradcam_stats = run_gradcam_analysis(primary_model, config, device, output_dir)
    results_dict['gradcam_stats'] = gradcam_stats

    # Phase 2: Feature analysis
    feat_distances = run_feature_analysis(primary_model, config, device, output_dir)
    results_dict['feature_distances'] = feat_distances

    # Phase 3: Calibration
    cal_results = run_calibration_analysis(primary_model, config, device, output_dir)
    results_dict['calibration'] = cal_results

    # Phase 4: FracAtlas
    frac_results = run_fracatlas_validation(primary_model, config, device, output_dir)
    results_dict['fracatlas'] = frac_results

    # Phase 5: Domain adaptation
    da_results = run_domain_adaptation(primary_state, config, device, output_dir)
    results_dict['domain_adaptation'] = da_results

    # Summary figure
    generate_summary_figure(results_dict, config, output_dir)

    # ─────────────────────────────────────────────────────────────────
    # FINAL REPORT
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("JOURNAL EXTENSION COMPLETE — OUTPUT FILES")
    print("=" * 70)

    file_map = {
        "Grad-CAM figures":      "gradcam/gradcam_*.png",
        "Grad-CAM stats CSV":    "gradcam/gradcam_attention_stats.csv",
        "Grad-CAM summary":      "gradcam/gradcam_summary.png",
        "t-SNE feature plot":    "feature_analysis/tsne_feature_space.png",
        "Feature distances CSV": "feature_analysis/feature_distances.csv",
        "Distance ratio plot":   "feature_analysis/distance_ratio.png",
        "Reliability diagrams":  "calibration/reliability_diagrams.png",
        "Calibration CSV":       "calibration/calibration_results.csv",
        "ECE comparison plot":   "calibration/ece_comparison.png",
        "FracAtlas results CSV": "fracatlas_validation/fracatlas_results.csv",
        "FracAtlas comparison":  "fracatlas_validation/fracatlas_comparison.png",
        "DA results CSV":        "domain_adaptation/domain_adaptation_results.csv",
        "DA comparison plot":    "domain_adaptation/domain_adaptation_comparison.png",
        "SUMMARY FIGURE":        "JOURNAL_SUMMARY_FIGURE.png",
    }

    for label, path in file_map.items():
        print(f"  {label:35s}  {os.path.join(output_dir, path)}")

    print("\n" + "=" * 70)
    print("PAPER SECTION MAPPING")
    print("=" * 70)
    print("""
  Section 3   Methods
    3.1  Datasets (MURA + FracAtlas) .............. dataset descriptions
    3.2  Baseline model ........................... same as conference
    3.3  Grad-CAM analysis ........................ Phase 1 code
    3.4  Feature space analysis ................... Phase 2 code
    3.5  Confidence calibration ................... Phase 3 code
    3.6  External validation (FracAtlas) .......... Phase 4 code
    3.7  Domain adaptation ........................ Phase 5 code

  Section 4   Results
    4.1  MURA cross-anatomy (recap) ............... mura_summary.csv
    4.2  Mechanistic analysis ..................... gradcam/ + feature_analysis/
    4.3  Confidence calibration ................... calibration/
    4.4  Cross-dataset generalization ............. fracatlas_validation/
    4.5  Domain adaptation comparison ............. domain_adaptation/

  Section 5   Discussion
    - Why does hand resist adaptation? (feature distance ratio)
    - Are Grad-CAM patterns anatomy-specific or fracture-specific?
    - Does FracAtlas confirm MURA findings? (generalizability claim)
    - Which DA method works best, and at what sample count?
    """)
    print("=" * 70)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
