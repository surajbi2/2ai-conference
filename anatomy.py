# ===============================
# 0. ENVIRONMENT CHECK
# ===============================
import os
import torch


torch.backends.cudnn.benchmark = True


# ===============================
# 1. DATASET
# ===============================
import cv2
import numpy as np
from torch.utils.data import Dataset

class MURADataset(Dataset):
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
# 2. MODEL
# ===============================
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

def get_densenet121():
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


# ===============================
# 3. TRAIN / EVAL
# ===============================
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

def train_epoch(model, loader, optimizer):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for x, y in tqdm(loader, leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            preds.extend(probs.tolist())
            targets.extend(y.numpy().flatten().tolist())

    preds = np.array(preds)
    targets = np.array(targets)

    if len(np.unique(targets)) < 2:
        return float("nan"), accuracy_score(targets, preds > np.mean(preds))

    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, preds > 0.5)
    return auc, acc


# ===============================
# 4–7. MAIN EXECUTION (WINDOWS SAFE)
# ===============================
def main():
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE. Fix your GPU setup."

    DEVICE = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)


    BASE_PATH = r"C:\Users\Suraj\Documents\python\MURA-v1.1"
    TRAIN_ROOT = os.path.join(BASE_PATH, "train")
    VALID_ROOT = os.path.join(BASE_PATH, "valid")

    TRAIN_ANATOMY = "XR_WRIST"
    CROSS_ANATOMY = "XR_ELBOW"

    train_set = MURADataset(TRAIN_ROOT, TRAIN_ANATOMY)
    val_set   = MURADataset(VALID_ROOT, TRAIN_ANATOMY)
    cross_set = MURADataset(VALID_ROOT, CROSS_ANATOMY)

    train_loader = DataLoader(
        train_set, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    val_loader = DataLoader(
        val_set, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True
    )

    cross_loader = DataLoader(
        cross_set, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = get_densenet121().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer)
        auc, _ = evaluate(model, val_loader)
        print(f"Epoch {epoch+1:02d} | Loss {loss:.4f} | Wrist AUC {auc:.4f}")

    wrist_auc, _ = evaluate(model, val_loader)
    elbow_auc, _ = evaluate(model, cross_loader)

    print("\nFINAL RESULTS")
    print("WRIST AUC:", wrist_auc)
    print("ELBOW AUC:", elbow_auc)

    # ===== Feature Extraction + PCA =====
    class FeatureExtractor(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.features = model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            return torch.flatten(x, 1)

    def extract_features(feature_model, loader, label):
        feats, labs = [], []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(DEVICE)
                f = feature_model(x).detach().cpu().numpy()
                feats.append(f)
                labs.append(np.full(f.shape[0], label))
        return np.vstack(feats), np.concatenate(labs)

    feature_model = FeatureExtractor(model).to(DEVICE).eval()

    wrist_feat, wrist_lab = extract_features(feature_model, val_loader, 0)
    elbow_feat, elbow_lab = extract_features(feature_model, cross_loader, 1)

    X = np.vstack([wrist_feat, elbow_feat])
    Y = np.concatenate([wrist_lab, elbow_lab])

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(X2[Y == 0, 0], X2[Y == 0, 1], alpha=0.6, label="Wrist")
    plt.scatter(X2[Y == 1, 0], X2[Y == 1, 1], alpha=0.6, label="Elbow")
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Cross-Anatomy Feature Separation")
    plt.tight_layout()
    plt.show()


# ===============================
# ENTRY POINT (CRITICAL ON WINDOWS)
# ===============================
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
