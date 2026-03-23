# Quick Configuration Guide

## Common Modifications

### 1. Change Source/Target Anatomies

**Location:** Line 733-737

```python
# Current
SOURCE_ANATOMY = "XR_WRIST"
TARGET_ANATOMIES = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER", "XR_FINGER"]

# To change (example: train on ELBOW, test on others)
SOURCE_ANATOMY = "XR_ELBOW"
TARGET_ANATOMIES = ["XR_WRIST", "XR_HAND", "XR_SHOULDER", "XR_FINGER"]
```

### 2. Change Training Epochs

**Location:** Line 777

```python
# Current (fast training)
EPOCHS = 10

# For better performance
EPOCHS = 20

# For quick testing
EPOCHS = 3
```

### 3. Adjust Batch Size (If OOM)

**Location:** Lines 763-770

```python
# Current
train_loader = DataLoader(train_set, batch_size=16, ...)

# If GPU memory issues
train_loader = DataLoader(train_set, batch_size=8, ...)

# If you have more GPU memory
train_loader = DataLoader(train_set, batch_size=32, ...)
```

### 4. Change Sample Sizes for Efficiency Curve

**Location:** Line 603

```python
# Current
sample_sizes = [10, 25, 50, 100, 200, 500]

# Faster (fewer points)
sample_sizes = [10, 50, 100, 200]

# More detailed
sample_sizes = [5, 10, 25, 50, 100, 150, 200, 300, 500, 1000]
```

### 5. Modify Layer-wise Freeze Depths

**Location:** Line 500

```python
# Current (5 points)
layer_configs = [0, total_params // 4, total_params // 2, 
                3 * total_params // 4, total_params]

# More detailed (9 points)
layer_configs = [int(total_params * i / 8) for i in range(9)]

# Faster (3 points)
layer_configs = [0, total_params // 2, total_params]
```

### 6. Change Image Size

**Location:** Line 51-52

```python
# Current
image = cv2.resize(image, (224, 224))

# Higher resolution (slower, may improve performance)
image = cv2.resize(image, (512, 512))

# Lower resolution (faster)
image = cv2.resize(image, (128, 128))

# IMPORTANT: Also update line 54 for correct dimensions
```

### 7. Use Different Model

**Location:** Lines 69-72

```python
# Current: DenseNet121
from torchvision.models import densenet121, DenseNet121_Weights

def get_densenet121():
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model

# To use ResNet50
from torchvision.models import resnet50, ResNet50_Weights

def get_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# To use EfficientNet
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model
```

### 8. Skip Phases (For Faster Testing)

Comment out phases you don't need:

```python
# Skip layerwise analysis (saves ~15 min)
# Comment out lines 817-837

# Skip sample efficiency (saves ~10 min)  
# Comment out lines 853-882

# Skip feature extraction (saves ~10 min)
# Comment out lines 809-815
```

### 9. Change Output Directory Name

**Location:** Lines 741-743

```python
# Current (timestamped)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results_{timestamp}"

# Custom name
output_dir = "experiment_wrist_transfer"

# Include config in name
output_dir = f"results_{SOURCE_ANATOMY}_to_targets_{EPOCHS}epochs"
```

### 10. Adjust Learning Rate

**Location:** Multiple places (search for `lr=`)

```python
# Current
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Faster convergence (may overfit)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# More stable (slower)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
```

---

## Quick Testing Configuration

For **fast testing** (get results in 15 minutes):

```python
EPOCHS = 3                                    # Line 777
sample_sizes = [10, 50, 100]                  # Line 603
layer_configs = [0, total_params // 2, total_params]  # Line 500

# Comment out sample efficiency section (lines 853-882)
```

---

## Publication-Ready Configuration

For **best results** (takes ~90 minutes):

```python
EPOCHS = 20                                   # Line 777
sample_sizes = [10, 25, 50, 100, 200, 500, 1000]  # Line 603
batch_size = 32 (if GPU allows)               # Lines 763, 769

# Keep all phases enabled
```

---

## Debugging Configuration

If code crashes, try:

```python
EPOCHS = 1
batch_size = 4
num_workers = 0  # Disable multiprocessing
sample_sizes = [10, 50]
TARGET_ANATOMIES = ["XR_ELBOW"]  # Test with just one target

# Add this after model creation to check GPU usage:
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

---

## Adding Custom Metrics

To add your own evaluation metrics, modify the `evaluate_detailed` function (lines 118-157):

```python
# Example: Add F1 score
from sklearn.metrics import f1_score

def evaluate_detailed(model, loader, device):
    # ... existing code ...
    
    # Add after line 147
    f1 = f1_score(targets, preds > 0.5)
    
    return {
        'auc': auc,
        'acc': acc,
        'f1': f1,  # NEW
        'preds': preds,
        'targets': targets,
        'confidences': confidences
    }
```

---

## System Requirements

### Minimum
- GPU: 8GB VRAM (GTX 1080, RTX 2070)
- RAM: 16GB
- Storage: 50GB for MURA dataset

### Recommended  
- GPU: 12GB+ VRAM (RTX 3080, 4070)
- RAM: 32GB
- Storage: SSD for faster loading

### CPU-Only Mode (Not Recommended)
Change line 727:
```python
# Current
device = torch.device("cuda")

# CPU mode (will take 10-20× longer)
device = torch.device("cpu")
```

---

## Parallelization Options

To speed up on multi-GPU systems:

```python
# After model creation (line 778)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

---

## Data Augmentation (Optional)

To add augmentation (may improve performance):

```python
# Add to __getitem__ in MURADataset (line 58)
import torchvision.transforms as T

def __getitem__(self, idx):
    # ... existing code up to line 58 ...
    
    # Add augmentation
    if hasattr(self, 'augment') and self.augment:
        image = torch.tensor(image, dtype=torch.float32)
        augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.RandomAffine(0, translate=(0.1, 0.1))
        ])
        image = augment(image)
    else:
        image = torch.tensor(image, dtype=torch.float32)
    
    return (image, torch.tensor([label], dtype=torch.float32))
```

---

## Common Errors & Fixes

### Error: "CUDA out of memory"
**Fix:** Reduce batch_size to 8 or 4

### Error: "FileNotFoundError: XR_FINGER"  
**Fix:** Normal - code automatically skips missing anatomies

### Error: "RuntimeError: DataLoader worker"
**Fix:** Set `num_workers=0` in all DataLoader calls

### Error: "NaN in evaluation"
**Fix:** Check if dataset has both fracture and non-fracture samples

---

## Performance Expectations

### Expected Runtimes (RTX 3090)
- Phase 1 (Train): ~2 min/epoch × 10 = 20 min
- Phase 2 (Transfer): ~5 min  
- Phase 3 (Features): ~10 min
- Phase 4 (Layerwise): ~15 min
- Phase 5 (Clinical): ~10 min
**Total: ~60 min**

### Expected Performance
- Source AUC: 0.80-0.90 (WRIST)
- Target AUC: 0.60-0.75 (average)
- Performance drop: 10-25%
- Distance ratio: 2.5-4.0

If your drops are < 5%, the contribution is weak.
If your drops are > 30%, something is broken - check data.

---

## Before You Submit Your Paper

Run this checklist:

- [ ] Tested on all 4 target anatomies
- [ ] Results saved in timestamped folder
- [ ] All CSV files generated
- [ ] All PNG plots generated  
- [ ] Performance drop > 10%
- [ ] Distance ratio > 2.5
- [ ] False negative analysis shows increase
- [ ] Sample efficiency curve is monotonic
- [ ] README.md interpretation matches your data

**If all checked → You're ready to write the paper.**
