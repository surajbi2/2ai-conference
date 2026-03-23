"""
MURA Dataset Verification Script
Run this before your main experiments to verify you have enough data
"""

import os
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding="utf-8")


def verify_mura_dataset(base_path):
    """
    Verify MURA dataset structure and sample counts
    Returns: True if dataset is ready for experiments
    """
    
    train_root = os.path.join(base_path, "train")
    valid_root = os.path.join(base_path, "valid")
    
    print("="*70)
    print("MURA DATASET VERIFICATION")
    print("="*70)
    print(f"Base path: {base_path}\n")
    
    # Check paths exist
    if not os.path.exists(train_root):
        print(f"ERROR: Train directory not found: {train_root}")
        return False
    
    if not os.path.exists(valid_root):
        print(f"ERROR: Valid directory not found: {valid_root}")
        return False
    
    print("Train and valid directories found\n")
    
    # Define anatomies to check
    anatomies = ["XR_WRIST", "XR_ELBOW", "XR_HAND", "XR_SHOULDER", 
                 "XR_FINGER", "XR_FOREARM", "XR_HUMERUS"]
    
    # Count samples
    train_counts = {}
    valid_counts = {}
    
    for anatomy in anatomies:
        train_path = os.path.join(train_root, anatomy)
        valid_path = os.path.join(valid_root, anatomy)
        
        # Count train
        if os.path.exists(train_path):
            train_counts[anatomy] = count_images_recursive(train_path)
        else:
            train_counts[anatomy] = 0
        
        # Count valid
        if os.path.exists(valid_path):
            valid_counts[anatomy] = count_images_recursive(valid_path)
        else:
            valid_counts[anatomy] = 0
    
    # Display results
    print("ANATOMY SAMPLE COUNTS:")
    print("-" * 70)
    print(f"{'Anatomy':<15} {'Train':<10} {'Valid':<10} {'Status':<20}")
    print("-" * 70)
    
    source_ready = False
    target_count = 0
    
    for anatomy in anatomies:
        train_n = train_counts[anatomy]
        valid_n = valid_counts[anatomy]
        
        # Determine status
        if train_n == 0 and valid_n == 0:
            status = "Missing"
        elif train_n >= 1000 and valid_n >= 200:
            status = "Good (Source ready)"
            source_ready = True
        elif valid_n >= 200:
            status = "Good (Target ready)"
            target_count += 1
        elif valid_n > 0:
            status = " Low samples"
        else:
            status = "No valid data"
        
        print(f"{anatomy:<15} {train_n:<10} {valid_n:<10} {status:<20}")
    
    print("-" * 70)
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 70)
    
    if not source_ready:
        print("NO SUITABLE SOURCE ANATOMY FOUND")
        print("   You need at least one anatomy with:")
        print("   - Train samples: 1000+")
        print("   - Valid samples: 200+")
        print()
        return False
    
    if target_count < 3:
        print(f" ONLY {target_count} TARGET ANATOMIES AVAILABLE")
        print("   For a strong paper, you need 3-4 target anatomies.")
        print("   Each should have 200+ validation samples.")
        print()
    else:
        print(f"{target_count} TARGET ANATOMIES AVAILABLE")
        print("   This is sufficient for cross-anatomy experiments.")
        print()
    
    # Recommended configuration
    print("RECOMMENDED EXPERIMENT CONFIGURATION:")
    print("-" * 70)
    
    # Find best source (most training data)
    best_source = max(train_counts.items(), key=lambda x: x[1])
    source_anatomy = best_source[0]
    
    print(f"SOURCE ANATOMY: {source_anatomy}")
    print(f"  Train samples: {train_counts[source_anatomy]}")
    print(f"  Valid samples: {valid_counts[source_anatomy]}")
    print()
    
    print("TARGET ANATOMIES:")
    target_anatomies = [a for a in anatomies 
                       if a != source_anatomy and valid_counts[a] >= 200]
    
    for anat in target_anatomies:
        print(f"  {anat}: {valid_counts[anat]} samples")
    
    print()
    
    # Code configuration
    print("CODE CONFIGURATION:")
    print("-" * 70)
    print(f'SOURCE_ANATOMY = "{source_anatomy}"')
    print(f'TARGET_ANATOMIES = {target_anatomies}')
    print()
    
    # Warnings
    print("WARNINGS:")
    print("-" * 70)
    
    # Check class balance
    for anatomy in [source_anatomy] + target_anatomies:
        pos, neg = count_positive_negative(
            os.path.join(valid_root, anatomy)
        )
        total = pos + neg
        if total > 0:
            balance = min(pos, neg) / max(pos, neg) * 100
            if balance < 30:
                print(f" {anatomy}: Imbalanced dataset")
                print(f"   Positive: {pos} ({pos/total*100:.1f}%)")
                print(f"   Negative: {neg} ({neg/total*100:.1f}%)")
    
    print("="*70)
    
    return True


def count_images_recursive(directory):
    """Count all PNG images in directory recursively"""
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len([f for f in files if f.endswith('.png')])
    return count


def count_positive_negative(directory):
    """Count positive and negative samples"""
    pos_count = 0
    neg_count = 0
    
    for root, dirs, files in os.walk(directory):
        images = len([f for f in files if f.endswith('.png')])
        
        if 'positive' in root.lower():
            pos_count += images
        elif 'negative' in root.lower():
            neg_count += images
    
    return pos_count, neg_count


if __name__ == "__main__":
    # UPDATE THIS PATH
    MURA_PATH = r"C:\MURA\MURA\muramskxrays\MURA-v1.1\MURA-v1.1"
    
    is_ready = verify_mura_dataset(MURA_PATH)
    
    print()
    if is_ready:
        print("DATASET READY FOR EXPERIMENTS")
        print("   You can proceed with the analysis pipeline.")
    else:
        print("DATASET NOT READY")
        print("   Fix the issues above before running experiments.")
