import os
from collections import defaultdict

BASE_PATH = r"C:\MURA-v1.1"

SPLITS = ["train", "test"]

def analyze_split(split_path):
    stats = {}

    for anatomy in os.listdir(split_path):
        anatomy_path = os.path.join(split_path, anatomy)
        if not os.path.isdir(anatomy_path):
            continue

        study_count = 0
        image_count = 0
        positive_studies = 0
        negative_studies = 0
        positive_images = 0
        negative_images = 0

        for patient in os.listdir(anatomy_path):
            patient_path = os.path.join(anatomy_path, patient)
            if not os.path.isdir(patient_path):
                continue

            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue

                study_count += 1

                is_positive = "positive" in study.lower()
                is_negative = "negative" in study.lower()

                if is_positive:
                    positive_studies += 1
                elif is_negative:
                    negative_studies += 1
                else:
                    continue

                for img in os.listdir(study_path):
                    if img.endswith(".png"):
                        image_count += 1
                        if is_positive:
                            positive_images += 1
                        else:
                            negative_images += 1

        stats[anatomy] = {
            "studies": study_count,
            "images": image_count,
            "positive_studies": positive_studies,
            "negative_studies": negative_studies,
            "positive_images": positive_images,
            "negative_images": negative_images
        }

    return stats


def print_stats(split_name, stats):
    print("\n" + "="*70)
    print(f"{split_name.upper()} DATASET STATISTICS")
    print("="*70)

    print(f"{'Anatomy':15} {'Studies':10} {'Images':10} "
          f"{'PosStud':10} {'NegStud':10} "
          f"{'PosImg':10} {'NegImg':10}")
    print("-"*70)

    for anatomy, s in sorted(stats.items()):
        print(f"{anatomy:15} "
              f"{s['studies']:10} "
              f"{s['images']:10} "
              f"{s['positive_studies']:10} "
              f"{s['negative_studies']:10} "
              f"{s['positive_images']:10} "
              f"{s['negative_images']:10}")

    print("="*70)


if __name__ == "__main__":
    for split in SPLITS:
        split_path = os.path.join(BASE_PATH, split)

        if not os.path.exists(split_path):
            print(f"Missing split: {split_path}")
            continue

        stats = analyze_split(split_path)
        print_stats(split, stats)