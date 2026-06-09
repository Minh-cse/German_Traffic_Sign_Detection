import os
import cv2
import yaml
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import kagglehub

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(os.getcwd())
DATASET_DIR = BASE_DIR / "datasets"
IMGSZ = 640
SEED = 42

DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# REPRODUCIBILITY
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

# ==========================================
# DOWNLOAD DATA
# ==========================================
def download_data(force_download=False):
    """Download dataset from Kaggle"""
    print("--- Downloading Dataset ---")
    
    try:
        path = kagglehub.dataset_download(
            "aakcodebreaker/german-traffic-sign-recognition-benchmark"
        )
        
        print(f"Dataset downloaded to: {path}")
        
        # Verify dataset structure
        raw_path = Path(path)
        gtsrb_path = raw_path / "GTSRB"
        
        if not gtsrb_path.exists():
            if (raw_path / "Final_Training").exists():
                gtsrb_path = raw_path
            else:
                print(f"Contents of {raw_path}:")
                for item in raw_path.iterdir():
                    print(f"  - {item.name}")
                raise RuntimeError(
                    f"GTSRB folder not found at expected location. "
                    f"Expected: {gtsrb_path}"
                )
        
        # Verify we have the full dataset (43 classes)
        images_base = gtsrb_path / "Final_Training" / "Images"
        if images_base.exists():
            class_folders = [d for d in images_base.iterdir() if d.is_dir()]
            print(f"Found {len(class_folders)} class folders")
            if len(class_folders) < 43:
                print(f"WARNING: Expected 43 classes, found {len(class_folders)}")
        
        return gtsrb_path
        
    except Exception as e:
        raise RuntimeError(
            "Failed to download dataset. Ensure Kaggle API is configured.\n"
            f"Error: {e}"
        ) from e

# ==========================================
# DATASET PROCESSING - SCAN ALL FILES
# ==========================================
def process_dataset(raw_path: Path, dataset_dir: Path, imgsz: int, seed: int, force_rebuild=True):
    """
    Process dataset: resize images and convert annotations to YOLO format.
    
    Args:
        raw_path: Path to raw dataset directory
        dataset_dir: Path to output dataset directory
        imgsz: Image size for resizing (e.g., 640)
        seed: Random seed for reproducibility
        force_rebuild: Whether to delete and rebuild the dataset
    """
    print("\n--- Processing Dataset ---")

    if force_rebuild and dataset_dir.exists():
        print("Force rebuild enabled → deleting old dataset")
        shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

    output_images = dataset_dir / "images"
    output_labels = dataset_dir / "labels"
    for split in ["train", "val"]:
        (output_images / split).mkdir(parents=True, exist_ok=True)
        (output_labels / split).mkdir(parents=True, exist_ok=True)

    raw_path = Path(raw_path)
    
    if (raw_path / "GTSRB").exists():
        root_gt_dir = raw_path / "GTSRB"
    else:
        root_gt_dir = raw_path
    
    images_base_dir = root_gt_dir / "Final_Training" / "Images"
    
    if not images_base_dir.exists():
        raise RuntimeError(f"Images directory not found at: {images_base_dir}")

    print(f"Loading images from: {images_base_dir}")
    print("Scanning all class folders...")

    # Collect all image files and their labels from EVERY class folder
    all_images = []
    all_labels_data = []
    found_classes = 0
    class_stats = {}
    
    for class_id in range(43):
        class_folder = str(class_id).zfill(5)
        class_dir = images_base_dir / class_folder
        csv_path = class_dir / f"GT-{class_folder}.csv"
        
        if not class_dir.exists() or not csv_path.exists():
            print(f"  Class {class_id:2d} ({class_folder}): NOT FOUND")
            continue
        
        # Read CSV with annotations
        df = pd.read_csv(csv_path, sep=";")
        df = df.rename(columns={
            "Filename": "filename",
            "Width": "width",
            "Height": "height",
            "Roi.X1": "x1",
            "Roi.Y1": "y1",
            "Roi.X2": "x2",
            "Roi.Y2": "y2",
            "ClassId": "class_id",
        })
        df["class_folder"] = class_folder
        df["class_id_int"] = class_id
        
        found_classes += 1
        class_stats[class_id] = len(df)
        
        print(f"  Class {class_id:2d} ({class_folder}): {len(df):4d} images")
        all_labels_data.append(df)
        
        # Collect all image files in this class folder
        for img_file in class_dir.glob("*.ppm"):
            all_images.append({
                "path": img_file,
                "filename": img_file.name,
                "class_folder": class_folder,
                "class_id": class_id
            })

    print(f"\n✅ Found {found_classes}/43 classes")
    print(f"✅ Total image files found: {len(all_images)}")
    
    if found_classes == 0:
        raise RuntimeError("No class folders found. Dataset may not be downloaded correctly.")

    # Combine all label data
    if all_labels_data:
        labels_df = pd.concat(all_labels_data, ignore_index=True)
    else:
        raise RuntimeError("No labels found in any class folder")
    
    print(f"✅ Total annotations: {len(labels_df)}")

    # Split into train/val
    unique_files = labels_df["filename"].unique()
    train_files_set, val_files_set = train_test_split(
        unique_files, test_size=0.2, random_state=seed
    )
    
    train_files_set = set(train_files_set)
    val_files_set = set(val_files_set)

    print(f"✅ Train images: {len(train_files_set)}, Val images: {len(val_files_set)}")
    print("\nConverting all images & labels → YOLO format...")
    
    skipped = 0
    processed = 0
    global_idx = 0
    class_processed = {i: 0 for i in range(43)}
    
    # Process ALL images found in directory
    for img_info in tqdm(all_images, desc="Processing images"):
        src_img = img_info["path"]
        filename = img_info["filename"]
        class_folder = img_info["class_folder"]
        class_id = img_info["class_id"]
        
        # Determine train/val split
        split = "train" if filename in train_files_set else "val"
        
        # Get annotations for this image
        matching_rows = labels_df[
            (labels_df["filename"] == filename) & 
            (labels_df["class_folder"] == class_folder)
        ]
        
        if matching_rows.empty:
            skipped += 1
            continue

        # Read image
        img = cv2.imread(str(src_img))
        if img is None:
            skipped += 1
            continue

        # Resize image
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (imgsz, imgsz))

        # Create unique filename with class prefix
        unique_name = f"{class_folder}_{global_idx:06d}"
        output_img_path = output_images / split / f"{unique_name}.jpg"
        cv2.imwrite(str(output_img_path), img_resized)

        # Write label file
        label_path = output_labels / split / f"{unique_name}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            for _, row in matching_rows.iterrows():
                # Scale bounding box to resized image
                sx = imgsz / row["width"]
                sy = imgsz / row["height"]
                x1 = max(0, min(imgsz - 1, row["x1"] * sx))
                y1 = max(0, min(imgsz - 1, row["y1"] * sy))
                x2 = max(0, min(imgsz - 1, row["x2"] * sx))
                y2 = max(0, min(imgsz - 1, row["y2"] * sy))
                
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                bw = x2 - x1
                bh = y2 - y1

                # Skip tiny bboxes
                if bw < 2 or bh < 2:
                    continue

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Write in YOLO format
                f.write(
                    f"{int(row['class_id_int'])} "
                    f"{cx/imgsz:.6f} {cy/imgsz:.6f} "
                    f"{bw/imgsz:.6f} {bh/imgsz:.6f}\n"
                )
        
        processed += 1
        class_processed[class_id] += 1
        global_idx += 1

    print("\n" + "="*60)
    print("CLASS PROCESSING SUMMARY")
    print("="*60)
    for class_id in range(43):
        class_folder = str(class_id).zfill(5)
        count = class_processed[class_id]
        expected = class_stats.get(class_id, 0)
        status = "✅" if count > 0 else "⚠️"
        print(f"{status} Class {class_id:2d} ({class_folder}): {count:4d} processed (expected {expected})")

    print("\n" + "="*60)
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} corrupted/missing images")
    
    print(f"✅ Successfully processed {processed} total images")
    print(f"Dataset location: {dataset_dir}")
    print("="*60)

# ==========================================
# YAML CONFIG
# ==========================================
def create_yaml():
    yaml_path = DATASET_DIR / "data.yaml"
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 43,
        "names": [str(i) for i in range(43)],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    print(f"✅ Created: {yaml_path}")
    return yaml_path

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    set_seed(SEED)

    raw_path = download_data(force_download=False)
    process_dataset(raw_path, DATASET_DIR, IMGSZ, SEED, force_rebuild=True)
    create_yaml()
    
    print("\n✅ Dataset download and processing complete!")
    print(f"Check {DATASET_DIR} for processed images and labels")