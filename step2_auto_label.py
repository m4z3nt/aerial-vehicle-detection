"""
STEP 2: Auto-label frames using pretrained YOLOv8
This is the HEART of the exercise — auto-labeling pipeline.

Strategy:
1. Use pretrained YOLOv8x (largest, most accurate) for pseudo-labels
2. Filter only vehicle classes: car, truck, bus, motorcycle
3. Merge all into single class "vehicle" (class 0)
4. Save in YOLO format: class x_center y_center width height (normalized)
5. Light manual cleanup described in README

Run: python step2_auto_label.py
"""

import os
import cv2
import glob
import shutil
from ultralytics import YOLO

# ============================================
# SETTINGS
# ============================================

# Use YOLOv8x for highest quality pseudo-labels
# (we train with yolov8n later — smaller model learns from bigger model's labels)
LABELER_MODEL = "yolov8x.pt"

# COCO classes that count as "vehicle"
# 2=car, 5=bus, 7=truck, 3=motorcycle
VEHICLE_CLASSES = {2, 3, 5, 7}

# Minimum confidence for auto-labels
CONFIDENCE_THRESHOLD = 0.25  # Low threshold — catch more, clean later

# Minimum bbox area (fraction of image) — filter tiny noise
MIN_BBOX_AREA = 0.0005  # 0.05% of image area

# Maximum bbox area — filter huge false positives
MAX_BBOX_AREA = 0.15  # 15% of image area

# NMS IoU threshold
NMS_IOU = 0.5

FRAMES_DIR = "data/frames"
LABELS_DIR = "data/labels_auto"
TRAIN_CLIPS = ["train_A", "train_B", "train_C", "train_D"]
EVAL_CLIP = "eval"


def auto_label_frames(model, frames_dir, labels_dir, clip_name):
    """
    Run pretrained YOLO on frames and save pseudo-labels.
    
    Args:
        model: loaded YOLO model
        frames_dir: directory with frames
        labels_dir: directory to save labels
        clip_name: name of the clip (for logging)
    
    Returns:
        dict with statistics
    """
    input_dir = os.path.join(frames_dir, clip_name)
    output_dir = os.path.join(labels_dir, clip_name)
    os.makedirs(output_dir, exist_ok=True)
    
    frames = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    if not frames:
        print(f"  No frames found in {input_dir}")
        return {"frames": 0, "detections": 0}
    
    stats = {
        "frames": len(frames),
        "frames_with_vehicles": 0,
        "total_detections": 0,
        "filtered_small": 0,
        "filtered_large": 0,
        "filtered_low_conf": 0,
    }
    
    print(f"  Processing {clip_name}: {len(frames)} frames...")
    
    for i, frame_path in enumerate(frames):
        # Run inference
        results = model(frame_path, conf=0.1, iou=NMS_IOU, verbose=False)
        
        # Get image dimensions for normalization check
        img = cv2.imread(frame_path)
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w
        
        # Filter and convert detections
        labels = []
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Only vehicle classes
                if cls_id not in VEHICLE_CLASSES:
                    continue
                
                # Confidence filter
                if conf < CONFIDENCE_THRESHOLD:
                    stats["filtered_low_conf"] += 1
                    continue
                
                # Get normalized coordinates (YOLO format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Convert to YOLO format: x_center, y_center, width, height (normalized)
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                bbox_w = (x2 - x1) / img_w
                bbox_h = (y2 - y1) / img_h
                
                # Area filter
                bbox_area = bbox_w * bbox_h
                if bbox_area < MIN_BBOX_AREA:
                    stats["filtered_small"] += 1
                    continue
                if bbox_area > MAX_BBOX_AREA:
                    stats["filtered_large"] += 1
                    continue
                
                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                bbox_w = max(0, min(1, bbox_w))
                bbox_h = max(0, min(1, bbox_h))
                
                # Class 0 = vehicle (single class)
                labels.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")
                stats["total_detections"] += 1
        
        # Save label file
        label_filename = os.path.splitext(os.path.basename(frame_path))[0] + ".txt"
        label_path = os.path.join(output_dir, label_filename)
        
        with open(label_path, "w") as f:
            f.write("\n".join(labels))
        
        if labels:
            stats["frames_with_vehicles"] += 1
        
        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(frames):
            print(f"    {i+1}/{len(frames)} frames processed, "
                  f"{stats['total_detections']} detections so far")
    
    return stats


def create_yolo_dataset(frames_dir, labels_dir, dataset_dir, train_clips, val_split=0.1):
    """
    Create YOLO dataset structure from auto-labeled frames.
    
    Train clips → dataset/images/train + dataset/labels/train
    Last 10% of each train clip → dataset/images/val + dataset/labels/val
    """
    print("  Creating YOLO dataset structure...")
    
    train_count = 0
    val_count = 0
    
    for clip_name in train_clips:
        clip_frames_dir = os.path.join(frames_dir, clip_name)
        clip_labels_dir = os.path.join(labels_dir, clip_name)
        
        frames = sorted(glob.glob(os.path.join(clip_frames_dir, "*.jpg")))
        
        if not frames:
            continue
        
        # Split: last 10% for validation
        split_idx = int(len(frames) * (1 - val_split))
        train_frames = frames[:split_idx]
        val_frames = frames[split_idx:]
        
        # Copy train frames and labels
        for frame_path in train_frames:
            basename = os.path.splitext(os.path.basename(frame_path))[0]
            unique_name = f"{clip_name}_{basename}"
            
            # Copy frame
            dst_frame = os.path.join(dataset_dir, "images/train", f"{unique_name}.jpg")
            shutil.copy2(frame_path, dst_frame)
            
            # Copy label
            src_label = os.path.join(clip_labels_dir, f"{basename}.txt")
            dst_label = os.path.join(dataset_dir, "labels/train", f"{unique_name}.txt")
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                # Empty label file (no vehicles)
                open(dst_label, 'w').close()
            
            train_count += 1
        
        # Copy val frames and labels
        for frame_path in val_frames:
            basename = os.path.splitext(os.path.basename(frame_path))[0]
            unique_name = f"{clip_name}_{basename}"
            
            dst_frame = os.path.join(dataset_dir, "images/val", f"{unique_name}.jpg")
            shutil.copy2(frame_path, dst_frame)
            
            src_label = os.path.join(clip_labels_dir, f"{basename}.txt")
            dst_label = os.path.join(dataset_dir, "labels/val", f"{unique_name}.txt")
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                open(dst_label, 'w').close()
            
            val_count += 1
    
    print(f"  Train images: {train_count}")
    print(f"  Val images:   {val_count}")
    
    return train_count, val_count


def create_data_yaml(dataset_dir):
    """Create YOLO data.yaml configuration"""
    abs_path = os.path.abspath(dataset_dir)
    
    yaml_content = f"""# Aerial Vehicle Detection Dataset
# Auto-labeled using YOLOv8x, single class: vehicle
# TFL ML Engineer Test Task

path: {abs_path}
train: images/train
val: images/val

nc: 1
names:
  0: vehicle
"""
    
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"  Created: {yaml_path}")
    return yaml_path


def main():
    print("=" * 55)
    print("  STEP 2: AUTO-LABELING")
    print("  Using pretrained YOLOv8x for pseudo-labels")
    print("=" * 55)
    print()
    
    # Load pretrained model
    print("[1/4] Loading pretrained model...")
    print(f"  Model: {LABELER_MODEL}")
    print(f"  Vehicle classes: car(2), motorcycle(3), bus(5), truck(7)")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Min bbox area: {MIN_BBOX_AREA}")
    print(f"  Max bbox area: {MAX_BBOX_AREA}")
    model = YOLO(LABELER_MODEL)
    print("  Model loaded!")
    print()
    
    # Auto-label TRAIN clips only
    print("[2/4] Auto-labeling TRAIN clips...")
    print("  (eval clip is NOT touched — held out)")
    print()
    
    all_stats = {}
    for clip_name in TRAIN_CLIPS:
        stats = auto_label_frames(model, FRAMES_DIR, LABELS_DIR, clip_name)
        all_stats[clip_name] = stats
        print(f"  {clip_name}: {stats['total_detections']} detections "
              f"in {stats['frames_with_vehicles']}/{stats['frames']} frames")
        print(f"    Filtered: {stats['filtered_small']} small, "
              f"{stats['filtered_large']} large, "
              f"{stats['filtered_low_conf']} low-conf")
        print()
    
    # Auto-label EVAL clip separately (for ground truth comparison only)
    print("  Auto-labeling EVAL clip (for GT comparison only)...")
    eval_stats = auto_label_frames(model, FRAMES_DIR, LABELS_DIR, EVAL_CLIP)
    all_stats[EVAL_CLIP] = eval_stats
    print(f"  eval: {eval_stats['total_detections']} detections")
    print()
    
    # Create YOLO dataset
    print("[3/4] Creating YOLO dataset...")
    dataset_dir = "data/dataset"
    train_count, val_count = create_yolo_dataset(
        FRAMES_DIR, LABELS_DIR, dataset_dir, TRAIN_CLIPS
    )
    print()
    
    # Create data.yaml
    print("[4/4] Creating data.yaml...")
    yaml_path = create_data_yaml(dataset_dir)
    print()
    
    # Summary
    print("=" * 55)
    print("  AUTO-LABELING COMPLETE")
    print("=" * 55)
    total_det = sum(s['total_detections'] for s in all_stats.values())
    total_frames = sum(s['frames'] for s in all_stats.values())
    print(f"  Total frames:     {total_frames}")
    print(f"  Total detections: {total_det}")
    print(f"  Train images:     {train_count}")
    print(f"  Val images:       {val_count}")
    print(f"  Dataset:          {dataset_dir}/")
    print(f"  Config:           {yaml_path}")
    print()
    print("  💡 MANUAL CLEANUP (recommended):")
    print("     Open data/labels_auto/ and spot-check labels.")
    print("     Remove obvious false positives (trees, shadows).")
    print("     Add missed vehicles if clearly visible.")
    print("     Copy cleaned labels to data/labels_clean/.")
    print("     Then re-run dataset creation.")
    print()
    print("  Next step: python step3_train.py")
    print()


if __name__ == "__main__":
    main()
