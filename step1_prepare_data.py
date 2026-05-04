"""
STEP 1: Download videos from Pexels and extract frames
Run: python step1_prepare_data.py
"""

import os
import cv2
import requests
import sys

# ============================================
# PROJECT STRUCTURE
# ============================================
# aerial-vehicle-detection/
# ├── data/
# │   ├── videos/          ← downloaded videos
# │   ├── frames/          ← extracted frames
# │   │   ├── train_A/
# │   │   ├── train_B/
# │   │   ├── train_C/
# │   │   ├── train_D/
# │   │   └── eval/
# │   ├── labels_auto/     ← auto-generated labels
# │   ├── labels_clean/    ← after manual cleanup
# │   └── dataset/         ← final YOLO dataset
# │       ├── images/
# │       │   ├── train/
# │       │   └── val/
# │       ├── labels/
# │       │   ├── train/
# │       │   └── val/
# │       └── data.yaml
# ├── runs/                ← training results
# ├── step1_prepare_data.py
# ├── step2_auto_label.py
# ├── step3_train.py
# ├── step4_evaluate.py
# ├── step5_visualize.py
# └── README.md

# ============================================
# VIDEO URLS (Pexels)
# ============================================
# Pexels doesn't allow direct download via URL.
# You need to download manually from the links below
# or use Pexels API with your API key.

VIDEOS = {
    "train_A": {
        "url": "https://www.pexels.com/video/8968356",
        "desc": "highway interchange"
    },
    "train_B": {
        "url": "https://www.pexels.com/video/5382494",
        "desc": "rural highway, sparse traffic"
    },
    "train_C": {
        "url": "https://www.pexels.com/video/8457857",
        "desc": "simple highway, top-down"
    },
    "train_D": {
        "url": "https://www.pexels.com/video/3405804",
        "desc": "urban intersection"
    },
    "eval": {
        "url": "https://www.pexels.com/video/32179597",
        "desc": "city highway, daytime (HELD OUT - DO NOT USE FOR TRAINING)"
    }
}

# ============================================
# SETTINGS
# ============================================
FRAME_INTERVAL = 15  # Extract every Nth frame (for ~2 FPS from 30 FPS video)
OUTPUT_BASE = "data/frames"
VIDEO_DIR = "data/videos"


def create_dirs():
    """Create project directory structure"""
    dirs = [
        "data/videos",
        "data/frames/train_A",
        "data/frames/train_B",
        "data/frames/train_C",
        "data/frames/train_D",
        "data/frames/eval",
        "data/labels_auto",
        "data/labels_clean",
        "data/dataset/images/train",
        "data/dataset/images/val",
        "data/dataset/labels/train",
        "data/dataset/labels/val",
        "runs",
        "output",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  Created: {d}/")


def extract_frames(video_path, output_dir, interval=FRAME_INTERVAL):
    """
    Extract frames from video at given interval.
    
    Args:
        video_path: path to video file
        output_dir: where to save frames
        interval: extract every Nth frame
    
    Returns:
        number of frames extracted
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0
    
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  FPS: {fps:.1f}, Frames: {total}, Duration: {duration:.1f}s")
    print(f"  Extracting every {interval}th frame (~{fps/interval:.1f} FPS effective)")
    
    count = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % interval == 0:
            filename = f"frame_{frame_idx:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            count += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"  Extracted: {count} frames → {output_dir}/")
    return count


def main():
    print("=" * 55)
    print("  STEP 1: PREPARE DATA")
    print("  Aerial Vehicle Detection — TFL Test Task")
    print("=" * 55)
    print()
    
    # Create directories
    print("[1/3] Creating project structure...")
    create_dirs()
    print()
    
    # Check if videos exist
    print("[2/3] Checking videos...")
    print()
    print("  Download videos manually from Pexels:")
    print("  Save them to data/videos/ with these names:")
    print()
    
    all_exist = True
    for name, info in VIDEOS.items():
        filename = f"{name}.mp4"
        filepath = os.path.join(VIDEO_DIR, filename)
        exists = os.path.exists(filepath)
        status = "✅" if exists else "❌ MISSING"
        print(f"  {status}  {filename:20s} ← {info['url']}")
        print(f"          ({info['desc']})")
        if not exists:
            all_exist = False
    
    print()
    
    if not all_exist:
        print("  ⚠️  Download missing videos from Pexels and save as:")
        print(f"     {VIDEO_DIR}/train_A.mp4")
        print(f"     {VIDEO_DIR}/train_B.mp4")
        print(f"     {VIDEO_DIR}/train_C.mp4")
        print(f"     {VIDEO_DIR}/train_D.mp4")
        print(f"     {VIDEO_DIR}/eval.mp4")
        print()
        print("  Then run this script again.")
        print()
        
        # Still create directory structure
        response = input("  Continue with available videos? (y/n): ").strip().lower()
        if response != 'y':
            return
    
    # Extract frames
    print("[3/3] Extracting frames...")
    print()
    
    total_frames = 0
    for name in VIDEOS:
        filepath = os.path.join(VIDEO_DIR, f"{name}.mp4")
        outdir = os.path.join(OUTPUT_BASE, name)
        
        if os.path.exists(filepath):
            n = extract_frames(filepath, outdir)
            total_frames += n
            print()
        else:
            print(f"  Skipping {name} (video not found)")
            print()
    
    print(f"  Total frames extracted: {total_frames}")
    print()
    print("  Next step: python step2_auto_label.py")
    print()


if __name__ == "__main__":
    main()
