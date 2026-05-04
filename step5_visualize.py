"""
STEP 5: Visualize detections on eval clip
- Draw predictions (green) and GT (red) boxes
- Save example frames and output video
- Show distance band coloring

Run: python step5_visualize.py
"""

import os
import cv2
import glob
import math
from ultralytics import YOLO

# ============================================
# SETTINGS
# ============================================
TRAINED_MODEL = "runs/aerial_vehicle_v1/weights/best.pt"
EVAL_VIDEO = "data/videos/eval.mp4"
EVAL_FRAMES_DIR = "data/frames/eval"
EVAL_LABELS_DIR = "data/labels_auto/eval"
OUTPUT_DIR = "output"
CONF_THRESHOLD = 0.3

# Distance estimation
REFERENCE_CAR_LENGTH_M = 4.5
CAMERA_FOV_DEG = 70.0

# Colors (BGR)
GREEN = (0, 255, 0)     # Predictions
RED = (0, 0, 255)       # Ground truth
CYAN = (255, 255, 0)    # Near band (0-200m)
ORANGE = (0, 165, 255)  # Far band (200-400m)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def estimate_distance(bbox_w_px, img_w_px):
    fov_rad = math.radians(CAMERA_FOV_DEG)
    focal_px = img_w_px / (2 * math.tan(fov_rad / 2))
    return (REFERENCE_CAR_LENGTH_M * focal_px) / bbox_w_px if bbox_w_px > 0 else 9999


def load_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            xc = float(parts[1]) * img_w
            yc = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            boxes.append([int(xc - w/2), int(yc - h/2), int(xc + w/2), int(yc + h/2)])
    return boxes


def draw_frame(frame, pred_boxes, gt_boxes, frame_idx):
    """Draw predictions and GT on a frame"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Draw GT boxes (red, dashed-like with thin line)
    for box in gt_boxes:
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), RED, 1)
        cv2.putText(overlay, "GT", (box[0], box[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, RED, 1)
    
    # Draw predictions (green with distance)
    for box in pred_boxes:
        bw = box[2] - box[0]
        dist = estimate_distance(bw, w)
        
        color = CYAN if dist <= 200 else (ORANGE if dist <= 400 else GREEN)
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        label = f"{dist:.0f}m"
        cv2.putText(overlay, label, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # HUD
    cv2.rectangle(overlay, (0, 0), (w, 30), BLACK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)
    
    cv2.putText(overlay, f"Frame: {frame_idx}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(overlay, f"Pred: {len(pred_boxes)}  GT: {len(gt_boxes)}", (200, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(overlay, "GREEN=pred  RED=GT  CYAN=<200m  ORANGE=200-400m", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
    
    return overlay


def main():
    print("=" * 55)
    print("  STEP 5: VISUALIZE DETECTIONS")
    print("=" * 55)
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print("[1/3] Loading model...")
    model = YOLO(TRAINED_MODEL)
    print()
    
    # Process eval frames
    print("[2/3] Processing eval frames...")
    frames = sorted(glob.glob(os.path.join(EVAL_FRAMES_DIR, "*.jpg")))
    
    if not frames:
        print("  No eval frames found!")
        return
    
    # Get video properties for output
    sample = cv2.imread(frames[0])
    h, w = sample.shape[:2]
    
    # Video writer
    video_path = os.path.join(OUTPUT_DIR, "eval_detection.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 5.0, (w, h))
    
    example_frames = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
    
    for idx, frame_path in enumerate(frames):
        img = cv2.imread(frame_path)
        ih, iw = img.shape[:2]
        
        # Load GT
        basename = os.path.splitext(os.path.basename(frame_path))[0]
        label_path = os.path.join(EVAL_LABELS_DIR, f"{basename}.txt")
        gt_boxes = load_labels(label_path, iw, ih)
        
        # Run inference
        results = model(frame_path, conf=CONF_THRESHOLD, verbose=False)
        pred_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pred_boxes.append([x1, y1, x2, y2])
        
        # Draw
        vis = draw_frame(img, pred_boxes, gt_boxes, idx)
        video_writer.write(vis)
        
        # Save example frames
        if idx in example_frames:
            example_path = os.path.join(OUTPUT_DIR, f"example_frame_{idx:04d}.jpg")
            cv2.imwrite(example_path, vis)
            print(f"  Saved example: {example_path}")
        
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(frames)}")
    
    video_writer.release()
    print(f"  Video saved: {video_path}")
    print()
    
    # Also run on full eval video if available
    print("[3/3] Running on full eval video...")
    if os.path.exists(EVAL_VIDEO):
        cap = cv2.VideoCapture(EVAL_VIDEO)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        full_video_path = os.path.join(OUTPUT_DIR, "eval_full_detection.mp4")
        out = cv2.VideoWriter(full_video_path, fourcc, fps, (w, h))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            pred_boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    pred_boxes.append([x1, y1, x2, y2])
            
            vis = draw_frame(frame, pred_boxes, [], frame_idx)
            out.write(vis)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"  {frame_idx}/{total}")
        
        cap.release()
        out.release()
        print(f"  Full video: {full_video_path}")
    else:
        print(f"  Eval video not found: {EVAL_VIDEO}")
    
    print()
    print("  Done! Check output/ folder for results.")
    print()


if __name__ == "__main__":
    main()
