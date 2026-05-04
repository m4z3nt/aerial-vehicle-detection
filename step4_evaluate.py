"""
STEP 4: Evaluate trained model on held-out eval clip
Compute metrics across two distance bands: 0-200m and 200-400m

Distance estimation assumptions (stated clearly):
- Average car length: 4.5 meters
- Camera FOV: ~70° horizontal (typical drone/surveillance camera)
- Video resolution: from the actual eval video
- Distance ≈ (real_object_size × image_width) / (bbox_width_pixels × 2 × tan(FOV/2))

Run: python step4_evaluate.py
"""

import os
import cv2
import glob
import json
import math
import numpy as np
from ultralytics import YOLO

# ============================================
# SETTINGS
# ============================================
TRAINED_MODEL = "runs/aerial_vehicle_v1/weights/best.pt"
EVAL_FRAMES_DIR = "data/frames/eval"
EVAL_LABELS_DIR = "data/labels_auto/eval"  # GT from auto-labeling with YOLOv8x
EVAL_VIDEO = "data/videos/eval.mp4"
OUTPUT_DIR = "output"
CONF_THRESHOLD = 0.1
IOU_MATCH_THRESHOLD = 0.3

# Distance estimation parameters
REFERENCE_CAR_LENGTH_M = 4.5   # meters — average car length
CAMERA_FOV_DEG = 70.0          # degrees — horizontal FOV assumption
DISTANCE_BAND_NEAR = (0, 200)  # meters
DISTANCE_BAND_FAR = (200, 400) # meters


def estimate_distance(bbox_w_pixels, img_w_pixels, 
                       ref_size_m=REFERENCE_CAR_LENGTH_M,
                       fov_deg=CAMERA_FOV_DEG):
    """
    Estimate distance to object based on bbox width in pixels.
    
    Assumptions:
    - bbox_w_pixels corresponds to an object of ref_size_m meters
    - Camera has horizontal FOV of fov_deg degrees
    - Pinhole camera model
    
    Formula:
    distance = (ref_size_m × img_w_pixels) / (bbox_w_pixels × 2 × tan(FOV/2))
    
    Returns: estimated distance in meters
    """
    if bbox_w_pixels <= 0:
        return float('inf')
    
    fov_rad = math.radians(fov_deg)
    focal_length_pixels = img_w_pixels / (2 * math.tan(fov_rad / 2))
    distance = (ref_size_m * focal_length_pixels) / bbox_w_pixels
    
    return distance


def load_labels(label_path, img_w, img_h):
    """
    Load YOLO format labels and convert to pixel coordinates.
    Returns list of [x1, y1, x2, y2] in pixels.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            
            boxes.append([x1, y1, x2, y2])
    
    return boxes


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def match_detections(gt_boxes, pred_boxes, iou_threshold=IOU_MATCH_THRESHOLD):
    """
    Match predicted boxes to ground truth boxes.
    Returns: TP, FP, FN counts and matched pairs.
    """
    matched_gt = set()
    matched_pred = set()
    matches = []
    
    # Sort predictions by confidence (if available) or area
    for pi, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gi = -1
        
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        
        if best_iou >= iou_threshold and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            matches.append((best_gi, pi, best_iou))
    
    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    
    return tp, fp, fn, matches


def evaluate_by_distance(model, frames_dir, labels_dir, img_w, img_h):
    """
    Evaluate model on frames, computing metrics per distance band.
    """
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    if not frames:
        print("  No eval frames found!")
        return None
    
    # Per-band accumulators
    bands = {
        "0-200m": {"tp": 0, "fp": 0, "fn": 0, "first_det_frame": None},
        "200-400m": {"tp": 0, "fp": 0, "fn": 0, "first_det_frame": None},
    }
    
    total_frames = len(frames)
    fps_video = 30.0  # Assume 30 FPS — adjust based on actual video
    
    print(f"  Evaluating {total_frames} frames...")
    
    for frame_idx, frame_path in enumerate(frames):
        img = cv2.imread(frame_path)
        h, w = img.shape[:2]
        
        # Load GT labels
        basename = os.path.splitext(os.path.basename(frame_path))[0]
        label_path = os.path.join(labels_dir, f"{basename}.txt")
        gt_boxes = load_labels(label_path, w, h)
        
        # Run inference
        results = model(frame_path, conf=CONF_THRESHOLD, verbose=False)
        pred_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_boxes.append([x1, y1, x2, y2])
        
        # Assign distance bands to GT boxes
        for gt_box in gt_boxes:
            gt_w_px = gt_box[2] - gt_box[0]
            dist = estimate_distance(gt_w_px, w)
            
            if dist <= 200:
                band_key = "0-200m"
            elif dist <= 400:
                band_key = "200-400m"
            else:
                continue  # Beyond our bands
            
            # Find best matching prediction
            best_iou = 0
            for pred_box in pred_boxes:
                iou = compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
            
            if best_iou >= IOU_MATCH_THRESHOLD:
                bands[band_key]["tp"] += 1
                if bands[band_key]["first_det_frame"] is None:
                    bands[band_key]["first_det_frame"] = frame_idx
            else:
                bands[band_key]["fn"] += 1
        
        # Count FP per band
        for pred_box in pred_boxes:
            pred_w_px = pred_box[2] - pred_box[0]
            dist = estimate_distance(pred_w_px, w)
            
            matched = False
            for gt_box in gt_boxes:
                if compute_iou(pred_box, gt_box) >= IOU_MATCH_THRESHOLD:
                    matched = True
                    break
            
            if not matched:
                if dist <= 200:
                    bands["0-200m"]["fp"] += 1
                elif dist <= 400:
                    bands["200-400m"]["fp"] += 1
        
        # Progress
        if (frame_idx + 1) % 50 == 0 or (frame_idx + 1) == total_frames:
            print(f"    {frame_idx + 1}/{total_frames}")
    
    # Compute metrics
    metrics = {}
    for band_name, band_data in bands.items():
        tp = band_data["tp"]
        fp = band_data["fp"]
        fn = band_data["fn"]
        
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        false_alarms_per_min = (fp * 60 * fps_video) / total_frames if total_frames > 0 else 0
        
        first_frame = band_data["first_det_frame"]
        time_to_first = first_frame / fps_video if first_frame is not None else float('inf')
        
        metrics[band_name] = {
            "detection_rate": detection_rate,
            "precision": precision,
            "false_alarms_per_min": false_alarms_per_min,
            "time_to_first_detection": time_to_first,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    
    return metrics, total_frames


def print_metrics_table(metrics):
    """Print metrics in the required format"""
    print()
    print("  ┌─────────────────────────┬───────────┬────────────┐")
    print("  │ Metric                  │  0-200 m  │ 200-400 m  │")
    print("  ├─────────────────────────┼───────────┼────────────┤")
    
    for metric_name, display_name in [
        ("detection_rate", "Detection rate TP/(TP+FN)"),
        ("precision", "Precision TP/(TP+FP)"),
        ("false_alarms_per_min", "False alarms / min"),
        ("time_to_first_detection", "Time to first det (s)"),
    ]:
        near = metrics.get("0-200m", {}).get(metric_name, 0)
        far = metrics.get("200-400m", {}).get(metric_name, 0)
        
        if metric_name in ("detection_rate", "precision"):
            near_str = f"{near:.2f}"
            far_str = f"{far:.2f}"
        elif metric_name == "false_alarms_per_min":
            near_str = f"{near:.1f}"
            far_str = f"{far:.1f}"
        else:
            near_str = f"{near:.1f}" if near != float('inf') else "N/A"
            far_str = f"{far:.1f}" if far != float('inf') else "N/A"
        
        print(f"  │ {display_name:23s} │ {near_str:>9s} │ {far_str:>10s} │")
    
    print("  └─────────────────────────┴───────────┴────────────┘")
    print()
    
    # Raw counts
    for band in ["0-200m", "200-400m"]:
        m = metrics.get(band, {})
        print(f"  {band}: TP={m.get('tp',0)}, FP={m.get('fp',0)}, FN={m.get('fn',0)}")


def main():
    print("=" * 55)
    print("  STEP 4: EVALUATE MODEL")
    print("  Metrics by distance band on held-out eval clip")
    print("=" * 55)
    print()
    
    # Check model exists
    if not os.path.exists(TRAINED_MODEL):
        print(f"  ERROR: {TRAINED_MODEL} not found!")
        print("  Run step3_train.py first.")
        return
    
    # Load trained model
    print("[1/3] Loading trained model...")
    model = YOLO(TRAINED_MODEL)
    print(f"  Model: {TRAINED_MODEL}")
    print()
    
    # Distance estimation assumptions
    print("[2/3] Distance estimation assumptions:")
    print(f"  Reference object: car, {REFERENCE_CAR_LENGTH_M}m length")
    print(f"  Camera FOV: {CAMERA_FOV_DEG}° horizontal")
    print(f"  Model: pinhole camera, distance from bbox width")
    print(f"  Formula: d = (obj_size × focal_px) / bbox_w_px")
    print(f"  Bands: near = {DISTANCE_BAND_NEAR}m, far = {DISTANCE_BAND_FAR}m")
    print()
    
    # Evaluate
    print("[3/3] Running evaluation on eval clip...")
    
    # Get image dimensions from first frame
    first_frame = sorted(glob.glob(os.path.join(EVAL_FRAMES_DIR, "*.jpg")))[0]
    img = cv2.imread(first_frame)
    img_h, img_w = img.shape[:2]
    print(f"  Image size: {img_w}x{img_h}")
    
    metrics, total_frames = evaluate_by_distance(
        model, EVAL_FRAMES_DIR, EVAL_LABELS_DIR, img_w, img_h
    )
    
    # Print results
    print()
    print("=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    
    print_metrics_table(metrics)
    
    # Save metrics to JSON
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model": TRAINED_MODEL,
            "eval_frames": total_frames,
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_MATCH_THRESHOLD,
            "assumptions": {
                "reference_car_length_m": REFERENCE_CAR_LENGTH_M,
                "camera_fov_deg": CAMERA_FOV_DEG,
            },
            "metrics": metrics,
        }, f, indent=2)
    
    print(f"  Metrics saved: {metrics_path}")
    print()
    print("  Next step: python step5_visualize.py")
    print()


if __name__ == "__main__":
    main()
