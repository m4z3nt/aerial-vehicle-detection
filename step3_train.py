"""
STEP 3: Train YOLOv8n on auto-labeled aerial vehicle dataset

Strategy:
- Use YOLOv8n (nano) — small, fast, good for aerial/edge deployment
- Transfer learning from COCO pretrained weights
- 50 epochs, imgsz=640
- Auto-augmentation by YOLO (flip, scale, mosaic, mixup)

Run: python step3_train.py
"""

import os
from ultralytics import YOLO

# ============================================
# SETTINGS
# ============================================
MODEL = "yolov8n.pt"          # Nano — fast, suitable for edge/drone
DATA_YAML = "data/dataset/data.yaml"
EPOCHS = 50
IMGSZ = 640
BATCH = 16                    # Reduce to 8 if GPU OOM
PROJECT = "runs"
NAME = "aerial_vehicle_v1"
PATIENCE = 15                 # Early stopping


def main():
    print("=" * 55)
    print("  STEP 3: TRAIN MODEL")
    print("  YOLOv8n fine-tune on aerial vehicle data")
    print("=" * 55)
    print()
    
    # Verify dataset exists
    if not os.path.exists(DATA_YAML):
        print(f"  ERROR: {DATA_YAML} not found!")
        print("  Run step2_auto_label.py first.")
        return
    
    # Count images
    train_dir = "data/dataset/images/train"
    val_dir = "data/dataset/images/val"
    train_count = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
    val_count = len(os.listdir(val_dir)) if os.path.exists(val_dir) else 0
    
    print(f"  Model:    {MODEL}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Train:    {train_count} images")
    print(f"  Val:      {val_count} images")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  ImgSize:  {IMGSZ}")
    print(f"  Batch:    {BATCH}")
    print(f"  Patience: {PATIENCE}")
    print()
    
    # Load model
    print("  Loading YOLOv8n pretrained on COCO...")
    model = YOLO(MODEL)
    print("  Model loaded!")
    print()
    
    # Train
    print("  Starting training...")
    print("  (This may take 10-30 min on RTX GPU)")
    print()
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        patience=PATIENCE,
        save=True,
        plots=True,
        verbose=True,
        
        # Augmentation (YOLO defaults are good)
        augment=True,
        mosaic=1.0,           # Mosaic augmentation
        mixup=0.1,            # MixUp augmentation
        copy_paste=0.1,       # Copy-paste augmentation
        
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,            # Initial learning rate
        lrf=0.01,             # Final LR as fraction of lr0
        weight_decay=0.0005,
        
        # Other
        workers=4,
        seed=42,
        deterministic=True,
    )
    
    print()
    print("=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    
    # Show results
    best_model = f"{PROJECT}/{NAME}/weights/best.pt"
    if os.path.exists(best_model):
        print(f"  Best model: {best_model}")
        
        # Validate
        print()
        print("  Running validation...")
        model = YOLO(best_model)
        metrics = model.val(data=DATA_YAML)
        
        print()
        print(f"  mAP50:     {metrics.box.map50:.4f}")
        print(f"  mAP50-95:  {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall:    {metrics.box.mr:.4f}")
    else:
        print(f"  WARNING: {best_model} not found")
    
    print()
    print(f"  Results:  {PROJECT}/{NAME}/")
    print(f"  Plots:    {PROJECT}/{NAME}/results.png")
    print(f"  Weights:  {PROJECT}/{NAME}/weights/best.pt")
    print()
    print("  Next step: python step4_evaluate.py")
    print()


if __name__ == "__main__":
    main()
