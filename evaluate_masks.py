# evaluate_masks.py
import os
import cv2
import time
import yaml
from src.algorithms.classical.morphology_detector import MorphologyDetector
from src.utils.metrics import evaluate_detections

def get_boxes_from_mask(mask_path: str):
    """
    Reads a binary mask image and returns bounding boxes for all white blobs.
    """
    boxes = []
    if not os.path.exists(mask_path):
        return boxes

    # Load mask in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return boxes

    # Ensure binary (sometimes masks are saved as 0 and 255, or 0 and 1)
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Extract contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Add a tiny 2-pixel buffer to the ground truth box for fair IoU calculation
        boxes.append((x-2, y-2, x+w+2, y+h+2))
        
    return boxes

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Make sure your paths point to the raw W0001 and W0002 folders
    raw_img_dir = "data/raw/W0001"
    mask_dir = "data/raw/W0002"
    
    detector = MorphologyDetector(config)
    
    total_tp, total_fp, total_fn = 0, 0, 0
    total_latency_ms = 0
    num_images = 0

    print("Starting Mask-Based Baseline Evaluation...")
    
    # Iterate through the 10 original images
    for i in range(1, 11):
        img_name = f"W0001_{i:04d}.png"
        mask_name = f"W0002_{i:04d}.png"
        
        img_path = os.path.join(raw_img_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        image = cv2.imread(img_path)
        if image is None: 
            continue
            
        # 1. Inference
        start_time = time.time()
        results = detector.detect(image)
        latency_ms = (time.time() - start_time) * 1000
        
        pred_boxes = [res.box for res in results]
        
        # 2. Ground Truth from Mask
        gt_boxes = get_boxes_from_mask(mask_path)
        
        # 3. Evaluate (IoU threshold kept low for fuzzy classical boundaries)
        tp, fp, fn = evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.05)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_latency_ms += latency_ms
        num_images += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency = total_latency_ms / num_images if num_images > 0 else 0.0

    print("\n--- Absolute Operational Feasibility ---")
    print(f"Images Evaluated: {num_images}")
    print(f"Total True Positives (Defects Found): {total_tp}")
    print(f"Total False Negatives (Defects Missed): {total_fn}")
    print(f"Total False Positives (Noise Flagged): {total_fp}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print(f"Latency:   {avg_latency:.2f} ms per image")

if __name__ == "__main__":
    main()