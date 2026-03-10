import cv2
import yaml
import os
import numpy as np
from src.algorithms.classical.morphology_detector import MorphologyDetector

def get_boxes_from_mask(mask_path: str):
    """Extracts bounding boxes from the W0002 binary masks."""
    boxes = []
    if not os.path.exists(mask_path):
        return boxes

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return boxes

    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x-2, y-2, x+w+2, y+h+2)) # Slight buffer for visibility
        
    return boxes

def main():
    # 1. Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    detector = MorphologyDetector(config)

    # 2. Select an image pair to debug (Let's use image 1)
    img_name = "W0001_0001.png"
    mask_name = "W0002_0001.png"
    
    img_path = os.path.join("data/raw/W0001", img_name)
    mask_path = os.path.join("data/raw/W0002", mask_name)

    # Load the raw X-ray
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not load {img_path}")
        return

    # Create a copy for drawing so we don't mess up the inference image
    display_image = image.copy()

    # 3. Get Predictions (Red Boxes)
    print("Running detector...")
    results = detector.detect(image)
    for res in results:
        x1, y1, x2, y2 = res.box
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2) 

    # 4. Get Ground Truth (Green Boxes)
    print("Loading ground truth masks...")
    gt_boxes = get_boxes_from_mask(mask_path)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2) 

    # 5. Display the Results
    print(f"Found {len(gt_boxes)} Ground Truth boxes (Green) and {len(results)} Predictions (Red).")
    
    # Resize for the screen (X-rays are very long)
    h, w = display_image.shape[:2]
    screen_width = 1400
    scale = screen_width / w
    resized_display = cv2.resize(display_image, (int(w * scale), int(h * scale)))

    cv2.imshow("Debug: Green=Truth, Red=Prediction", resized_display)
    
    print("Press any key in the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()