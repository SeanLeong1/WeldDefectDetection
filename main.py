import cv2
import yaml
import time
import os
from src.algorithms import MorphologyDetector
from src.utils.visualisation import draw_detections

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 1. Load Configuration
    config = load_config("config.yaml")
    
    # 2. Initialize Program 1 (Classical)
    detector = MorphologyDetector(config)
    print("Algorithm initialized: Classical Morphology")

    # 3. Load Data
    img_path = config["paths"]["input_image"]
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        print("Please place a test image in data/raw/")
        return

    image = cv2.imread(img_path)

    # 4. Run Inference & Measure "Operational Feasibility" (Latency)
    start_time = time.time()
    results = detector.detect(image)
    end_time = time.time()

    latency_ms = (end_time - start_time) * 1000

    # 5. Output Stats
    print(f"Processing Time: {latency_ms:.2f} ms")
    print(f"Defects Detected: {len(results)}")

    # 6. Visualize
    vis_image = draw_detections(image, results)
    cv2.imshow("Classical Detection", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()