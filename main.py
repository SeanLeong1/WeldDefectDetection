import cv2
import time
import numpy as np
from src.algorithms.classical.morphology_detector import ClassicalMorphologyDetector

def main():
    # 1. Configuration (Simulates a config.yaml)
    config = {
        'blur_kernel': (7, 7),
        'min_area': 150,  # Minimum pixel size for a defect
        'threshold': 40
    }

    # 2. Instantiate the chosen algorithm
    # Later, you can swap this class for YoloDetector()
    detector = ClassicalMorphologyDetector(config)
    detector.load_model()

    # 3. Load Sample Image (Replace with your RIAWELC/GDXray path)
    image_path = "data/raw/RRT-13R_Img1_A80_S3_[10][24].png"
    # Create a dummy image for testing if file doesn't exist
    image = cv2.imread(image_path) 
    if image is None:
        print("Creating dummy weld image for demonstration...")
        image = np.ones((512, 512, 3), dtype=np.uint8) * 200
        cv2.circle(image, (256, 256), 20, (50, 50, 50), -1) # Simulate porosity

    # 4. Measure "Operational Feasibility" (Latency)
    start_time = time.time()
    detections = detector.detect(image)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000

    # 5. Visualization
    output_img = image.copy()
    print(f"Algorithm: Classical Morphology")
    print(f"Inference Time: {latency_ms:.2f} ms")
    print(f"Defects Found: {len(detections)}")

    for det in detections:
        x_min, y_min, x_max, y_max = det.bbox
        cv2.rectangle(output_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(output_img, det.label, (x_min, y_min-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Show result
    cv2.imshow("Defect Detection", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()