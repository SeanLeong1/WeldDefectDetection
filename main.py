import cv2
import yaml
import time
import os
from src.algorithms import MorphologyDetector
from src.utils.visualisation import build_stage_montage, draw_detections, fit_to_bounds

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Developer toggle: 1 = show full stage pipeline, 0 = show boxed output only.
    developer_show_pipeline_stages = 1
    # Developer toggle: 1 = vertical stack, 0 = horizontal strip.
    developer_vertical_stage_layout = 1
    # Developer display bounds to keep output on-screen.
    developer_max_display_width = 1920
    developer_max_display_height = 1080

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
    if developer_show_pipeline_stages == 1:
        results, stages = detector.detect_with_stages(image)
    else:
        results = detector.detect(image)
        stages = {}
    end_time = time.time()

    latency_ms = (end_time - start_time) * 1000

    # 5. Output Stats
    print(f"Processing Time: {latency_ms:.2f} ms")
    print(f"Defects Detected: {len(results)}")

    # 6. Visualize
    vis_image = draw_detections(image, results)
    if developer_show_pipeline_stages == 1:
        stage_layout = "vertical" if developer_vertical_stage_layout == 1 else "horizontal"
        output_image = build_stage_montage([
            ("Input", stages["input"]),
            ("Black-Hat Response", stages["blackhat"]),
            ("Threshold Mask", stages["threshold"]),
            ("Cleaned Mask", stages["opened"]),
            ("Contours", stages["contours"]),
            ("Final Output (Boxes)", vis_image),
        ], layout=stage_layout)
        output_image = fit_to_bounds(
            output_image,
            max_width=developer_max_display_width,
            max_height=developer_max_display_height,
        )
        cv2.namedWindow("Classical Detection Stages", cv2.WINDOW_NORMAL)
        cv2.imshow("Classical Detection Stages", output_image)
    else:
        cv2.imshow("Classical Detection", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
