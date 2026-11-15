import cv2
import os
from collections import defaultdict

def convert_gdxray_to_yolo(folder_path, output_dir, folder_prefix="W0001", class_id=0):
    """
    Parses the specific GDXray scientific-notation ground_truth.txt 
    and generates YOLO format .txt files.
    """
    ground_truth_file = os.path.join(folder_path, "ground_truth.txt")
    
    if not os.path.exists(ground_truth_file):
        print(f"Error: {ground_truth_file} not found.")
        return

    # Dictionary to hold all bounding boxes for each image
    # Format: image_labels[1] =["0 0.5 0.5 0.2 0.2", "0 0.6 0.6 0.1 0.1"]
    image_labels = defaultdict(list)

    # 1. Read the text file
    with open(ground_truth_file, 'r') as f:
        lines = f.readlines()

    # 2. Parse every line
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        
        # Convert scientific notation to standard floats
        img_index = int(float(parts[0]))
        y_min = float(parts[1])
        y_max = float(parts[2])
        x_min = float(parts[3])
        x_max = float(parts[4])

        # 3. Get image dimensions (YOLO needs normalized coordinates)
        # Construct the image name, e.g., W0001_0001.png
        img_name = f"{folder_prefix}_{img_index:04d}.png"
        img_path = os.path.join(folder_path, img_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} found in text file but not on disk.")
            continue

        # Load image just to get width and height
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        height, width, _ = img.shape

        # 4. Calculate YOLO format (Center X, Center Y, Width, Height) -> Normalized to 0.0 - 1.0
        box_width = x_max - x_min
        box_height = y_max - y_min
        
        center_x = (x_min + (box_width / 2.0)) / width
        center_y = (y_min + (box_height / 2.0)) / height
        norm_w = box_width / width
        norm_h = box_height / height

        # 5. Store the formatted line
        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"
        image_labels[img_index].append(yolo_line)

    # 6. Write the YOLO .txt files
    os.makedirs(output_dir, exist_ok=True)
    
    for img_index, labels in image_labels.items():
        txt_filename = f"{folder_prefix}_{img_index:04d}.txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        
        with open(txt_filepath, 'w') as f:
            f.write("\n".join(labels))
            
    print(f"Successfully generated {len(image_labels)} YOLO label files in {output_dir}")


if __name__ == "__main__":
    # === ADJUST THESE PATHS based on where your data is ===
    
    # Example: If your W0001 folder is in data/raw/
    INPUT_W0001_DIR = "data/raw/W0001" 
    
    # Where you want the YOLO text files to go
    OUTPUT_LABEL_DIR = "data/annotations" 
    
    convert_gdxray_to_yolo(
        folder_path=INPUT_W0001_DIR,
        output_dir=OUTPUT_LABEL_DIR,
        folder_prefix="W0001",
        class_id=0  # 0 represents 'defect' or 'porosity'
    )