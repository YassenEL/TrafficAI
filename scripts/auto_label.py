from ultralytics import YOLO
import pathlib
from tqdm import tqdm

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # or use yolov8s.pt/yolov8m.pt for better accuracy

# Setup paths
frames_dir = pathlib.Path("../dataset_source/frames_filtered")
labels_dir = frames_dir.parent / "labels"
labels_dir.mkdir(exist_ok=True)

def convert_to_yolo_format(box, img_width, img_height):
    """Convert box coordinates to YOLO format."""
    x, y, w, h = box
    return [
        (x + w/2) / img_width,  # center x
        (y + h/2) / img_height, # center y
        w / img_width,          # width
        h / img_height          # height
    ]

# Process each image
for img_path in tqdm(list(frames_dir.glob("*.jpg"))):
    # Run detection
    results = model(img_path, conf=0.25)[0]  # adjust confidence threshold as needed
    
    # Get image dimensions
    img_height, img_width = results.orig_shape
    
    # Create label file
    label_path = labels_dir / f"{img_path.stem}.txt"
    
    with open(label_path, 'w') as f:
        for box in results.boxes:
            if box.cls == 2:  # car
                # Convert box coordinates to YOLO format
                x, y, w, h = box.xywh[0].tolist()  # get box in xywh format
                yolo_box = convert_to_yolo_format((x-w/2, y-h/2, w, h), img_width, img_height)
                
                # Write to file: class_id x_center y_center width height
                f.write(f"0 {' '.join(f'{x:.6f}' for x in yolo_box)}\n")
            elif box.cls == 5:  # bus
                yolo_box = convert_to_yolo_format(box.xywh[0].tolist(), img_width, img_height)
                f.write(f"1 {' '.join(f'{x:.6f}' for x in yolo_box)}\n")
            elif box.cls == 7:  # truck
                yolo_box = convert_to_yolo_format(box.xywh[0].tolist(), img_width, img_height)
                f.write(f"2 {' '.join(f'{x:.6f}' for x in yolo_box)}\n")

print("Auto-labeling complete! Labels saved in:", labels_dir)