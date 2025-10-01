from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # small, fast version
results = model.predict("traffic_sample.jpg", show=True)
