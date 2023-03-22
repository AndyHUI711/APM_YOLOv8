from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model = YOLO("weights/YOLOv8_best.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
success = model.export(format="onnx")

