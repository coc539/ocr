from ultralytics import YOLO
 
model = YOLO("yolov8n.pt")
 
results = model.train(data=r"D:\Yolo\data\data.yaml", epochs=50)

