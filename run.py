from ultralytics import YOLO
model = YOLO('C:/Users/himan/OneDrive/Desktop/major project/bestf.pt')
model.predict(source=0,imgsz=480,conf=0.8,show=True)