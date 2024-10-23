import cv2
from ultralytics import YOLO
import pandas as pd
import os
model = YOLO("C:/Users/himan/OneDrive/Desktop/major project/bestf.pt")

cap = cv2.VideoCapture("C:/Users/himan/Downloads/Untitled video - Made with Clipchamp (4).mp4")
data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

    for result in results: 
        if result.boxes: 
            for box in result.boxes: 
               
                x1, y1, x2, y2 = box.xyxy[0]  
                conf = box.conf[0] 
                cls = int(box.cls[0])  
                label = model.names[cls]

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                vehicle_image = frame[int(y1):int(y2), int(x1):int(x2)]

                image_path = f"images/{timestamp}_{label}.jpg"
                cv2.imwrite(image_path, vehicle_image)

                data.append({
                    "Class": label,
                    "Timestamp": timestamp,
                    "Image Path": image_path
                })

cap.release()

df = pd.DataFrame(data)

df.to_excel("C:/Users/himan/OneDrive/Desktop/major project/vehicle_data.xlsx", index=False)