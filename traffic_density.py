import cv2
import torch
from pathlib import Path


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')  # use yolov5s for speed

# Setting classes to detect only vehicles (car=2, motorcycle=3, bus=5, truck=7)
vehicle_classes = [2, 3, 5, 7]
model.classes = vehicle_classes


video_path = 'traffic.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_density.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

def get_density_label(count):
    if count < 5:
        return "Low", (0, 255, 0)   
    elif 5 <= count < 15:
        return "Medium", (0, 255, 255)  
    else:
        return "High", (0, 0, 255)   

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)
    detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, class]

    vehicle_count = 0

    for *box, conf, cls in detections:
        vehicle_count += 1
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    density_label, color = get_density_label(vehicle_count)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Traffic: {density_label}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    out.write(frame)
    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
