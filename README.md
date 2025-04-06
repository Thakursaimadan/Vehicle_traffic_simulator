# Vehicle Detection and Traffic Density Estimator 🚗🚌🚦

This project detects vehicles in a video feed and classifies traffic density using YOLOv5 and OpenCV.

## 🔍 Features
- Vehicle detection using YOLOv5 (`yolov5s` model).
- Detects cars, trucks, buses, and motorcycles.
- Counts vehicles in each frame.
- Displays traffic density level (Low / Medium / High) on the video.
- Saves output video with annotations.

## 🛠️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
Also, clone the YOLOv5 repo (if you haven't):
```bash
git clone https://github.com/ultralytics/yolov5
```
▶️ How to Run
```bash
python traffic_density.py
```
📦 Output
The script creates output_with_density.mp4 with bounding boxes and traffic level overlays.

📸 Example Output
![image](https://github.com/user-attachments/assets/0fff3de4-9125-450c-921e-2aa948cdbe5d)

📁 Dataset
You can test on any traffic video. Make sure to name it traffic.mp4 or change the video_path in the code.

🤖 Model Info
Using YOLOv5s, pre-trained model via PyTorch Hub.
