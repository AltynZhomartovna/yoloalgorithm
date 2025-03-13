import cv2
import torch
from ultralytics import YOLO
from google.colab import files
import os

# Upload video file
uploaded = files.upload()
video_path = list(uploaded.keys())[0]  # Get uploaded filename

# Load YOLO model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt', 'yolov8m.pt', etc.

# Open video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter("output_yolo.mp4", fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Draw detections on frame
    for result in results:
        frame = result.plot()

    # Write frame to output video
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# Display output video in Colab
from IPython.display import display, HTML
import shutil

shutil.move("output_yolo.mp4", "/content/output_yolo.mp4")
display(HTML("""
  <video width="640" height="480" controls>
    <source src="/content/output_yolo.mp4" type="video/mp4">
  </video>
"""))
