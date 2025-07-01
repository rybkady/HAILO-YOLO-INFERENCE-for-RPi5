import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("flies.onnx")

# Set image source directory
image_dir = "images_val"

# Set model parameters
input_size = 320  # trained resolution
conf_threshold = 0.4
iou_threshold = 0.8

image = cv2.imread('test.jpg')

if image is None:
   print(f"Failed to load {filename}")

h, w = image.shape[:2]  # Image dimensions

# Run inference
results = model.predict(source=image,conf=conf_threshold,iou=iou_threshold,imgsz=input_size,verbose=False)

print(f"\nResults for ONNX:")
for i, box in enumerate(results[0].boxes.xyxy.cpu()):
    x1, y1, x2, y2 = map(int, box)
    conf = results[0].boxes.conf[i].item()
     # Relative coordinates to compare
    rel_x1 = x1 / w
    rel_y1 = y1 / h
    rel_x2 = x2 / w
    rel_y2 = y2 / h
    print(f"Box {i+1} coords(native output): [{x1}, {y1}, {x2}, {y2}], confidence={conf:.2f}.")
    print(f"Box {i+1} relative(calculated)=({rel_x1:.3f}, {rel_y1:.3f}, {rel_x2:.3f}, {rel_y2:.3f})")
 # Draw boxes on the original image
for box in results[0].boxes.xyxy.cpu():
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
 # Optionally draw confidence scores
if results[0].boxes.conf is not None:
    for i, conf in enumerate(results[0].boxes.conf.cpu()):
        x1, y1 = map(int, results[0].boxes.xyxy[i][:2].cpu())
        cv2.putText(image, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
 # Show image with boxes
cv2.imshow("Result", image)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
