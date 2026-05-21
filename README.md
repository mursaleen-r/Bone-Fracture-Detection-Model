# Bone-Fracture-Detection-Model
Bone Fracture Detection from X-Rays
This is a computer vision project I built to automatically detect and localize bone fractures in X-ray images.

Raw medical images can be pretty noisy and hard to read, so I set up a pipeline that first uses OpenCV to clean up the image and boost the contrast so the skeletal structure really stands out. Once the image is processed, it gets passed to a custom-trained YOLOv8 model that scans the bones and draws bounding boxes around any detected fractures.

How it works
Image Preprocessing: Uses OpenCV to strip out visual noise and isolate the bone structures.

Detection: Feeds the enhanced image into YOLOv8 to spot anomalies and output bounding box coordinates.

Tech Stack
Python

OpenCV (Image processing)

YOLOv8 (Object detection)
