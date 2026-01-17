# Construction Safety Object Detection (OpenCV)

This project demonstrates a **computer vision–based system** that detects safety-related objects on construction sites using **Python and OpenCV**.

The goal is to help monitor whether workers are following basic safety protocols (such as wearing protective gear) by analyzing live video or webcam input through a simple web interface.

---

## What this project does

- Captures live video frames from a webcam  
- Processes each frame using OpenCV  
- Detects construction safety–related objects in real time  
- Displays detection results through a Flask-based web application  

This is a **classic OpenCV-based object detection project**, intended for learning and demonstration purposes.

---

## Why this project matters

Construction sites are high-risk environments.  
Automating safety monitoring using computer vision can help reduce accidents and improve compliance with safety rules.

This project is a **baseline implementation** that can be extended to:
- Deep learning–based object detection models  
- Multi-camera monitoring systems  
- Real-world industrial safety applications  

---

## DISCLAIMER

This project is for educational and demonstration purposes only and should not be considered a fully reliable safety monitoring system for real-world deployment.

---

## How to run the project locally

```bash
git clone https://github.com/PrathamGit23/Construction-Safety-Object-Detection-OpenCV.git
cd Construction-Safety-Object-Detection-OpenCV
pip install flask opencv-python numpy
python app.py

Open your browser and go to:

http://127.0.0.1:5000/
