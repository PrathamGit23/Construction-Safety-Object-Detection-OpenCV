from flask import Flask, Response, jsonify, render_template
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
import threading
import os

# Import your existing detection system
from main_code import SafetyDetectionSystem

app = Flask(__name__)

# Global variables
detection_system = SafetyDetectionSystem()
camera = None
camera_lock = threading.Lock()
latest_stats = {
    'total': 0, 
    'helmet_violations': 0, 
    'glasses_violations': 0, 
    'fully_compliant': 0
}
is_running = False

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def generate_frames():
    global latest_stats, is_running
    is_running = True
    
    while is_running:
        with camera_lock:
            cam = get_camera()
            success, frame = cam.read()
            
            if not success:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame with your detection system
            annotated_frame = detection_system.process_frame(frame)
            
            # Update stats
            latest_stats = dict(detection_system.stats)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(latest_stats)

@app.route('/capture', methods=['POST'])
def capture():
    with camera_lock:
        cam = get_camera()
        success, frame = cam.read()
        
        if success:
            frame = cv2.flip(frame, 1)
            annotated_frame = detection_system.process_frame(frame)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"safety_capture_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            
            return jsonify({'success': True, 'filename': filename})
    
    return jsonify({'success': False})

@app.route('/stop')
def stop():
    global is_running, camera
    is_running = False
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'success': True})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Construction Safety Detection System")
    print("="*50)
    print("📁 Looking for index.html in 'build' folder...")
    
    if os.path.exists('build/index.html'):
        print("✅ index.html found!")
    else:
        print("❌ ERROR: build/index.html not found!")
        print("Please create 'build' folder and add index.html")
    
    print("\n🌐 Open your browser and go to:")
    print("   http://localhost:5000")
    print("   or")
    print("   http://127.0.0.1:5000")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)