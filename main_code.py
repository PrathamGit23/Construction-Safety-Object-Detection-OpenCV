import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict

# ============================================================================
# COLOR DEFINITIONS FOR HELMET
# ============================================================================

HELMET_COLORS = {
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'orange': [(10, 100, 100), (20, 255, 255)],
    'white': [(0, 0, 200), (180, 30, 255)],
    'red': [(0, 100, 100), (10, 255, 255)],
    'blue': [(100, 100, 100), (130, 255, 255)],
    'green': [(40, 30, 50), (85, 255, 255)]
}

MIN_HELMET_AREA = 300

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_color_mask(hsv_image, color_ranges):
    """Create combined mask for multiple color ranges"""
    masks = []
    for color_range in color_ranges.values():
        lower, upper = np.array(color_range[0]), np.array(color_range[1])
        mask = cv2.inRange(hsv_image, lower, upper)
        masks.append(mask)
    
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    return combined_mask

# ============================================================================
# FACE DETECTOR (Haar Cascade)
# ============================================================================

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✅ Face detector loaded")
    
    def detect(self, frame):
        """Detect faces (bareheads) with reduced false positives"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(35, 35),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        for (x, y, w, h) in faces:
            aspect_ratio = w / h if h > 0 else 0
            if 0.7 < aspect_ratio < 1.3:
                detections.append((x, y, w, h))
        
        return detections

# ============================================================================
# EYE/GLASSES DETECTOR (Haar Cascade)
# ============================================================================

class GlassesDetector:
    def __init__(self):
        # Try to load eye cascade for glasses detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        print("✅ Glasses detector loaded")
    
    def detect_glasses(self, frame, face_box):
        """Detect if person is wearing safety glasses by checking eye region"""
        fx, fy, fw, fh = face_box
        
        # Define eye region (upper half of face)
        eye_region_y = fy + int(fh * 0.2)
        eye_region_h = int(fh * 0.4)
        eye_region_x = fx
        eye_region_w = fw
        
        # Ensure region is within frame
        if eye_region_y + eye_region_h > frame.shape[0]:
            eye_region_h = frame.shape[0] - eye_region_y
        if eye_region_x + eye_region_w > frame.shape[1]:
            eye_region_w = frame.shape[1] - eye_region_x
        
        if eye_region_h <= 0 or eye_region_w <= 0:
            return False
        
        # Extract eye region
        eye_roi = frame[eye_region_y:eye_region_y + eye_region_h,
                       eye_region_x:eye_region_x + eye_region_w]
        
        gray_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes/glasses
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        # Simple ML-like logic: If we detect 2 eye-like regions, person likely has glasses
        # (Eyes are more visible with glasses due to frame contrast)
        if len(eyes) >= 2:
            return True
        
        # Additional check: Look for darker regions (glasses frames)
        # This is a basic feature detection approach
        blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        _, binary = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
        
        # Count dark pixels (potential glasses frames)
        dark_pixels = np.sum(binary == 0)
        total_pixels = binary.size
        dark_ratio = dark_pixels / total_pixels
        
        # If significant dark regions exist, likely wearing glasses
        if dark_ratio > 0.3:
            return True
        
        return False

# ============================================================================
# HELMET CHECKER
# ============================================================================

class HelmetChecker:
    def __init__(self):
        print("✅ Helmet checker loaded")
    
    def check_helmet_above_face(self, frame, face_box):
        """Check if there's a helmet above a specific face"""
        fx, fy, fw, fh = face_box
        
        # Define helmet search region (above the face)
        helmet_region_y = max(0, fy - int(fh * 1.2))
        helmet_region_h = int(fh * 1.2)
        helmet_region_x = max(0, fx - int(fw * 0.2))
        helmet_region_w = int(fw * 1.4)
        
        # Ensure region is within frame
        if helmet_region_y + helmet_region_h > frame.shape[0]:
            helmet_region_h = frame.shape[0] - helmet_region_y
        if helmet_region_x + helmet_region_w > frame.shape[1]:
            helmet_region_w = frame.shape[1] - helmet_region_x
        
        if helmet_region_h <= 0 or helmet_region_w <= 0:
            return False
        
        # Extract helmet search region
        helmet_roi = frame[helmet_region_y:helmet_region_y + helmet_region_h,
                        helmet_region_x:helmet_region_x + helmet_region_w]
        
        # Convert to HSV and check for helmet colors
        hsv_roi = cv2.cvtColor(helmet_roi, cv2.COLOR_BGR2HSV)
        helmet_mask = create_color_mask(hsv_roi, HELMET_COLORS)
        
        # Apply additional morphological filtering to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        helmet_mask = cv2.morphologyEx(helmet_mask, cv2.MORPH_OPEN, kernel)
        helmet_mask = cv2.morphologyEx(helmet_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and check for helmet-like shape
        contours, _ = cv2.findContours(helmet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        # Must be significant size
        if contour_area < 400:
            return False
        
        # Check aspect ratio (helmets are roughly round/square)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        if not (0.6 < aspect_ratio < 1.8):
            return False
        
        # Check coverage (must be solid, not sparse)
        coverage = contour_area / (w * h) if (w * h) > 0 else 0
        if coverage < 0.45:
            return False
        
        return True


# ============================================================================
# MAIN DETECTION SYSTEM
# ============================================================================

class SafetyDetectionSystem:
    def __init__(self):
        print("\nInitializing detectors...")
        self.face_detector = FaceDetector()
        self.helmet_checker = HelmetChecker()
        self.glasses_detector = GlassesDetector()
        
        self.stats = defaultdict(int)
        self.frame_count = 0
        
        print("✅ All detectors initialized!")
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        
        # Step 1: Detect all faces
        face_boxes = self.face_detector.detect(frame)
        
        # Step 2: For each face, check helmet and glasses
        helmet_violations = 0
        glasses_violations = 0
        fully_compliant = 0
        
        faces_with_status = []
        for face_box in face_boxes:
            has_helmet = self.helmet_checker.check_helmet_above_face(frame, face_box)
            has_glasses = self.glasses_detector.detect_glasses(frame, face_box)
            
            faces_with_status.append((face_box, has_helmet, has_glasses))
            
            if not has_helmet:
                helmet_violations += 1
            if not has_glasses:
                glasses_violations += 1
            if has_helmet and has_glasses:
                fully_compliant += 1
        
        # Step 3: Update stats
        total_workers = len(face_boxes)
        self.stats['total'] = total_workers
        self.stats['helmet_violations'] = helmet_violations
        self.stats['glasses_violations'] = glasses_violations
        self.stats['fully_compliant'] = fully_compliant
        
        # Draw detections
        annotated_frame = frame.copy()
        
        for (face_box, has_helmet, has_glasses) in faces_with_status:
            fx, fy, fw, fh = face_box
            
            # Determine status color
            if has_helmet and has_glasses:
                # FULLY COMPLIANT - Green
                color = (0, 255, 0)
                status = "SAFE"
            elif has_helmet or has_glasses:
                # PARTIAL COMPLIANCE - Orange
                color = (0, 165, 255)
                status = "PARTIAL"
            else:
                # NO COMPLIANCE - Red
                color = (0, 0, 255)
                status = "UNSAFE"
            
            # Draw face rectangle
            cv2.rectangle(annotated_frame, (fx, fy), (fx + fw, fy + fh), color, 3)
            
            # Draw status
            cv2.putText(annotated_frame, status, (fx, fy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw detailed info below face
            y_pos = fy + fh + 20
            if not has_helmet:
                cv2.putText(annotated_frame, "NO HELMET", (fx, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y_pos += 20
            if not has_glasses:
                cv2.putText(annotated_frame, "NO GLASSES", (fx, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw stats
        y_offset = 30
        cv2.putText(annotated_frame, f"Total Workers: {self.stats['total']}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(annotated_frame, f"Fully Compliant: {self.stats['fully_compliant']}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(annotated_frame, f"Helmet Violations: {self.stats['helmet_violations']}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        cv2.putText(annotated_frame, f"Glasses Violations: {self.stats['glasses_violations']}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated_frame

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("\n" + "="*50)
    print("🏗️ Construction Site Safety Detection System")
    print("="*50)
    
    system = SafetyDetectionSystem()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n✅ Webcam configured successfully!")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("\n" + "="*50 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading frame")
            break
        
        frame = cv2.flip(frame, 1)
        annotated_frame = system.process_frame(frame)
        cv2.imshow('Construction Safety Detection', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"safety_capture_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ System shutdown complete")

if __name__ == "__main__":
    main()