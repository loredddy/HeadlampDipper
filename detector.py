import cv2
from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLOv8 detector.
        Downloads the model automatically if not present.
        """
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        # Targeted classes for traffic/pedestrians
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [0, 1, 2, 3, 5, 7]
        
        # Initialize Pose model for human verification
        print("Loading YOLO Pose model...")
        self.pose_model = YOLO('yolov8n-pose.pt') 
        print("Pose model loaded.")

    def is_facing_camera(self, person_crop):
        """
        Run pose estimation on the person crop.
        Returns True if facial features (eyes/nose) are detected with high confidence.
        """
        if person_crop.size == 0:
            return False
            
        results = self.pose_model(person_crop, verbose=False)
        
        for result in results:
            if result.keypoints is None or len(result.keypoints.data) == 0:
                continue
                
            # Keypoints: 0: Nose, 1: Left Eye, 2: Right Eye
            # data is a tensor of shape (N, 17, 3) where N is number of persons in crop
            # We took the crop of a SINGLE person, so we expect N=1 ideally.
            kpts = result.keypoints.data[0] 
            
            nose_conf = float(kpts[0][2])
            l_eye_conf = float(kpts[1][2])
            r_eye_conf = float(kpts[2][2])
            
            # DEBUG: Print confidences
            # print(f"Nose: {nose_conf:.2f}, L_Eye: {l_eye_conf:.2f}, R_Eye: {r_eye_conf:.2f}")

            # Lower threshold to 0.4 for better sensitivity in low light
            if nose_conf > 0.4 or (l_eye_conf > 0.4 and r_eye_conf > 0.4):
                return True
                
        return False

    def detect(self, frame):
        """
        Run detection on a single frame.
        Returns a list of detections: {'box': [x1, y1, x2, y2], 'class': id, 'conf': float, 'facing': bool}
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id in self.target_classes and conf > 0.4:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    is_facing = True # Default for vehicles
                    
                    # Special check for Person (class 0)
                    if cls_id == 0:
                        # Ensure crop is within bounds
                        h, w, _ = frame.shape
                        cy1, cy2 = max(0, y1), min(h, y2)
                        cx1, cx2 = max(0, x1), min(w, x2)
                        
                        person_crop = frame[cy1:cy2, cx1:cx2]
                        is_facing = self.is_facing_camera(person_crop)

                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'class_id': cls_id,
                        'class_name': self.model.names[cls_id],
                        'conf': conf,
                        'facing': is_facing # Add facing status
                    })
        return detections
