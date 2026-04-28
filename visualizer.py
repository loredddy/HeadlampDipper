import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    def draw_detections(self, frame, detections):
        """
        Draws bounding boxes and labels for detections.
        """
        for d in detections:
            x1, y1, x2, y2 = d['box']
            facing_str = "Y" if d.get('facing', True) else "N"
            label = f"{d['class_name']} {d['conf']:.2f} F:{facing_str}"
            
            # Color code: Green if facing (Threat), Blue if ignored
            box_color = (0, 255, 0) if d.get('facing', True) else (255, 0, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    def draw_status(self, frame, beam_state):
        """
        Draws the system status (BEAM: HIGH/LOW) on the top bar.
        """
        h, w, _ = frame.shape
        color = (0, 0, 255) if beam_state == 'HIGH' else (255, 255, 0) # Red for high, Cyan for Low
        status_text = f"BEAM: {beam_state}"
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def draw_headlight_simulation(self, frame, beam_state):
        """
        Simulates the visual effect of headlights on the video feed.
        """
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Define beam colors (Yellowish white)
        beam_color = (150, 255, 255) # BGR (Light Yellow)
        alpha = 0.3 # Transparency
        
        if beam_state == 'HIGH':
            # High Beam: Illuminates mostly the center/upper part
            # Draw a trapezoid focused further up
            pts = np.array([[0, h], [w, h], [w//2 + 100, h//2 - 50], [w//2 - 100, h//2 - 50]], np.int32)
            cv2.fillPoly(overlay, [pts], beam_color)
            cv2.putText(overlay, "HIGH BEAM ON", (w//2 - 60, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        else: # LOW
            # Low Beam: Illuminates only the bottom part (road)
            # Draw a wider, lower trapezoid
            pts = np.array([[0, h], [w, h], [w, h//2 + 50], [0, h//2 + 50]], np.int32)
            cv2.fillPoly(overlay, [pts], beam_color)
            cv2.putText(overlay, "LOW BEAM (DIPPED)", (w//2 - 90, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
        # Apply the overlay
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
