import time

class HeadlampController:
    def __init__(self, switch_delay=2.0):
        """
        state: 'HIGH' or 'LOW'
        switch_delay: Seconds to wait before switching back to HIGH beam 
                      after the last threat has passed (hysteresis).
        """
        self.state = 'HIGH'
        self.last_threat_time = 0
        self.switch_delay = switch_delay
        self.threat_active = False
        self.consecutive_threats = 0

    def update(self, detections, frame_width, frame_height):
        """
        Analyze detections to decide headlamp state.
        detections: List of dicts from ObjectDetector
        """
        current_time = time.time()
        threat_detected = False

        # Simple logic: If any vehicle is detected with a certain size (implies proximity)
        # or just presence in the scene for this prototype.
        # We can refine this to check if the object is in the 'center' or 'oncoming' lane.
        
        for d in detections:
            box = d['box']
            w = box[2] - box[0]
            h = box[3] - box[1]
            
            # Area ratio to frame
            area_ratio = (w * h) / (frame_width * frame_height)
            
            # Threshold: if object is big enough (close enough), it's a threat.
            # Tuning this value is key. For now, let's say 1% of screen.
            # ALSO: Check if it's facing us (for persons)
            is_facing = d.get('facing', True) # Default to True if key missing
            
            if area_ratio > 0.01 and is_facing: 
                threat_detected = True
                break
        
        self.threat_active = threat_detected

        # Debounce Logic:
        # We need 3 consecutive frames of threat to confirm it's real.
        if self.threat_active:
            self.consecutive_threats += 1
        else:
            self.consecutive_threats = 0

        # Only trigger "Active Threat" logic if we passed the debounce threshold
        if self.consecutive_threats >= 3:
            self.last_threat_time = current_time
            
            # Track the "primary" threat's position for exit logic
            # We take the largest vehicle/person as the primary threat
            max_area = 0
            primary_box = None
            
            for d in detections:
                 # Check if this detection is a threat (Vehicle or Facing Person)
                is_facing = d.get('facing', True)
                box = d['box']
                w = box[2] - box[0]
                h = box[3] - box[1]
                area = w * h
                
                if (d['class_id'] == 0 and is_facing) or (d['class_id'] != 0):
                    if area > max_area:
                        max_area = area
                        primary_box = box

            if primary_box:
                self.last_threat_box = primary_box

            # Adaptive Hysteresis Setup
            # Default to Vehicle (Low Beam Hold = 2.0s) or Person (5.0s)
            
            is_person_threat = False
            for d in detections:
                if d['class_id'] == 0 and d.get('facing', False):
                    is_person_threat = True
                    break
            
            if is_person_threat:
                self.switch_delay = 5.0
            else:
                self.switch_delay = 2.0 

            if self.state == 'HIGH':
                print(f">>> THREAT CONFIRMED (Frames: {self.consecutive_threats})! DIPPING HEADLIGHTS (Hold: {self.switch_delay}s) <<<")
                self.state = 'LOW'

        else:
            # NO CONFIRMED THREAT (either no detections, or not enough consecutive ones)
            
            # Note: We only check for switching back to HIGH if we are currently LOW.
            # And we rely on 'current_time - self.last_threat_time' which was set when we *last* had a confirmed threat.
            
            # Check for "Fast Exit" (Highway Mode)
            # If the last known threat was at the edge of the screen, it passed us.
            # We can switch back faster (0.5s).
            if self.state == 'LOW' and hasattr(self, 'last_threat_box') and self.switch_delay == 2.0:
                lx1, ly1, lx2, ly2 = self.last_threat_box
                frame_w = frame_width
                
                # Check if close to Left (0) or Right (frame_w) margin
                margin = frame_w * 0.1 # 10% margin
                if lx1 < margin or lx2 > (frame_w - margin):
                     # Vehicle exited frame -> Fast Switch
                     if self.switch_delay > 0.5: # Don't overwrite if we already set it low
                        self.switch_delay = 0.5
                        # print("DEBUG: Fast Exit Detected -> Switching in 0.5s")

            # If no threat, check if we waited enough time
            if self.state == 'LOW' and (current_time - self.last_threat_time > self.switch_delay):
                print(f">>> CLEAR! SWITCHING TO HIGH BEAM (After {self.switch_delay}s) <<<")
                self.state = 'HIGH'

        return self.state
