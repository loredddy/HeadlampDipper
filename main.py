import cv2
import argparse
from detector import ObjectDetector
from controller import HeadlampController
from visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description='Auto Headlamp Dipper System')
    parser.add_argument('--source', type=str, default='0', help='Video source: "0" for webcam, or path to video file.')
    parser.add_argument('--simulate', action='store_true', help='Enable virtual headlight simulation overlay.')
    args = parser.parse_args()

    print("Initializing Auto Headlamp Dipper System...")
    
    # 1. Initialize Components
    try:
        detector = ObjectDetector()
        controller = HeadlampController(switch_delay=1.5)
        visualizer = Visualizer()
    except Exception as e:
        print(f"Error initializing components: {e}")
        return

    # 2. Open Video Source
    if args.source.isdigit():
        video_source = int(args.source)
    else:
        video_source = args.source
        
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{args.source}'")
        return

    print("System Running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or end of video.")
            break

        h, w, _ = frame.shape

        # 3. Detection
        detections = detector.detect(frame)

        # 4. Control Logic
        beam_state = controller.update(detections, w, h)

        # 5. Visualization
        visualizer.draw_detections(frame, detections)
        visualizer.draw_status(frame, beam_state)

        # Optional Simulation Overlay
        if args.simulate:
            visualizer.draw_headlight_simulation(frame, beam_state)
        
        # Show feed
        cv2.imshow('Auto Dipper View', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("System Shutdown.")

if __name__ == "__main__":
    main()
