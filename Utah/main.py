import cv2 as cv
import mediapipe as mp
import time
import math
from collections import deque

latest_result = None
last_timestamp_ms = 0

def save_result_callback(result, output_image, timestamp_ms):
    global latest_result
    # The 'result' parameter contains your .gestures and .hand_landmarks
    latest_result = result

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=save_result_callback)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm base
]


with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.flip(frame, 1)
        h, w, _ = frame.shape

        color_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB).copy()
        
        if cv.waitKey(5) == ord('q'):
            break

        frame_timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_frame)
        if frame_timestamp_ms <= last_timestamp_ms:
            frame_timestamp_ms = last_timestamp_ms + 1
        
        # Update the tracker
        last_timestamp_ms = frame_timestamp_ms

        try:
            recognizer.recognize_async(mp_image, frame_timestamp_ms)
        except Exception as e:
            print(f"Inference error skipped: {e}")
            continue
        
        if latest_result is not None and latest_result.hand_landmarks:
            for i, hand_landmarks in enumerate(latest_result.hand_landmarks):
                # Convert normalized landmarks to pixel coordinates
                pixel_landmarks = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pixel_landmarks.append((cx, cy))
                    
                    # Draw a small blue circle on every keypoint joint
                    cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                # Draw the skeletal connecting lines between joints
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    cv.line(frame, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 255, 0), 2)

                if latest_result.gestures and i < len(latest_result.gestures):
                    hand_gestures = latest_result.gestures[i]
                    if hand_gestures:
                        top_gesture = hand_gestures[0]
                        gesture_name = top_gesture.category_name
                        confidence = top_gesture.score

                        # Put the text right above the wrist joint (index 0)
                        wrist_x, wrist_y = pixel_landmarks[0]
                        text = f"{gesture_name} ({confidence:.2f})"
                        cv.putText(frame, text, (wrist_x, wrist_y - 20), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
                
                
        cv.imshow('MediaPipe Gesture Recognition', frame)
cap.release()
cv.destroyAllWindows()