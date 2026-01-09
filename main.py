import cv2
import mediapipe as mp
import time
from src.ear import calculate_ear, L_HORIZONTAL, L_VERTICAL, R_HORIZONTAL, R_VERTICAL
from src.blink_detector import BlinkDetector

# Eye landmark indices for visualization (MediaPipe Standard)
LEFT_EYE_VIS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_VIS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def main():
    """
    Main application loop for BlinkLoad.
    Initializes webcam, face mesh, and blink detector.
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize Blink Detector
    detector = BlinkDetector()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0
    is_blinking = False
    
    print("BlinkLoad started. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # Convert the BGR image to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        avg_ear = 0
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Calculate EAR for both eyes
                left_ear = calculate_ear(face_landmarks, L_HORIZONTAL, L_VERTICAL, w, h)
                right_ear = calculate_ear(face_landmarks, R_HORIZONTAL, R_VERTICAL, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                # Update Blink Detector state
                is_blinking = detector.update(left_ear, right_ear)

                # Draw the full face mesh for visualization
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Highlight eye landmarks specifically
                for idx in LEFT_EYE_VIS + RIGHT_EYE_VIS:
                    landmark = face_landmarks.landmark[idx]
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)

        # Performance Monitoring (FPS)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Dashboard UI
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Blinks: {detector.total_blinks}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Visual Debugging: Blink indicator
        if is_blinking:
            cv2.putText(frame, "--- BLINKING ---", (w // 2 - 100, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        # Render combined frame
        cv2.imshow("BlinkLoad - Final Dashboard", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Resources released. Exiting cleanly.")

if __name__ == "__main__":
    main()
