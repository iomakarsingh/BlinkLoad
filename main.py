import cv2
import mediapipe as mp
import time
from src.ear import calculate_ear, LEFT_EYE, RIGHT_EYE, L_HORIZONTAL, L_VERTICAL, R_HORIZONTAL, R_VERTICAL
from src.blink_detector import BlinkDetector

def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Refine landmarks is TRUE for detailed eye contour tracking
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize modular detector
    detector = BlinkDetector(threshold=0.22, consec_frames=3)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0
    print("BlinkLoad Modular V1.0 started. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        avg_ear = 0
        is_blinking = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # 1. Calculate EAR for both eyes
                left_ear = calculate_ear(face_landmarks, L_HORIZONTAL, L_VERTICAL, w, h)
                right_ear = calculate_ear(face_landmarks, R_HORIZONTAL, R_VERTICAL, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                # 2. Update blink detector state
                is_blinking = detector.update(left_ear, right_ear)

                # 3. Draw the full face mesh for visual feedback
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Highlight eye landmarks
                for idx in LEFT_EYE + RIGHT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)

        # 4. Performance & HUD Overlay
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Display Stats
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Blinks: {detector.total_blinks}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Visual Debugging: Blink Indicator
        if is_blinking:
            cv2.putText(frame, "--- BLINKING ---", (w // 2 - 100, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("BlinkLoad - Final Modular Dashboard", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Resources released. Exiting cleanly.")

if __name__ == "__main__":
    main()
