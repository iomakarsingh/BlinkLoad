import cv2
import mediapipe as mp
import time
import numpy as np

# Eye landmark indices (MediaPipe Standard)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# EAR landmark indices
L_HORIZONTAL = [362, 263]
L_VERTICAL = [(385, 380), (387, 373)]
R_HORIZONTAL = [33, 133]
R_VERTICAL = [(160, 144), (158, 153)]

# Blink Detection Constants
EAR_THRESHOLD = 0.22      # Empirical threshold for eye closure
EAR_CONSEC_FRAMES = 3     # Minimum frames for a valid blink

def calculate_ear(landmarks, horizontal_indices, vertical_indices, w, h):
    """
    Calculate Eye Aspect Ratio (EAR).
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    try:
        # Get coordinates
        horiz_pts = [np.array([landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]) for idx in horizontal_indices]
        vert_pts = []
        for v1, v2 in vertical_indices:
            v1_pt = np.array([landmarks.landmark[v1].x * w, landmarks.landmark[v1].y * h])
            v2_pt = np.array([landmarks.landmark[v2].x * w, landmarks.landmark[v2].y * h])
            vert_pts.append((v1_pt, v2_pt))

        # Vertical distances
        v_dist1 = np.linalg.norm(vert_pts[0][0] - vert_pts[0][1])
        v_dist2 = np.linalg.norm(vert_pts[1][0] - vert_pts[1][1])

        # Horizontal distance
        h_dist = np.linalg.norm(horiz_pts[0] - horiz_pts[1])

        # EAR formula
        ear = (v_dist1 + v_dist2) / (2.0 * h_dist)
        return ear
    except Exception as e:
        return 0.0

def main():
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

    # Initialize blink state variables
    blink_counter = 0
    total_blinks = 0

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0
    
    print("BlinkLoad started. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        avg_ear = 0
        # Draw face mesh landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Calculate EAR for both eyes
                left_ear = calculate_ear(face_landmarks, L_HORIZONTAL, L_VERTICAL, w, h)
                right_ear = calculate_ear(face_landmarks, R_HORIZONTAL, R_VERTICAL, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                # Blink Detection State Machine (Robust: require both eyes)
                if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    # If eyes were closed for at least EAR_CONSEC_FRAMES
                    if blink_counter >= EAR_CONSEC_FRAMES:
                        total_blinks += 1
                    blink_counter = 0

                # Draw the full face mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Highlight eye landmarks in real-time
                for idx in LEFT_EYE + RIGHT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Display Stats
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Blinks: {total_blinks}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Visual Debugging: Show BLINK text
        if blink_counter > 0:
            cv2.putText(frame, "--- BLINKING ---", (w // 2 - 100, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        # Show the frame
        cv2.imshow("BlinkLoad - Blink Counter & Dashboard", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Resources released. Exiting cleanly.")

if __name__ == "__main__":
    main()
