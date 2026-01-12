import numpy as np

# MediaPipe standard landmark indices for eyes
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Landmark indices specifically used for EAR calculation
# Points chosen based on MediaPipe Face Mesh canonical model
L_HORIZONTAL = [362, 263]
L_VERTICAL = [(385, 380), (387, 373)]
R_HORIZONTAL = [33, 133]
R_VERTICAL = [(160, 144), (158, 153)]

def calculate_ear(landmarks, horizontal_indices, vertical_indices, w, h):
    """
    Calculate the Eye Aspect Ratio (EAR).
    
    The formula is: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    Where p1, p4 are horizontal landmarks and p2, p3, p5, p6 are vertical landmarks.
    
    Args:
        landmarks: MediaPipe normalized landmarks list.
        horizontal_indices: List of 2 indices for horizontal points.
        vertical_indices: List of 2 tuples for vertical point pairs.
        w: Width of the image.
        h: Height of the image.
        
    Returns:
        float: Calculated EAR value.
    """
    try:
        # Get pixel coordinates for horizontal points
        horiz_pts = [np.array([landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]) for idx in horizontal_indices]
        
        # Get pixel coordinates for vertical point pairs
        vert_pts = []
        for v1, v2 in vertical_indices:
            v1_pt = np.array([landmarks.landmark[v1].x * w, landmarks.landmark[v1].y * h])
            v2_pt = np.array([landmarks.landmark[v2].x * w, landmarks.landmark[v2].y * h])
            vert_pts.append((v1_pt, v2_pt))

        # Vertical distances (Euclidean)
        v_dist1 = np.linalg.norm(vert_pts[0][0] - vert_pts[0][1])
        v_dist2 = np.linalg.norm(vert_pts[1][0] - vert_pts[1][1])

        # Horizontal distance (Euclidean)
        h_dist = np.linalg.norm(horiz_pts[0] - horiz_pts[1])

        # EAR calculation
        ear = (v_dist1 + v_dist2) / (2.0 * h_dist)
        return ear
    except (IndexError, AttributeError) as e:
        # Return 0 if landmarks are missing or malformed
        return 0.0
