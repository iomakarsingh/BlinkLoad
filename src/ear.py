import numpy as np

# EAR landmark indices (MediaPipe Standard)
L_HORIZONTAL = [362, 263]
L_VERTICAL = [(385, 380), (387, 373)]
R_HORIZONTAL = [33, 133]
R_VERTICAL = [(160, 144), (158, 153)]

def calculate_ear(landmarks, horizontal_indices, vertical_indices, w, h):
    """
    Calculate Eye Aspect Ratio (EAR) using Euclidean distance between landmarks.
    
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Args:
        landmarks: MediaPipe face landmarks object.
        horizontal_indices: List of indices for horizontal eye corners.
        vertical_indices: List of tuples for vertical eye pairs.
        w: Frame width.
        h: Frame height.
        
    Returns:
        float: Calculated EAR value.
    """
    try:
        # Extract pixel coordinates for horizontal landmarks
        horiz_pts = [np.array([landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]) for idx in horizontal_indices]
        
        # Extract pixel coordinates for vertical pairs
        vert_pts = []
        for v1, v2 in vertical_indices:
            v1_pt = np.array([landmarks.landmark[v1].x * w, landmarks.landmark[v1].y * h])
            v2_pt = np.array([landmarks.landmark[v2].x * w, landmarks.landmark[v2].y * h])
            vert_pts.append((v1_pt, v2_pt))

        # Calculate vertical Euclidean distances
        v_dist1 = np.linalg.norm(vert_pts[0][0] - vert_pts[0][1])
        v_dist2 = np.linalg.norm(vert_pts[1][0] - vert_pts[1][1])

        # Calculate horizontal Euclidean distance
        h_dist = np.linalg.norm(horiz_pts[0] - horiz_pts[1])

        # Apply EAR formula
        ear = (v_dist1 + v_dist2) / (2.0 * h_dist)
        return ear
    except Exception:
        return 0.0
