class BlinkDetector:
    """
    A state machine to detect blinks based on Eye Aspect Ratio (EAR) thresholds.
    """
    def __init__(self, threshold=0.22, consec_frames=3):
        """
        Initialize the detector.
        
        Args:
            threshold (float): EAR value below which the eyes are considered closed.
            consec_frames (int): Minimum number of consecutive frames of closure 
                                to register a valid blink.
        """
        self.threshold = threshold
        self.consec_frames = consec_frames
        
        self.counter = 0
        self.total_blinks = 0

    def update(self, left_ear, right_ear):
        """
        Update the state machine with the latest EAR values.
        A blink is only counted on the closed -> open transition to ensure robustness.
        Both eyes must be closed simultaneously to avoid false positives from winks.
        
        Args:
            left_ear (float): EAR for the left eye.
            right_ear (float): EAR for the right eye.
            
        Returns:
            bool: True if eyes are currently detected as closed (for visual feedback).
        """
        is_closed = left_ear < self.threshold and right_ear < self.threshold
        
        if is_closed:
            self.counter += 1
        else:
            # Check if eyes were closed long enough before opening
            if self.counter >= self.consec_frames:
                self.total_blinks += 1
            self.counter = 0
            
        return is_closed

    def reset_total(self):
        """Reset the total blink count."""
        self.total_blinks = 0
