# Blink Detection Constants
EAR_THRESHOLD = 0.22      # Threshold below which eyes are considered closed
EAR_CONSEC_FRAMES = 3     # Minimum frames eyes must be closed to count as a blink

class BlinkDetector:
    """
    State machine for detecting blinks based on EAR values.
    Ensures both eyes are closed for a valid blink and counts on closed -> open transition.
    """
    def __init__(self, threshold=EAR_THRESHOLD, consec_frames=EAR_CONSEC_FRAMES):
        self.threshold = threshold
        self.consec_frames = consec_frames
        self.counter = 0
        self.total_blinks = 0

    def update(self, left_ear, right_ear):
        """
        Processes current EAR values and updates blink count.
        
        Args:
            left_ear (float): EAR of the left eye.
            right_ear (float): EAR of the right eye.
            
        Returns:
            bool: True if currently in a 'closed' state.
        """
        # Robust check: both eyes must be below threshold
        if left_ear < self.threshold and right_ear < self.threshold:
            self.counter += 1
            is_blinking_state = True
        else:
            # If eyes were closed for at least the required consecutive frames
            if self.counter >= self.consec_frames:
                self.total_blinks += 1
            self.counter = 0
            is_blinking_state = False
            
        return is_blinking_state
