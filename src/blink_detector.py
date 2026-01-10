import time

# Blink Detection Constants
EAR_THRESHOLD = 0.22      # Threshold below which eyes are considered closed
EAR_CONSEC_FRAMES = 3     # Minimum frames eyes must be closed to count as a blink
WINDOW_SIZE_SEC = 30      # Time window for aggregate metrics

class BlinkDetector:
    """
    State machine for detecting blinks based on EAR values.
    Ensures both eyes are closed for a valid blink and counts on closed -> open transition.
    Maintains a rolling buffer of blink timestamps for window-based metrics.
    """
    def __init__(self, threshold=EAR_THRESHOLD, consec_frames=EAR_CONSEC_FRAMES):
        self.threshold = threshold
        self.consec_frames = consec_frames
        self.counter = 0
        self.total_blinks = 0
        self.blink_timestamps = []

    def update(self, left_ear, right_ear, current_time=None):
        """
        Processes current EAR values and updates blink count.
        
        Args:
            left_ear (float): EAR of the left eye.
            right_ear (float): EAR of the right eye.
            current_time (float, optional): Current timestamp. If None, uses time.time().
            
        Returns:
            bool: True if currently in a 'closed' state.
        """
        if current_time is None:
            current_time = time.time()
            
        is_blinking_state = False
        
        # Robust check: both eyes must be below threshold
        if left_ear < self.threshold and right_ear < self.threshold:
            self.counter += 1
            is_blinking_state = True
        else:
            # Transition from closed to open
            # If eyes were closed for at least the required consecutive frames
            if self.counter >= self.consec_frames:
                self.total_blinks += 1
                self.blink_timestamps.append(current_time)
            self.counter = 0
            is_blinking_state = False
            
        return is_blinking_state

    def get_window_count(self, current_time=None):
        """
        Removes expired timestamps and returns the count of blinks in the current window.
        
        Args:
            current_time (float, optional): Current timestamp. If None, uses time.time().
            
        Returns:
            int: Number of blinks in the last WINDOW_SIZE_SEC seconds.
        """
        if current_time is None:
            current_time = time.time()
            
        # Remove timestamps older than WINDOW_SIZE_SEC
        self.blink_timestamps = [t for t in self.blink_timestamps if current_time - t <= WINDOW_SIZE_SEC]
        return len(self.blink_timestamps)
