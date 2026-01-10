import time

# Blink Detection Constants
EAR_THRESHOLD = 0.22      # Threshold below which eyes are considered closed
EAR_CONSEC_FRAMES = 3     # Minimum frames eyes must be closed to count as a blink
WINDOW_SIZE_SEC = 30      # Time window for aggregate metrics

# Physiological Constraints (ms)
MIN_BLINK_DURATION = 100
MAX_BLINK_DURATION = 400

class BlinkDetector:
    """
    State machine for detecting blinks based on EAR values.
    Ensures both eyes are closed for a valid blink and counts on closed -> open transition.
    Maintains a rolling buffer of ACCEPTED blink events for window-based metrics.
    """
    def __init__(self, threshold=EAR_THRESHOLD, consec_frames=EAR_CONSEC_FRAMES):
        self.threshold = threshold
        self.consec_frames = consec_frames
        self.counter = 0
        self.total_blinks = 0
        self.blink_events = []  # Stores {"timestamp": t, "duration": d}
        self.start_timestamp = None

    def update(self, left_ear, right_ear, current_time=None):
        """
        Processes current EAR values and updates blink count.
        """
        if current_time is None:
            current_time = time.time()
            
        is_blinking_state = False
        
        # Robust check: both eyes must be below threshold
        if left_ear < self.threshold and right_ear < self.threshold:
            if self.counter == 0:
                self.start_timestamp = current_time
            self.counter += 1
            is_blinking_state = True
        else:
            # Transition from closed to open
            if self.counter >= self.consec_frames:
                duration = (current_time - self.start_timestamp) * 1000  # in ms
                
                # Strict Data Flow: Only accept physiological blinks
                if MIN_BLINK_DURATION <= duration <= MAX_BLINK_DURATION:
                    self.total_blinks += 1
                    self.blink_events.append({
                        "timestamp": current_time,
                        "duration": duration
                    })
                else:
                    print(f"[Discarded closure: {duration:.1f}ms]")
                    
            self.counter = 0
            self.start_timestamp = None
            is_blinking_state = False
            
        return is_blinking_state

    def get_window_count(self, current_time=None):
        """
        Removes expired timestamps and returns the count of blinks in the current window.
        """
        if current_time is None:
            current_time = time.time()
            
        # Remove events older than WINDOW_SIZE_SEC
        self.blink_events = [e for e in self.blink_events if current_time - e["timestamp"] <= WINDOW_SIZE_SEC]
        return len(self.blink_events)
