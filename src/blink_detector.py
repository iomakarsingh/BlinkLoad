# Blink Detection Constants
EAR_THRESHOLD = 0.22      # Threshold below which eyes are considered closed
EAR_CONSEC_FRAMES = 3     # Minimum frames eyes must be closed to count as a blink

# Physiological Constraints (ms)
MIN_BLINK_DURATION = 100
MAX_BLINK_DURATION = 400
MAX_CLOSED_TIME = 500     # Threshold to discard deliberate closures / look-downs

class BlinkDetector:
    """
    State machine for detecting blinks based on EAR values.
    Ensures both eyes are closed for a valid blink and counts on closed -> open transition.
    Implements hard filtering to reject non-blink eye closures.
    """
    def __init__(self, threshold=EAR_THRESHOLD, consec_frames=EAR_CONSEC_FRAMES):
        self.threshold = threshold
        self.consec_frames = consec_frames
        self.counter = 0
        self.total_blinks = 0
        self.blink_events = []  # Store dicts: {"start": t1, "end": t2, "duration": ms}
        self.start_timestamp = None

    def update(self, left_ear, right_ear, current_time):
        """
        Processes current EAR values and updates blink count and timestamps.
        
        Args:
            left_ear (float): EAR of the left eye.
            right_ear (float): EAR of the right eye.
            current_time (float): Current timestamp from time.time().
            
        Returns:
            bool: True if currently in a 'closed' state.
        """
        is_blinking_state = False
        
        # Robust check: both eyes must be below threshold
        if left_ear < self.threshold and right_ear < self.threshold:
            self.counter += 1
            is_blinking_state = True
            
            # Record start time on first frame eyes go below threshold
            if self.counter == 1:
                self.start_timestamp = current_time
            
            # Guard: If eyes remain closed longer than MAX_CLOSED_TIME, reset and discard
            current_duration = (current_time - self.start_timestamp) * 1000
            if current_duration > MAX_CLOSED_TIME:
                if self.counter > 0:
                    print(f"[Discarded closure: >{MAX_CLOSED_TIME}ms]")
                self.counter = 0
                self.start_timestamp = None
                is_blinking_state = False # Reset state
        else:
            # Transition from closed to open
            if self.counter >= self.consec_frames:
                end_timestamp = current_time
                duration = (end_timestamp - self.start_timestamp) * 1000  # in ms
                
                # Rule: Hard filter before counting
                if MIN_BLINK_DURATION <= duration <= MAX_BLINK_DURATION:
                    self.total_blinks += 1
                    event = {
                        "start": self.start_timestamp,
                        "end": end_timestamp,
                        "duration": duration
                    }
                    self.blink_events.append(event)
                    print(f"Blink {self.total_blinks}: {duration:.1f}ms (ACCEPTED)")
                else:
                    print(f"[Discarded closure: {duration:.1f}ms]")
                
            self.counter = 0
            self.start_timestamp = None
            is_blinking_state = False
            
        return is_blinking_state
