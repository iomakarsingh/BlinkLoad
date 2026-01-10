import time
import numpy as np

# Blink Detection Constants
EAR_THRESHOLD = 0.23      # Slightly increased for better sensitivity
EAR_CONSEC_FRAMES = 3     # Minimum frames eyes must be closed to count as a blink
WINDOW_SIZE_SEC = 30      # Time window for aggregate metrics

# Physiological Constraints (ms)
MIN_BLINK_DURATION = 70   # Lowered to capture 3-frame blinks at 30 FPS (~99ms)
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

    def _cleanup_window(self, current_time):
        """Internal helper to drop expired events."""
        self.blink_events = [e for e in self.blink_events if current_time - e["timestamp"] <= WINDOW_SIZE_SEC]

    def get_window_count(self, current_time=None):
        """
        Returns the count of blinks in the current window.
        """
        if current_time is None:
            current_time = time.time()
        self._cleanup_window(current_time)
        return len(self.blink_events)

    def get_metrics(self, current_time=None):
        """
        Computes core metrics on the current window.
        Returns: (blink_rate, mean_duration, variance, ibi)
        """
        if current_time is None:
            current_time = time.time()
            
        self._cleanup_window(current_time)
        events = self.blink_events
        n = len(events)
        
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        # 1. Blink Rate (Blinks per Minute)
        blink_rate = n / (WINDOW_SIZE_SEC / 60.0)
        
        # 2. Mean Blink Duration (ms)
        durations = [e["duration"] for e in events]
        mean_duration = np.mean(durations)
        
        # 3. Blink Duration Variance
        variance = np.var(durations) if n > 1 else 0.0
        
        # 4. Inter-Blink Interval (IBI)
        # Defined as the time since the last blink happened, for live display
        ibi = current_time - events[-1]["timestamp"]
        
        return blink_rate, mean_duration, variance, ibi
