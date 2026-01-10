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

    def get_features(self, current_time=None):
        """
        Computes core blink features for the current window.
        
        Returns:
            dict: { "br": blink_rate, "mbd": mean_duration, "bdv": variance, "ibi": inter_blink_interval }
        """
        if current_time is None:
            current_time = time.time()
            
        # 1. Cleanup window (Strict: only accepted events)
        self.blink_events = [e for e in self.blink_events if current_time - e["timestamp"] <= WINDOW_SIZE_SEC]
        
        count = len(self.blink_events)
        durations = [e["duration"] for e in self.blink_events]
        
        # Default empty features
        features = {
            "br": 0.0,
            "mbd": 0.0,
            "bdv": 0.0,
            "ibi": 0.0
        }
        
        if count > 0:
            # 1. Blink Rate (BR) - BPM
            # window_minutes = WINDOW_SIZE_SEC / 60
            window_minutes = WINDOW_SIZE_SEC / 60.0
            features["br"] = count / window_minutes
            
            # 2. Mean Blink Duration (MBD) - ms
            features["mbd"] = sum(durations) / count
            
            # 3. Blink Duration Variance (BDV) - ms^2
            if count > 1:
                mean = features["mbd"]
                features["bdv"] = sum((d - mean) ** 2 for d in durations) / count
            else:
                features["bdv"] = 0.0
                
            # 4. Inter-Blink Interval (IBI) - seconds
            # Time between now and the last accepted blink
            last_blink_time = self.blink_events[-1]["timestamp"]
            features["ibi"] = current_time - last_blink_time
        
        return features

    def get_window_count(self, current_time=None):
        """
        Returns the count of blinks in the current window.
        """
        if current_time is None:
            current_time = time.time()
        self.blink_events = [e for e in self.blink_events if current_time - e["timestamp"] <= WINDOW_SIZE_SEC]
        return len(self.blink_events)
