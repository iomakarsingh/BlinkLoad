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
        self.ear_open_events = [] # Stores {"timestamp": t, "ear": e}
        self.start_timestamp = None

    def update(self, left_ear, right_ear, current_time=None):
        """
        Processes current EAR values and updates blink count and stability metrics.
        """
        if current_time is None:
            current_time = time.time()
            
        is_blinking_state = False
        avg_ear = (left_ear + right_ear) / 2.0
        
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

            # Collect EAR for stability index (ESI) only when eyes are fully open
            # We check if counter is 0 to ensure we aren't in the middle of a short jitter
            if avg_ear > self.threshold:
                self.ear_open_events.append({
                    "timestamp": current_time, 
                    "ear": avg_ear
                })
            
        return is_blinking_state

    def get_metrics(self, current_time=None):
        """
        Computes core and advanced blink features from the current sliding window.
        
        Returns:
            dict: { "br": float, "mbd": float, "bdv": float, "ibi": float, "bbi": float, "esi": float }
        """
        if current_time is None:
            current_time = time.time()

        # 1. Housekeeping: Remove expired events
        self.blink_events = [e for e in self.blink_events if current_time - e["timestamp"] <= WINDOW_SIZE_SEC]
        self.ear_open_events = [e for e in self.ear_open_events if current_time - e["timestamp"] <= WINDOW_SIZE_SEC]
        
        count = len(self.blink_events)
        metrics = {
            "br": 0.0, "mbd": 0.0, "bdv": 0.0, "ibi": 0.0, "bbi": 0.0, "esi": 0.0
        }

        # 2. EAR Stability Index (ESI) - Std Dev of open EAR values
        if len(self.ear_open_events) > 1:
            ears = [e["ear"] for e in self.ear_open_events]
            mean_ear = sum(ears) / len(ears)
            variance_ear = sum((e - mean_ear) ** 2 for e in ears) / len(ears)
            metrics["esi"] = variance_ear ** 0.5
        
        if count == 0:
            return metrics

        # 3. Blink Rate (BR) - blinks per minute
        metrics["br"] = count / (WINDOW_SIZE_SEC / 60.0)

        # 4. Mean Blink Duration (MBD)
        durations = [e["duration"] for e in self.blink_events]
        metrics["mbd"] = sum(durations) / count

        # 5. Blink Duration Variance (BDV)
        if count > 1:
            mean_dur = metrics["mbd"]
            metrics["bdv"] = sum((d - mean_dur) ** 2 for d in durations) / count
        else:
            metrics["bdv"] = 0.0

        # 6. Inter-Blink Interval (IBI)
        if count > 1:
            intervals = []
            for i in range(1, count):
                interval = self.blink_events[i]["timestamp"] - self.blink_events[i-1]["timestamp"]
                intervals.append(interval)
            metrics["ibi"] = sum(intervals) / len(intervals)
            
            # 7. Blink Burst Index (BBI)
            # Burst = >= 2 blinks within 2 seconds
            bursts = 0
            in_burst = False
            for i in range(1, count):
                if self.blink_events[i]["timestamp"] - self.blink_events[i-1]["timestamp"] <= 2.0:
                    if not in_burst:
                        bursts += 1
                        in_burst = True
                else:
                    in_burst = False
            metrics["bbi"] = bursts / count if count > 0 else 0.0
        else:
            metrics["ibi"] = 0.0
            metrics["bbi"] = 0.0

        return metrics

    def get_window_count(self, current_time=None):
        """
        Removes expired timestamps and returns the count of blinks in the current window.
        """
        if current_time is None:
            current_time = time.time()
            
        # Refresh window
        self.blink_events = [e for e in self.blink_events if current_time - e["timestamp"] <= WINDOW_SIZE_SEC]
        return len(self.blink_events)
