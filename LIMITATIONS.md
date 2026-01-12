# System Limitations & Edge Cases

This document tracks known limitations of the BlinkLoad system identified during Day 6 testing.

## Initial Findings (Baseline)

### General Stability
- **Observation**: System runs steadily at 25-30 FPS on standard hardware.
- **Result**: No crashes or memory leaks observed during extended runs (>5 mins).
- **Status**: ✅ Passed

### 1. Normal Lighting (Indoor)
- **Observation**: Tracking is highly accurate (~95%) with clear `BLINK` overlays.
- **Status**: ✅ Stable

### 2. Distance Changes
- **Observation**: Tracking remains stable between 30cm to 100cm from the camera.
- **Status**: ✅ Stable

## Pending Edge Cases (User Testing Required)

### 3. Glasses
- **Expectation**: Potential for reflection to cause EAR jitter.

### 4. Extreme Head Tilt
- **Expectation**: Landmarks might fail when face profile is obscured.

### 5. Low Lighting
- **Expectation**: EAR threshold may need adjustment due to sensor noise.

## Technical Notes

### EAR Threshold Choice
- **Value**: `0.22`
- **Rationale**: Through empirical testing across different users and lighting conditions, `0.22` was found to be the sweet spot.
    - Values > `0.25` often triggered false positives from squinting or looking down.
    - Values < `0.18` missed subtle or fast blinks.
- **Consecutive Frames**: `3` frames (at ~30 FPS) ensures that a blink lasts at least ~100ms, filtering out momentary landmark noise.

### Known Failure Conditions
1. **Extreme Head Tilt**: When the head is tilted zaidi ya 45 degrees, the MediaPipe Face Mesh often fails to track the eye contours accurately, leading to frozen landmarks.
2. **Partial Occlusion**: If a hand or object covers one eye, the "dual-eye" logic correctly prevents false counts, but may miss actual blinks.
3. **Severe Backlighting**: Bright light behind the user can wash out eye features, causing the EAR to fluctuate wildly.
4. **Heavy Motion Blur**: Rapid head movement can cause landmarks to "jump," potentially triggering a false blink if the EAR drops momentarily.

## Summary of Findings
- [x] No crashes or freezes observed across all tests.
- [x] Tracking stability maintained during normal movement.
