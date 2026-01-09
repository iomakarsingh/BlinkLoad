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

## Summary of Findings
- [ ] No crashes or freezes observed across all tests.
- [ ] Tracking stability maintained during normal movement.
