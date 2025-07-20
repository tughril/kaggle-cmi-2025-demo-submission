# CLAUDE.md

This repository is focused exclusively on winning the Kaggle CMI 2025 competition. All development should be optimized for competition performance.

## Essential Commands

- `uv sync` - Install dependencies
- `make lint` - Code quality check
- `make sync` - Sync notebook with Python file

## Kaggle CMI 2025 Competition - BFRB Detection

**Objective**: Classify 18 body-focused repetitive behaviors using wrist sensor data

### Data Files
All data files are located in the `data/` directory:
- `train.csv` (574,945 rows) - Sensor time series with labels
- `test.csv` (107 rows sample) - Real test via API
- `*_demographics.csv` - Participant characteristics

### Target Classes
- **8 BFRB-like**: Hair pulling, skin picking, scratching (target behaviors)
- **10 Non-BFRB-like**: Drinking, texting, waving (control behaviors)

### Sensor Data (332 features/timestep)
- **IMU (7)**: `acc_x/y/z` (m/s²), `rot_w/x/y/z` (quaternion)
- **Temperature (5)**: `thm_1-5` (infrared °C)
- **Distance (320)**: `tof_1-5_v0-63` (5×8×8 ToF sensors, 0-254 or -1)

### Critical Constraints
**Test environment removes:**
- `behavior` (Transition/Pause/Gesture phases)
- `orientation` (sitting/lying positions)
- `sequence_type` (BFRB/non-BFRB labels)

**50% of test data**: IMU only (temperature/distance = null)

### Evaluation
**Score = (Binary F1 + Macro F1) / 2**
- **Binary F1**: BFRB vs non-BFRB detection
- **Macro F1**: 9-class (8 BFRB types + 1 combined non-BFRB)

### Key Strategy
1. Train without phase/orientation info (simulate test)
2. Handle missing sensors (temperature/distance nulls)
3. Optimize both binary detection and multi-class accuracy
4. Use demographics for individual calibration
