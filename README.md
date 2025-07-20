# Kaggle CMI - Detect Behavior with Sensor Data

This repository contains code for the Kaggle competition "CMI - Detect Behavior with Sensor Data":
https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data

## Competition Overview

This competition challenges participants to develop models that can:
1. **Distinguish BFRB-like gestures from non-BFRB-like gestures** using wrist-worn sensor data
2. **Classify specific types of BFRB-like gestures** with high accuracy

### About BFRBs
Body-focused repetitive behaviors (BFRBs) like hair pulling, skin picking, and nail biting are self-directed habits that can cause physical harm and psychological distress. They're commonly associated with anxiety disorders and OCD, making them important indicators of mental health challenges.

### The Challenge
- **Dataset**: Sensor data from Helios wrist-worn device with IMU, thermopile, and time-of-flight sensors
- **Task**: Binary classification (BFRB vs non-BFRB) + multi-class gesture classification
- **Evaluation**: Average of binary F1 and macro F1 scores
- **Prize Pool**: $50,000 total (1st place: $15,000)
- **Deadline**: September 2, 2025

### Sensor Data
- **IMU**: Accelerometer, gyroscope, magnetometer for motion/orientation
- **5x Thermopiles**: Non-contact temperature sensors
- **5x Time-of-Flight**: Distance measurement sensors (8x8 pixel grids)
- **Test Split**: Half IMU-only, half all sensors (to evaluate sensor value)

### Technical Details
- 8 BFRB-like gestures (hair pulling, skin picking, scratching)
- 10 non-BFRB-like gestures (drinking, texting, waving)
- 4 body positions (sitting, leaning, lying back/side)
- ~3,500 test sequences
- Submission via Python evaluation API

## Quick Start

1. Install dependencies: `make install`
2. Start notebook development: `jupyter lab src/notebook/`
3. Use CLI commands: `ml-cli hello`, `ml-cli train`, `ml-cli predict`
4. Review competition details: See `Kaggle-Competition.md` for full competition description

See CLAUDE.md for detailed development commands and architecture information.

## Development Stack

This project uses a modern ML development environment:
- **Python Package Management**: uv for fast dependency resolution
- **Notebook Development**: Jupyter with jupytext sync (ipynb ↔ py:percent)
- **Code Quality**: ruff (formatting/linting), mypy/ty (type checking)
- **ML Stack**: PyTorch, torchvision, scikit-learn, pandas, numpy
- **CLI Interface**: typer for command-line tools
