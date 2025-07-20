# Kaggle CMI - Detect Behavior with Sensor Data Competition

## Overview

**Competition URL**: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data

**Challenge**: Use sensor data to differentiate between body-focused repetitive behaviors (BFRBs) like hair pulling from everyday gestures like adjusting glasses.

**Goal**: Develop a predictive model that distinguishes BFRB-like and non-BFRB-like activity using data from a wrist-worn device with multiple sensor types.

**Impact**: Improve design and accuracy of wearable BFRB-detection devices for mental health treatment.

## Quick Facts

- **Prize Pool**: $50,000 total
- **Timeline**: May 29 - September 2, 2025
- **Test Set**: ~3,500 sequences (half IMU-only, half all sensors)
- **Evaluation**: Average of binary F1 and macro F1 scores
- **Submission**: Python evaluation API (notebook-based)

## Description
Body-focused repetitive behaviors (BFRBs), such as hair pulling, skin picking, and nail biting, are self-directed habits involving repetitive actions that, when frequent or intense, can cause physical harm and psychosocial challenges. These behaviors are commonly seen in anxiety disorders and obsessive-compulsive disorder (OCD), thus representing key indicators of mental health challenges.

hand

To investigate BFRBs, the Child Mind Institute has developed a wrist-worn device, Helios, designed to detect these behaviors. While many commercially available devices contain Inertial Measurement Units (IMUs) to measure rotation and motion, the Helios watch integrates additional sensors, including 5 thermopiles (for detecting body heat) and 5 time-of-flight sensors (for detecting proximity). See the figure to the right for the placement of these sensors on the Helios device.

We conducted a research study to test the added value of these additional sensors for detecting BFRB-like movements. In the study, participants performed series of repeated gestures while wearing the Helios device:

They began a transition from “rest” position and moved their hand to the appropriate location (Transition);

They followed this with a short pause wherein they did nothing (Pause); and

Finally they performed a gesture from either the BFRB-like or non-BFRB-like category of movements (Gesture; see Table below).

Each participant performed 18 unique gestures (8 BFRB-like gestures and 10 non-BFRB-like gestures) in at least 1 of 4 different body-positions (sitting, sitting leaning forward with their non-dominant arm resting on their leg, lying on their back, and lying on their side). These gestures are detailed in the table below, along with a video of the gesture.

| BFRB-Like Gesture (Target Gesture) | Video Example |
| --- | --- |
| Above ear - Pull hair	| [Sitting](https://www.youtube.com/watch?v=WJVFUSJm4As&list=PL_qYbUpkYMRUJSsRnfl0UpyCCV4RspUmh&index=12) |
| Forehead - Pull hairline | [Sitting leaning forward](https://www.youtube.com/watch?v=Se2uamhyNAk&list=PL_qYbUpkYMRUJSsRnfl0UpyCCV4RspUmh&index=18) |
| Forehead - Scratch | [Sitting](https://www.youtube.com/watch?v=WJVFUSJm4As&list=PL_qYbUpkYMRUJSsRnfl0UpyCCV4RspUmh&index=12) |
| Eyebrow - Pull hair | [Sitting](https://www.youtube.com/watch?v=WJVFUSJm4As&list=PL_qYbUpkYMRUJSsRnfl0UpyCCV4RspUmh&index=12) |
| Eyelash - Pull hair | [Sitting](https://www.youtube.com/watch?v=WJVFUSJm4As&list=PL_qYbUpkYMRUJSsRnfl0UpyCCV4RspUmh&index=12) |
| Neck - Pinch skin | [Sitting](https://www.youtube.com/watch?v=WJVFUSJm4As&list=PL_qYbUpkYMRUJSsRnfl0UpyCCV4RspUmh&index=12) |
| Neck - Scratch | [Sitting](https://www.youtube.com/watch?v=WJVFUSJm4As&list=PL_qYbUpkYMRUJSsRnfl0UpyCCV4RspUmh&index=12) |
| Cheek - Pinch skin | [Sitting](https://www.youtube.com/watch?v=WJVFUSJm4As&list=PL_qYbUpkYMRUJSsRnfl0UpyCCV4RspUmh&index=12), [Sitting leaning forward](https://www.youtube.com/watch?v=Se2uamhyNAk&list=PL_qYbUpkYMRUJSsRnfl0UpyCCV4RspUmh&index=18), [Lying on back](https://www.youtube.com/watch?v=3A_cu7Se_Uw&list=PL_qYbUpkYMRUJSsRnfl0UpyCCV4RspUmh&index=18), [Lying on side](https://www.youtube.com/watch?v=hjqrw1XfsIw&list=PL_qYbUpkYMRUJSsRnfl0UpyCCV4RspUmh&index=19) |


| Non-BFRB-Like Gesture (Non-Target Gesture) | Video Example |
| --- | --- |
| Drink from bottle/cup | Sitting |
| Glasses on/off | Sitting |
| Pull air toward your face | Sitting |
| Pinch knee/leg skin | Sitting leaning forward |
| Scratch knee/leg skin | Sitting leaning forward |
| Write name on leg | Sitting leaning forward |
| Text on phone | Sitting |
| Feel around in tray and pull out an object | Sitting |
| Write name in air | Sitting |
| Wave hello | Sitting |


This competition challenges you to develop a predictive model capable of distinguishing (1) BFRB-like gestures from non-BFRB-like gestures and (2) the specific type of BFRB-like gesture. Critically, when your model is evaluated, half of the test set will include only data from the IMU, while the other half will include all of the sensors on the Helios device (IMU, thermopiles, and time-of-flight sensors).

Your solutions will have direct real-world impact, as the insights gained will inform design decisions about sensor selection — specifically whether the added expense and complexity of thermopile and time-of-flight sensors is justified by significant improvements in BFRB detection accuracy compared to an IMU alone. By helping us determine the added value of these thermopiles and time-of-flight sensors, your work will guide the development of better tools for detection and treatment of BFRBs.

Relevant articles:
Garey, J. (2025). What Is Excoriation, or Skin-Picking? Child Mind Institute. https://childmind.org/article/excoriation-or-skin-picking/

Martinelli, K. (2025). What is Trichotillomania? Child Mind Institute. https://childmind.org/article/what-is-trichotillomania/

# Evaluation
The evaluation metric for this contest is a version of macro F1 that equally weights two components:

Binary F1 on whether the gesture is one of the target or non-target types.
Macro F1 on gesture, where all non-target sequences are collapsed into a single non_target class
The final score is the average of the binary F1 and the macro F1 scores.

If your submission includes a gesture value not found in the train set your submission will trigger an error.

## Submission File
You must submit to this competition using the provided evaluation API, which ensures that models perform inference on a single sequence at a time. For each sequence_id in the test set, you must predict the corresponding gesture.

# Timeline
May 29, 2025 - Start Date.

August 26, 2025 - Entry Deadline. You must accept the competition rules before this date in order to compete.

August 26, 2025 - Team Merger Deadline. This is the last day participants may join or merge teams.

September 2, 2025 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

# Prizes

1st Place - $ 15,000
2nd Place - $ 10,000
3rd Place - $ 8,000
4th Place - $ 7,000
5th Place - $ 5,000
6th Place - $ 5,000

# Acknowledgements

The data used for this competition was provided in collaboration with the Healthy Brain Network, a landmark mental health study based in New York City that will help children around the world. In the Healthy Brain Network, families, community leaders, and supporters are partnering with the Child Mind Institute to unlock the secrets of the developing brain. Additional study participants were recruited from Child Mind Institute’s staff and community, and we are grateful for their collaboration. In addition to the generous support provided by the Kaggle team, financial support has been provided by the California Department of Health Care Services (DHCS) as part of the Children and Youth Behavioral Health Initiative (CYBHI).

# Code Requirements

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 9 hours run-time
GPU Notebook <= 9 hours run-time
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

# About the Child Mind Institute and the Healthy Brain Network

The Child Mind Institute (CMI) is the leading independent nonprofit in children’s mental health providing gold-standard, evidence-based care, delivering educational resources to millions of families each year, training educators in underserved communities, and developing open science initiatives and tomorrow’s breakthrough treatments.
The Healthy Brain Network (HBN) is a community-based research initiative of the Child Mind Institute. We provide no-cost, study-related mental health and learning evaluations to children ages 5–21 and connect families with community resources. We are collecting the information needed to find brain and body characteristics that are associated with mental health and learning disorders. The Healthy Brain Network stores and openly shares de-identified data about psychiatric, behavioral, cognitive, and lifestyle (e.g., fitness, diet) phenotypes, as well as multimodal brain imaging (MRI), electroencephalography (EEG), digital voice and video recordings, genetics, and actigraphy.

# Dataset Description
In this competition you will use sensor data to classify body-focused repetitive behaviors (BFRBs) and other gestures.

This dataset contains sensor recordings taken while participants performed 8 BFRB-like gestures and 10 non-BFRB-like gestures while wearing the Helios device on the wrist of their dominant arm. The Helios device contains three sensor types:

1x Inertial Measurement Unit (IMU; BNO080/BNO085): An integrated sensor that combines accelerometer, gyroscope, and magnetometer measurements with onboard processing to provide orientation and motion data.
5x Thermopile Sensor (MLX90632): A non-contact temperature sensor that measures infrared radiation.
5x Time-of-Flight Sensor (VL53L7CX): A sensor that measures distance by detecting how long it takes for emitted infrared light to bounce back from objects.
You must submit to this competition using the provided Python evaluation API, which serves test set data one sequence at a time. To use the API, follow the example in this notebook.

Expect approximately 3,500 sequences in the hidden test set.
Half of the hidden-test sequences are recorded with IMU only; the thermopile (thm_) and time-of-flight (tof__v*) columns are still present but contain null values for those sequences.
This will allow us to determine whether adding the time-of-flight and thermopile sensors improves our ability to detect BFRBs. Note also that there is known sensor communication failure in this dataset, resulting in missing data from some sensors in some sequences.

## Files

### [train/test].csv

* row_id
* sequence_id - An ID for the batch of sensor data. Each sequence includes one Transition, one Pause, and one Gesture.
* sequence_type - If the gesture is a target or non-target type. Train only.
* sequence_counter - A counter of the row within each sequence.
* subject - A unique ID for the subject who provided the data.
* gesture - The target column. Description of sequence Gesture. Train only.
* orientation - Description of the subject's orientation during the sequence. Train only.
* behavior - A description of the subject's behavior during the current phase of the sequence.
* acc_[x/y/z] - Measure linear acceleration along three axes in meters per second squared from the IMU sensor.
* rot_[w/x/y/z] - Orientation data which combines information from the IMU's gyroscope, accelerometer, and magnetometer to describe the device's orientation in 3D space.
* thm_[1-5] - There are five thermopile sensors on the watch which record temperature in degrees Celsius. Note that the index/number for each corresponds to the index in the photo on the Overview tab.
* tof_[1-5]_v[0-63] - There are five time-of-flight sensors on the watch that measure distance. In the dataset, the 0th pixel for the first time-of-flight sensor can be found with column name tof_1_v0, whereas the final pixel in the grid can be found under column tof_1_v63. This data is collected row-wise, where the first pixel could be considered in the top-left of the grid, with the second to its right, ultimately wrapping so the final value is in the bottom right (see image above). The particular time-of-flight sensor is denoted by the number at the start of the column name (e.g., 1_v0 is the first pixel for the first time-of-flight sensor while 5_v0 is the first pixel for the fifth time-of-flight sensor). If there is no sensor response (e.g., if there is no nearby object causing a signal reflection), a -1 is present in this field. Units are uncalibrated sensor values in the range 0-254. Each sensor contains 64 pixels arranged in an 8x8 grid, visualized in the figure below.

### [train/test]_demographics.csv

These tabular files contain demographic and physical characteristics of the participants.

* subject
* adult_child: Indicates whether the participant is a child (0) or an adult (1). Adults are defined as individuals aged 18 years or older.
* age: Participant's age in years at time of data collection.
* sex: Participants sex assigned at birth, 0= female, 1 = male.
* handedness: Dominant hand used by the participant, 0 = left-handed, 1 = right-handed.
* height_cm: Height of the participant in centimeters.
* shoulder_to_wrist_cm: Distance from shoulder to wrist in centimeters.
* elbow_to_wrist_cm: Distance from elbow to wrist in centimeters.

### sample_submission.csv

* sequence_id
* gesture

### kaggle_evaluation/

The files that implement the evaluation API. You can run the API locally on a Unix machine for testing purposes.
