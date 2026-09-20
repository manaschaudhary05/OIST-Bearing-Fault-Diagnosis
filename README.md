# OIST-Bearing-Fault-Diagnosis

## DSP-Based Bearing Fault Diagnosis Using Vibration Signals and Machine Learning

A research-oriented project investigating **bearing fault diagnosis from vibration signals** using digital signal processing (DSP), machine learning, and edge-deployment considerations.

The project explores how compact DSP-derived representations can be used for reliable condition classification while reducing the computational and memory requirements associated with raw vibration signals.

---

## Research Objective

The objective of this project is to develop and evaluate a reproducible pipeline for vibration-based bearing fault diagnosis.

The study focuses on four main questions:

1. Can DSP-derived vibration features distinguish different bearing conditions?
2. How do classical machine-learning models compare with a 1D CNN?
3. How robust are the learned representations to additive measurement noise?
4. How does feature-based processing affect memory and computational requirements for edge deployment?

---

## System Pipeline

```text
Vibration Signal
       │
       ▼
Signal Preprocessing
       │
       ▼
Butterworth Low-Pass Filtering
       │
       ▼
Windowing
       │
       ▼
DSP Feature Extraction
       │
       ├── Time-Domain Features
       │
       └── Frequency-Domain Features
       │
       ▼
Machine Learning
       │
       ├── SVM
       ├── Random Forest
       └── 1D CNN
       │
       ▼
Fault / Condition Classification
       │
       ▼
Robustness + Edge Deployment Analysis
