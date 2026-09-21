# OIST-Bearing-Fault-Diagnosis

## DSP-Based Bearing Fault Diagnosis Using Vibration Signals and Machine Learning

A research-oriented study of bearing-condition diagnosis using vibration signals, digital signal processing (DSP), classical machine learning, and a lightweight 1D convolutional neural network (CNN), with emphasis on leakage-free evaluation, robustness, compact representations, and edge-deployment considerations.

---

## Project Overview

This project investigates the use of vibration-signal analysis and machine learning for automated bearing-condition diagnosis.

The workflow combines:

- Digital signal processing (DSP)
- Time-domain statistical features
- Frequency-domain spectral features
- Support Vector Machine (SVM)
- Random Forest (RF)
- 1D Convolutional Neural Network (CNN)
- Recording-aware train/test splitting
- Grouped cross-validation
- Noise robustness analysis
- Representation-memory analysis
- Software-based inference profiling

The objective is not only to measure classification performance, but also to study how signal representation, validation methodology, robustness, and computational requirements affect a research-oriented diagnostic system.

---

## Research Question

The study investigates:

> How effectively can vibration signals be used for bearing-condition diagnosis when comparing compact DSP-based machine-learning representations with a raw-waveform 1D CNN under leakage-free evaluation?

The experiments specifically examine:

1. Performance of classical machine-learning models using DSP features.
2. Performance of a raw-waveform 1D CNN.
3. The contribution of time-domain and frequency-domain features.
4. Robustness to controlled additive Gaussian noise.
5. Memory requirements of compact DSP features versus raw waveform windows.
6. Software-based computational requirements relevant to edge deployment.

---

## Key Experimental Results

All reported test metrics below are calculated on the held-out test recordings.

| Model | Accuracy | Weighted F1 | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Tuned SVM | 0.8267 | 0.8308 | 0.8260 | 0.8228 |
| Tuned Random Forest | 0.8990 | 0.8996 | 0.9284 | 0.9158 |
| 1D CNN | **0.9657** | **0.9649** | **0.9526** | **0.9599** |

In this experimental setup, the 1D CNN achieved a measured weighted F1 of **0.9649** on the held-out recordings.

The classical machine-learning models provide compact DSP-based alternatives, with the Random Forest achieving a weighted F1 of **0.8996**.

These results describe the evaluated dataset and experimental configuration and should not be interpreted as universal performance across other datasets, machines, sensors, or operating conditions.

---

## Dataset

The project uses the **Case Western Reserve University (CWRU) bearing vibration dataset**.

Dataset source:

`https://github.com/XiongMeijing/CWRU-1`

The dataset is downloaded automatically by the notebook and is **not stored inside this repository**.

### Final Experimental Dataset

- Independent recordings: **133**
- Feature windows: **3,912**
- Raw waveform windows: **3,912**
- Sampling frequency: **12 kHz**
- Window length: **12,000 samples**
- Window duration: **1 second**
- Window overlap: **50%**
- Maximum windows per file: **100**
- Number of classes: **4**

### Condition Labels

The experiment uses the following directory-derived condition labels:

- `12k_DE`
- `12k_FE`
- `48k_DE`
- `Normal`

The labels are taken from the dataset directory structure used by the experiment.

---

## Signal Processing Pipeline

The vibration signals are processed using the following pipeline:

```text
Raw Vibration Signal
        ↓
Butterworth Low-Pass Filtering
        ↓
Windowing
        ↓
Time-Domain Features
        +
Frequency-Domain Features
        ↓
DSP Feature Vector
        ↓
SVM / Random Forest
