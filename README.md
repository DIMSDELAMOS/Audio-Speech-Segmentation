# 🎙️ Audio Speech Segmentation (University of Piraeus Project)

An end-to-end Machine Learning pipeline developed as an official **University of Piraeus Project** for the *Speech and Audio Signal Processing* course (Academic Year 2025-2026). 

The system automatically segments audio recordings, distinguishing between active human speech (foreground) and environmental noise (background) at the frame level.

## 🚀 Key Features
* **Feature Extraction:** Processes audio into 25ms frames extracting 13 MFCCs and RMS Energy using `librosa`.
* **Machine Learning Models:** Custom implementations of K-Nearest Neighbors (K-NN) and a Multilayer Perceptron (MLP) neural network using `scikit-learn`.
* **High Accuracy:** Achieved **>99.5% accuracy** on a balanced subset of ~270,000 frames using the MLP classifier.
* **Post-Processing:** Applies a Median Filter (31-frame window) to smooth out instantaneous misclassifications and generate continuous audio segments.

## 🛠️ Tech Stack
* **Python**
* **Librosa** (Audio Processing)
* **Scikit-learn** (Machine Learning)
* **Pandas & NumPy** (Data Manipulation)
* **SciPy** (Signal Post-processing)

## 📂 Project Structure
* `main.py`: The core script for feature extraction, model training, subsampling, and test evaluation.
* `results.csv`: The final output demonstrating the timestamped segments (start, end, class) of a test audio file.
* `Report.pdf`: The detailed technical documentation and methodology (in Greek).

---
*This repository is part of an academic assignment for the Department of Informatics, University of Piraeus.*
