# Driver Drowsiness Detection Using Temporal Eye and Head Movement Analysis

A research-oriented driver drowsiness detection system that uses temporal eye and head movement patterns to classify a driver as **Drowsy** or **Not Drowsy**.

The system extracts four facial-movement features from video frames using **MediaPipe Face Landmarker**:

- EAR Mean
- Pitch
- Yaw
- Roll

These features are organized into temporal sequences and processed using an **LSTM neural network** to capture changes in driver behavior over time.

> **Important:** This project intentionally does **not** use mouth or yawning features.

---

## Project Overview

Driver drowsiness is a major road-safety concern because drowsiness develops as a temporal behavioral pattern rather than being reliably identifiable from a single image.

Instead of classifying individual frames independently, this project:

1. Processes video frames using MediaPipe.
2. Extracts eye and head movement features.
3. Validates feature quality.
4. Splits videos at the video level to prevent temporal leakage.
5. Constructs 30-frame temporal sequences.
6. Trains LSTM-based models.
7. Evaluates the models on an untouched test set.
8. Performs repeated video-level robustness evaluation.
9. Provides a real-time webcam inference prototype.

---

## Objectives

- Detect driver drowsiness using facial movement information.
- Use only eye and head movement features.
- Capture temporal behavior using sequential modeling.
- Prevent data leakage between training, validation and testing.
- Compare LSTM-based architectures.
- Evaluate generalization across different videos.
- Develop a real-time drowsiness detection prototype.

---

## System Architecture

```text
NITYMED Dataset
       |
       v
Annotation Validation
       |
       v
Video Frames
       |
       v
MediaPipe Face Landmarker
       |
       +----------------------+
       |                      |
       v                      v
Eye Movement             Head Movement
       |                      |
       v                      v
EAR Mean              Pitch / Yaw / Roll
       |                      |
       +----------+-----------+
                  |
                  v
         Feature Validation
                  |
                  v
       Video-Level Data Split
                  |
                  v
      Temporal Sequence Creation
                  |
                  v
               30 x 4
                  |
                  v
            LSTM Classifier
                  |
                  v
       Drowsiness Probability
                  |
            +-----+-----+
            |           |
            v           v
       Not Drowsy     Drowsy
```

---

## Dataset

This project uses the **NITYMED (Nighttime In-Car Driver Monitoring and Drowsiness)** dataset.

For this project:

- Total local videos: **44**
- Total annotated records: **53,331**
- Valid binary classification records: **47,834**
- Not Drowsy: **38,954**
- Drowsy: **8,880**
- Yawning annotations were ignored.
- Frame rate: **25 FPS**

### Label Mapping

| Original State | Project Label |
|---|---|
| Alert | 0 - Not Drowsy |
| Microsleep | 1 - Drowsy |
| Yawning | Ignored |

---

## Feature Extraction

Facial landmarks are detected using **MediaPipe Face Landmarker**.

Only four features are used by the final model:

### 1. EAR Mean

Eye Aspect Ratio (EAR) represents eye opening.

For an eye with six landmarks:

\[
EAR = \frac{||p_2-p_6|| + ||p_3-p_5||}{2||p_1-p_4||}
\]

EAR is calculated separately for the left and right eye:

\[
EAR_{mean} = \frac{EAR_{left} + EAR_{right}}{2}
\]

A lower EAR generally indicates reduced eye opening.

### 2. Pitch

Represents vertical head movement.

### 3. Yaw

Represents left/right head movement.

### 4. Roll

Represents sideways head rotation/tilting.

Each valid frame is represented as:

```text
[EAR Mean, Pitch, Yaw, Roll]
```

### No Mouth Features

This project intentionally excludes:

- Mouth landmarks
- Mouth opening
- Yawning detection
- Mouth aspect ratio

The model relies only on eye and head movement.

---

## Temporal Sequence Construction

The model does not classify individual frames.

Instead, **30 consecutive frames** are grouped into one temporal sequence.

The videos are recorded at **25 FPS**.

\[
30/25 = 1.2\text{ seconds}
\]

Each model input contains:

```text
30 frames x 4 features
```

Example:

```text
Frame 01 -> [EAR, Pitch, Yaw, Roll]
Frame 02 -> [EAR, Pitch, Yaw, Roll]
...
Frame 30 -> [EAR, Pitch, Yaw, Roll]
```

The LSTM processes this sequence to learn how eye and head movements evolve over time.

---

## Data Leakage Prevention

A major focus of the project is preventing temporal leakage.

The split is performed at the **video level before temporal sequence generation**.

```text
Videos
 |
 +-- Train Videos
 +-- Validation Videos
 +-- Test Videos
        |
        v
Generate sequences
```

Sequences are never created first and then randomly split.

### Final Video Split

| Split | Videos | Sequences |
|---|---:|---:|
| Train | 31 | 395 |
| Validation | 7 | 128 |
| Test | 6 | 179 |
| **Total** | **44** | **702** |

The official test set remains untouched during model development.

---

## Model Architecture

The final model is an LSTM-based binary classifier.

```text
Input: 30 x 4
        |
        v
LSTM - 64 units
        |
        v
Dropout - 0.30
        |
        v
Dense - 32 neurons
        |
       ReLU
        |
        v
Dropout - 0.20
        |
        v
Dense - 1 neuron
        |
      Sigmoid
        |
        v
Drowsiness Probability
```

### Configuration

- LSTM units: **64**
- Dense neurons: **32**
- Dense activation: **ReLU**
- Output activation: **Sigmoid**
- Dropout: **0.30 and 0.20**
- Optimizer: **Adam**
- Learning rate: **0.001**
- Early stopping: Enabled
- Class weighting: Used during training

---

## Models Evaluated

Three model configurations were evaluated:

1. **LSTM Baseline**
2. **LSTM + Attention**
3. **LSTM + GRL (Gradient Reversal Layer)**

### Final Model Selection

The LSTM + Attention model produced exactly the same official test predictions as the baseline.

The GRL model achieved higher drowsy recall but substantially lower precision and F1-score.

Therefore, the simpler **LSTM baseline** was selected as the final model.

---

## Official Test Results

The final evaluation was performed on **179 previously unseen temporal sequences**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **LSTM Baseline** | **98.88%** | **91.67%** | **91.67%** | **91.67%** | **99.95%** |
| LSTM + Attention | 98.88% | 91.67% | 91.67% | 91.67% | 99.95% |
| LSTM + GRL | 92.18% | 46.15% | 100.00% | 63.16% | 99.90% |

### Final Model

**LSTM Baseline**

- Accuracy: **98.88%**
- Precision: **91.67%**
- Recall: **91.67%**
- F1-Score: **91.67%**
- ROC-AUC: **99.95%**

---

## Confusion Matrix

The final LSTM produced:

```text
                 Predicted
              Not Drowsy  Drowsy
Actual
Not Drowsy       166        1
Drowsy             1       11
```

Therefore:

- True Negative (TN): **166**
- False Positive (FP): **1**
- False Negative (FN): **1**
- True Positive (TP): **11**

---

## Video-Level Robustness Evaluation

Repeated video-level train/validation splits were performed using five random seeds:

```text
7, 21, 42, 84, 123
```

The official test set was kept untouched.

### Aggregate Results

| Metric | Mean ± Std |
|---|---:|
| Accuracy | **98.11 ± 2.16%** |
| Precision | **93.59 ± 6.52%** |
| Recall | **97.19 ± 3.96%** |
| F1-Score | **95.32 ± 5.08%** |

These results indicate strong performance across repeated video-level partitions, although performance varies between recordings.

---

## Difficult Video Analysis

The most difficult recording observed during the robustness analysis was:

**P1042762_720**

Performance:

- Accuracy: **88.89%**
- Drowsy Recall: **84.62%**

This indicates that recording-specific variability can affect model generalization.

---

## Recording Distribution Shift

Feature distributions were compared across videos to investigate recording-specific variability.

All evaluated videos showed statistically significant distribution differences in at least some of the selected features.

The strongest shifts were observed in:

- Yaw
- Pitch
- Roll
- EAR Mean

This suggests that **recording/video-specific distribution shift** is an important consideration for real-world deployment.

---

## GRL / Domain Generalization Experiment

A Gradient Reversal Layer (GRL) model was investigated to reduce recording-specific representation bias.

Official test results:

- Accuracy: **92.18%**
- Precision: **46.15%**
- Recall: **100.00%**
- F1: **63.16%**
- ROC-AUC: **99.90%**

Although recall was excellent, the model produced considerably more false positives.

Therefore, GRL was **not selected as the final model**.

---

## Real-Time Inference

A real-time webcam prototype was implemented using:

- OpenCV
- MediaPipe Face Landmarker
- Trained LSTM model

The real-time system:

1. Captures webcam frames.
2. Detects facial landmarks.
3. Extracts EAR Mean, Pitch, Yaw and Roll.
4. Maintains a temporal feature buffer.
5. Generates LSTM predictions.
6. Displays eye-position information.
7. Detects sustained drowsiness.
8. Provides an alarm mechanism.

### Real-Time Feature Validity

During one successful run:

```text
Processed frames: 908
Valid LSTM feature frames: 886
Feature validity rate: 97.58%
```

> **Note:** 97.58% is feature validity, not model accuracy. The official offline model accuracy is 98.88%.

---

## Sustained Drowsiness Detection

The real-time system uses temporal persistence rather than triggering an alarm from a single prediction.

```text
Drowsiness detected
        |
        v
Condition persists
        |
        v
8-second threshold
        |
        v
Drowsiness alarm
```

The system also includes a separate head-down posture condition.

Head position is calibrated relative to the driver's neutral head position rather than relying only on a fixed absolute pitch value.

---

## Project Structure

```text
Driver_Drowsiness_Detection/
|
+-- data/
|   +-- features/
|   +-- sequences/
|   +-- splits/
|
+-- models/
|   +-- lstm_baseline_best.keras
|
+-- results/
|   +-- final_evaluation/
|   +-- final_robustness/
|   +-- model_evaluation/
|   +-- publication_tables/
|   +-- lstm_baseline/
|   +-- lstm_attention/
|   +-- ...
|
+-- src/
|   +-- features/
|   +-- inference/
|   +-- models/
|   +-- preprocessing/
|
+-- requirements.txt
+-- .gitignore
+-- README.md
```

Dataset files and local virtual environments are excluded from version control.

---

## Technologies Used

### Programming

- Python 3.12

### Computer Vision

- OpenCV
- MediaPipe Face Landmarker

### Machine Learning / Deep Learning

- TensorFlow
- Keras
- Scikit-learn

### Data Processing

- NumPy
- Pandas
- SciPy

### Visualization

- Matplotlib
- Seaborn

### Development

- Git
- GitHub
- VS Code
- Conda / Virtual Environment

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Feature Preparation

```bash
python src/features/prepare_modeling_features.py
```

### Temporal Sequence Generation

```bash
python src/features/generate_temporal_sequences.py
```

### Train LSTM Baseline

```bash
python src/models/train_lstm_baseline.py
```

### Train LSTM + Attention

```bash
python src/models/train_lstm_attention.py
```

### Evaluate Models

```bash
python src/models/evaluate_models.py
```

### Generate Final Evaluation Plots

```bash
python src/models/final_evaluation_plots.py
```

### Generate Publication Tables

```bash
python src/models/create_results_tables.py
```

### Run Real-Time Inference

```bash
python src/inference/realtime_drowsiness.py
```

---

## Reproducibility

The main model uses fixed random seeds to improve reproducibility.

The robustness experiment uses multiple video-level random seeds:

```text
[7, 21, 42, 84, 123]
```

The model scaler is fitted using training data only.

The official test set is kept separate from training and model-development decisions.

---

## Limitations

1. The official test set contains only **12 drowsy sequences**, so individual errors have a relatively large effect on recall.
2. Recording-specific distribution shifts affect generalization.
3. The current feature representation uses only four facial movement features.
4. The real-time head-down detector has not yet undergone a separately labeled quantitative accuracy evaluation.
5. Real-world deployment would require evaluation across more drivers, environments, cameras and lighting conditions.
6. The reported 98.88% accuracy should not be interpreted as universal real-world accuracy.

---

## Future Work

Potential future improvements include:

- Evaluate sequence lengths such as 25, 30, 45 and 60 frames.
- Perform controlled temporal-window ablation studies.
- Improve cross-recording generalization.
- Evaluate additional domain-generalization strategies.
- Investigate Transformer-based temporal modeling if justified by experiments.
- Expand evaluation to additional drivers and recording conditions.
- Quantitatively evaluate the real-time head-position detector.
- Optimize inference latency for deployment on edge devices.
- Investigate model calibration and threshold selection using validation data.

---

## References

1. **NITYMED Dataset**  
   Nighttime In-Car Driver Monitoring and Drowsiness Dataset.  
   https://datasets.esdalab.ece.uop.gr/

2. **Soukupová, T. & Čech, J. (2016).**  
   Real-Time Eye Blink Detection using Facial Landmarks.  
   Computer Vision Winter Workshop.

3. **Albadawi, Y., Takruri, M., & Awad, M. (2022).**  
   A Review of Recent Developments in Driver Drowsiness Detection Systems.  
   *Sensors, 22(5), 2069.*

4. **MediaPipe Face Landmarker / Face Mesh**  
   Google AI Edge / MediaPipe documentation.

---

## Author

**Anoop Kumar Dwivedi**

MCA - Data Science & Informatics  
National Institute of Technology, Patna

---

## Key Result

> **A temporal LSTM model using only EAR Mean, Pitch, Yaw and Roll achieved 98.88% accuracy and 99.95% ROC-AUC on the official held-out test set, while repeated video-level evaluation achieved 98.11 ± 2.16% accuracy and 95.32 ± 5.08% F1-score.**

---

## Project Status

**Research-oriented implementation completed.**

The project includes:

- Dataset preprocessing
- Facial feature extraction
- Temporal sequence construction
- Video-level leakage prevention
- LSTM modeling
- Attention comparison
- GRL experiment
- Official test evaluation
- Repeated video-level robustness analysis
- Error analysis
- Real-time webcam prototype
