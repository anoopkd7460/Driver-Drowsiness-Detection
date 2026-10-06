# Driver Drowsiness Detection Using Temporal Facial and Head-Pose Features

## 1. Research Objective

The objective of this work is to develop a temporal driver drowsiness detection system capable of distinguishing between **Not Drowsy** and **Drowsy** states using facial and head-pose information extracted from driver video frames.

The system focuses on temporal behavioral information rather than making independent frame-level predictions. Four facial/head-pose features are used:

* Mean Eye Aspect Ratio (EAR)
* Head-pose pitch
* Head-pose yaw
* Head-pose roll

No mouth-based features are used in the proposed system.

The extracted frame-level features are organized into temporal sequences and classified using a Long Short-Term Memory (LSTM) neural network.

---

## 2. Dataset

The experiments use the NITYMED dataset available in the project directory.

The local dataset contains:

* 23 local videos
* 12 male videos
* 11 female videos
* 73,897 total video frames
* Frame rate: 25 FPS
* Video resolutions: 1280×720 and 1920×1080

The available annotation files contain frame-level driver states including:

* Alert
* Microsleep
* Yawning

For this study, the classification task is defined as:

| Original State |  Project Label |
| -------------- | -------------: |
| Alert          | 0 — Not Drowsy |
| Microsleep     |     1 — Drowsy |
| Yawning        |        Ignored |

The annotation files contain 53,331 records. After restricting the analysis to the relevant classes and valid annotations, 47,834 annotated records were available for the selected task.

The resulting class distribution was:

* Not Drowsy: 38,954
* Drowsy: 8,880

Therefore:

* Not Drowsy: 81.44%
* Drowsy: 18.56%

---

## 3. Frame-Level Feature Extraction

Facial landmarks were extracted using MediaPipe.

The feature extraction process identifies facial landmarks around the eyes and selected points on the face to estimate head orientation.

### 3.1 Eye Aspect Ratio

The Eye Aspect Ratio (EAR) is used as a measure of eye openness.

Six landmarks are used for each eye. EAR provides a compact representation of the geometric relationship between the vertical and horizontal dimensions of the eye.

A mean value across the two eyes is used as the final eye-related feature:

**EAR Mean**

Lower EAR values generally correspond to more closed eyes.

### 3.2 Head Pose

Three geometric head-pose features are calculated:

* Pitch
* Yaw
* Roll

**Pitch** represents the vertical orientation of the head.

**Yaw** represents the horizontal orientation of the head.

**Roll** represents the rotation of the head around the forward-facing axis.

The head-pose estimation used in this work is geometry-based rather than calibrated PnP-based estimation.

### 3.3 Final Feature Set

After feature analysis and redundancy inspection, four features were retained:

| Feature    | Description                 |
| ---------- | --------------------------- |
| `ear_mean` | Mean eye aspect ratio       |
| `pitch`    | Head vertical orientation   |
| `yaw`      | Head horizontal orientation |
| `roll`     | Head rotation               |

The final feature representation therefore consists of:

**4 features per frame.**

---

## 4. Feature Quality and Validity

The feature extraction process did not fabricate missing measurements through interpolation.

Frames where the required facial information could not be reliably obtained were retained as invalid rather than assigning artificial feature values.

Among the 47,834 processed records:

* Valid records for all four selected features: 35,934
* Invalid records: 11,900
* Face detection rate: approximately 75.12%

Class-level feature analysis showed that eye openness provided the strongest separation between the two classes.

The standardized class separation for the selected features showed:

* EAR Mean: strongest separation
* Pitch: meaningful separation
* Yaw: smaller separation
* Roll: very small standalone separation

Although yaw and roll showed weaker individual class separation, they were retained because the objective was temporal classification and changes in head orientation may provide useful sequential information.

---

## 5. Data Preparation and Leakage Prevention

A major methodological consideration in temporal video classification is data leakage.

Adjacent frames from the same video are highly correlated. Randomly splitting individual frames or overlapping temporal windows between training and testing can therefore result in highly optimistic performance estimates.

To prevent this, the dataset was split at the **video level before temporal sequence generation**.

Each video was assigned exclusively to one of:

* Training set
* Validation set
* Test set

No video was shared between these sets.

The final official split contained:

| Split      | Videos | Sequences | Not Drowsy | Drowsy |
| ---------- | -----: | --------: | ---------: | -----: |
| Training   |     31 |       395 |        338 |     57 |
| Validation |      7 |       128 |        112 |     16 |
| Test       |      6 |       179 |        167 |     12 |

The temporal sequences were subsequently generated only within the assigned split.

This procedure prevents temporal windows originating from the same video from appearing across different evaluation sets.

---

## 6. Temporal Sequence Construction

Instead of treating each frame independently, consecutive frame-level features were grouped into temporal sequences.

The selected sequence length was:

**30 frames**

At the dataset frame rate of 25 FPS:

**30 frames = 1.2 seconds**

Each sequence therefore has the shape:

**30 × 4**

where:

* 30 = temporal frames
* 4 = selected features

The four feature streams are:

1. EAR Mean
2. Pitch
3. Yaw
4. Roll

The sequence generation used non-overlapping windows with a step size of 30 frames.

Each generated sequence was assigned a single class label only when the frames within the sequence satisfied the required labeling and feature-validity conditions.

---

## 7. Feature Standardization

Feature standardization was performed using statistics calculated from the training data only.

The training-set mean and standard deviation were then applied to the validation and test sets.

The resulting training statistics were:

| Feature  |       Mean | Standard Deviation |
| -------- | ---------: | -----------------: |
| EAR Mean |   0.195844 |           0.070170 |
| Pitch    | -21.444487 |           7.267564 |
| Yaw      |   4.021970 |           8.950607 |
| Roll     |  -1.334033 |           5.163262 |

This prevents information from the validation or official test set from influencing the feature scaling process.

---

## 8. LSTM Baseline Architecture

The final model uses an LSTM architecture designed to learn temporal dependencies across the 30-frame sequence.

The architecture is:

```text
Input: 30 × 4
      ↓
LSTM(64)
      ↓
Dropout(0.30)
      ↓
Dense(32, ReLU)
      ↓
Dropout(0.20)
      ↓
Dense(1, Sigmoid)
```

The output is a probability representing the likelihood of the Drowsy class.

A threshold of 0.5 is used for the final binary classification.

### Training Configuration

The model uses:

* Optimizer: Adam
* Learning rate: 0.001
* Early stopping
* Best validation-loss weights restored
* Class weighting during training

The class weights were:

* Not Drowsy: 0.5843
* Drowsy: 3.4649

Class weighting was used to compensate for the imbalance between the two classes during training.

The best model was restored from epoch 4, while training stopped at epoch 12 according to the early-stopping criterion.

The resulting model is stored as:

`models/lstm_baseline_best.keras`

---

## 9. Attention-Based Comparison

A second model was evaluated by adding an attention mechanism to the LSTM representation.

Its structure was:

```text
Input: 30 × 4
      ↓
LSTM(64, return_sequences=True)
      ↓
Self-Attention
      ↓
Global Average Pooling
      ↓
Dropout
      ↓
Dense(32, ReLU)
      ↓
Dropout
      ↓
Dense(1, Sigmoid)
```

The attention model used the same:

* Dataset split
* Features
* Sequence length
* Standardization
* Optimizer
* Class weights
* Random seed

The official test performance was exactly the same as the baseline.

Therefore, the attention mechanism produced no measurable improvement under the current experimental configuration.

---

## 10. Gradient Reversal Experiment

A Gradient Reversal Layer (GRL) based model was also evaluated to investigate whether learning domain-invariant representations could improve robustness against recording/video-specific distribution differences.

The model used:

* A shared LSTM representation
* A drowsiness classification head
* A domain classification head
* Gradient reversal between the shared representation and the domain classifier

Different GRL coefficients were evaluated:

* 0.01
* 0.05
* 0.10
* 0.20

The coefficient 0.10 produced the strongest validation F1 among the tested values and was selected for the controlled comparison.

However, its official test performance was substantially worse than the baseline.

---

## 11. Model Comparison

The official test results are:

| Model             |   Accuracy |  Precision |      Recall |   F1-Score |    ROC-AUC |
| ----------------- | ---------: | ---------: | ----------: | ---------: | ---------: |
| **LSTM Baseline** | **98.88%** | **91.67%** |  **91.67%** | **91.67%** | **99.95%** |
| LSTM + Attention  |     98.88% |     91.67% |      91.67% |     91.67% |     99.95% |
| LSTM + GRL        |     92.18% |     46.15% | **100.00%** |     63.16% |     99.90% |

The baseline and attention models produced identical predictions on all 179 official test sequences.

The GRL model detected all drowsy test sequences but produced substantially more false positives.

Therefore, the baseline LSTM provides the best balance between:

* Accuracy
* Precision
* Recall
* F1-score
* Model simplicity

The LSTM baseline was consequently selected as the final model based on validation/robustness considerations and architectural simplicity, with its performance subsequently confirmed using the untouched official test set.

---

## 12. Official Test Evaluation

The final LSTM baseline was evaluated on 179 previously unseen temporal sequences.

The class distribution was:

* Not Drowsy: 167
* Drowsy: 12

The confusion matrix was:

| Actual / Predicted | Not Drowsy | Drowsy |
| ------------------ | ---------: | -----: |
| Not Drowsy         |        166 |      1 |
| Drowsy             |          1 |     11 |

Therefore:

* True Negatives = 166
* False Positives = 1
* False Negatives = 1
* True Positives = 11

The model made only two incorrect predictions among the 179 official test sequences.

### Final Test Metrics

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **98.88%** |
| Precision | **91.67%** |
| Recall    | **91.67%** |
| F1-Score  | **91.67%** |
| ROC-AUC   | **99.95%** |

The ROC-AUC indicates very strong separation between the two classes across probability thresholds.

---

## 13. Repeated Video-Level Robustness Evaluation

To examine whether the observed performance was dependent on one particular video split, repeated video-level validation was performed using five different random seeds:

* 7
* 21
* 42
* 84
* 123

Each experiment used an 80/20 video-level split.

The official test set remained untouched during these robustness experiments.

The five experiments produced:

| Seed | Accuracy | Precision |  Recall |      F1 |
| ---: | -------: | --------: | ------: | ------: |
|    7 |   98.90% |    91.67% | 100.00% |  95.65% |
|   21 |  100.00% |   100.00% | 100.00% | 100.00% |
|   42 |  100.00% |   100.00% | 100.00% | 100.00% |
|   84 |   95.35% |    84.62% |  91.67% |  88.00% |
|  123 |   96.32% |    91.67% |  94.29% |  92.96% |

Aggregate results:

| Metric    |       Mean | Standard Deviation | Minimum | Maximum |
| --------- | ---------: | -----------------: | ------: | ------: |
| Accuracy  | **98.11%** |              2.16% |  95.35% | 100.00% |
| Precision |     93.59% |              6.52% |  84.62% | 100.00% |
| Recall    | **97.19%** |              3.96% |  91.67% | 100.00% |
| F1-Score  | **95.32%** |              5.08% |  88.00% | 100.00% |

These results indicate strong performance across repeated video-level splits, although some variation exists depending on which videos are held out.

---

## 14. Recording-Level Distribution Shift

An additional analysis was conducted to investigate whether the videos exhibit recording-specific feature distributions.

Across the analyzed videos, statistically significant differences were observed in the distributions of the selected features.

The feature distributions therefore exhibit meaningful recording/video-specific variation.

The strongest average standardized differences were observed for:

* Yaw
* Pitch
* Roll
* EAR Mean

This supports the motivation for investigating domain-generalization techniques.

However, the available annotation metadata does not provide a clean independent driver/session/camera identifier. Consequently, the experiments are described as **recording/video-specific distribution shift** rather than definitive independent driver-domain information.

---

## 15. Video-Wise Error Analysis

Repeated video-level evaluation identified several relatively difficult recordings.

The most challenging observed case was:

**P1042762_720**

Under seed 84:

* Sequences: 54
* Drowsy sequences: 13
* Accuracy: 88.89%
* Drowsy recall: 84.62%

Other relatively difficult recordings included:

* P1042756_720
* P1042767_720
* P1042772_720

The presence of such difficult recordings demonstrates that aggregate accuracy alone does not fully describe model behavior.

The model performs strongly overall but is not equally reliable across every recording.

---

## 16. Discussion

The experimental results demonstrate that a relatively compact temporal LSTM can effectively classify drowsiness using only four frame-level facial and head-pose features.

The strongest individual feature separation was observed for eye openness, while head-pose information provided complementary temporal information.

The temporal formulation is important because drowsiness-related behavior is not necessarily represented by a single frame. By processing a sequence of 30 frames, the LSTM can model changes in eye openness and head orientation over approximately 1.2 seconds.

The attention experiment did not improve performance. Because the baseline already achieved strong results, the additional attention mechanism did not provide a measurable advantage under the current data and training configuration.

The GRL experiment produced an interesting trade-off. Although the GRL model achieved 100% drowsy recall on the official test set, its precision decreased to 46.15%, resulting in substantially more false alarms and an F1-score of only 63.16%. Thus, domain-adversarial training did not improve the overall classification objective in the current setting.

The repeated video-level experiments provide additional evidence that the strong performance is not restricted to a single random video split. Nevertheless, the variation across held-out videos demonstrates that recording-specific differences remain an important challenge.

---

## 17. Limitations

Several limitations should be considered when interpreting the results.

### 17.1 Limited number of local videos

The local dataset contains a relatively small number of videos compared with large-scale modern deep-learning datasets.

### 17.2 Imbalanced classes

The frame-level annotated data contain substantially more Not Drowsy samples than Drowsy samples.

Class weighting was therefore used during training.

### 17.3 Limited drowsy representation in the official test set

The official test set contains 179 temporal sequences but only 12 drowsy sequences.

Furthermore, the drowsy test sequences originate from only one test video containing drowsy sequences.

Therefore, the official test recall should not be interpreted as a definitive estimate of performance across a large population of drivers.

The repeated video-level validation results are consequently important for interpreting robustness.

### 17.4 Recording-specific distribution shift

Feature distributions vary across recordings.

This indicates that changes in camera conditions, driver appearance, positioning, or recording characteristics may affect model behavior.

### 17.5 Domain labels

The available annotation metadata does not provide explicit driver, session, camera, or recording-domain identifiers.

Therefore, video identity cannot automatically be interpreted as an independent driver domain.

### 17.6 Geometric head-pose estimation

The head-pose features are based on geometric calculations rather than a fully calibrated 3D camera model.

Future work could investigate calibrated pose estimation if additional robustness is required.

### 17.7 Real-world deployment

The current evaluation is based on prepared dataset videos and temporal feature sequences.

Real-time webcam/video evaluation is required before making claims about practical deployment performance.

---

## 18. Final Model

The final selected model is the **LSTM Baseline**.

Its complete processing pipeline is:

```text
Driver Video
     ↓
Frame Extraction
     ↓
MediaPipe Facial Landmark Detection
     ↓
EAR Mean + Pitch + Yaw + Roll
     ↓
Feature Validation
     ↓
30-Frame Temporal Sequence
     ↓
Training-Based Standardization
     ↓
LSTM(64)
     ↓
Dropout
     ↓
Dense(32)
     ↓
Dropout
     ↓
Sigmoid
     ↓
Drowsy / Not Drowsy
```

The final trained model is:

`models/lstm_baseline_best.keras`

---

## 19. Final Conclusion

This work demonstrates that temporal modeling of facial eye openness and head-pose features can provide highly effective driver drowsiness classification.

The final LSTM model achieved:

**98.88% accuracy**

**91.67% precision**

**91.67% recall**

**91.67% F1-score**

**99.95% ROC-AUC**

on the untouched official test set.

Repeated video-level evaluation further produced:

**98.11% ± 2.16% accuracy**

**97.19% ± 3.96% recall**

**95.32% ± 5.08% F1-score**

The attention-based model did not improve upon the baseline, while the GRL experiment demonstrated reduced overall classification performance despite achieving perfect drowsy recall.

The results support the LSTM baseline as the most appropriate final model for the current dataset and experimental setting.

However, the model should not yet be described as universally generalizable to all drivers or real-world driving conditions. The limited number of test drowsy sequences, recording-specific distribution shift, and absence of explicit independent driver/session metadata are important limitations.

The next step is therefore to evaluate the frozen model on **unseen webcam/video input** through a real-time inference pipeline before making practical deployment claims.
