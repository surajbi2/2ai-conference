# 📘 Medical Imaging Evaluation Handbook

*(For X-ray Fracture Detection & Beyond)*

---

# 🧠 0. First Principle (Don’t Skip This)

Every metric answers **one of three questions**:

1. **Discrimination** → Can the model separate classes?
2. **Calibration** → Are predicted probabilities trustworthy?
3. **Localization** → Is the model looking at the correct region?

If your evaluation doesn’t cover all three → incomplete.

---

# 🧩 1. Problem Types (Metrics depend on this)

| Task           | Example                      | Metrics               |
| -------------- | ---------------------------- | --------------------- |
| Classification | Fracture vs Normal           | Accuracy, Recall, AUC |
| Detection      | Find fracture location (box) | mAP, IoU              |
| Segmentation   | Pixel-wise fracture region   | Dice, IoU             |
| Multi-label    | Multiple conditions          | Macro/micro metrics   |

---

# 🧮 2. Confusion Matrix (Core of Everything)

|                 | Predicted Positive | Predicted Negative |
| --------------- | ------------------ | ------------------ |
| Actual Positive | TP                 | FN                 |
| Actual Negative | FP                 | TN                 |

---

## 🧠 Interpretation

* TP → Correct detection
* FN → Missed disease (**critical**)
* FP → False alarm
* TN → Correct rejection

👉 In medicine: **FN is usually the worst**

---

# 📊 3. Classification Metrics (Complete Set)

---

## ✅ Accuracy

[
\frac{TP + TN}{Total}
]

* Misleading in imbalanced datasets

---

## 🎯 Precision (Positive Predictive Value)

[
\frac{TP}{TP + FP}
]

👉 “When model says fracture, how often correct?”

---

## 🔍 Recall (Sensitivity)

[
\frac{TP}{TP + FN}
]

👉 “How many real fractures did we detect?”

🔥 Most critical in diagnosis

---

## 🛡️ Specificity

[
\frac{TN}{TN + FP}
]

👉 “How well we avoid false alarms”

---

## ⚖️ F1 Score

[
\frac{2PR}{P + R}
]

👉 Balance of precision & recall

---

## 📌 Balanced Accuracy

[
\frac{Sensitivity + Specificity}{2}
]

👉 Useful for imbalanced data

---

## 📊 MCC (Matthews Correlation Coefficient)

[
MCC = \frac{TP×TN - FP×FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
]

👉 Best single metric for imbalance

* +1 = perfect
* 0 = random

---

## 🎯 Cohen’s Kappa

Measures agreement beyond chance

👉 Useful when comparing with radiologists

---

# 📈 4. Threshold-Based vs Threshold-Free Metrics

---

## 🔥 ROC Curve & AUC

* Measures **ranking ability**

### Interpretation:

* AUC = probability model ranks positive higher than negative

---

![Image](https://emj.bmj.com/content/emermed/34/6/357/F1.large.jpg)

![Image](https://blog.alliedoffsets.com/hubfs/0_VmdsukltMmSfn1iK.webp)

![Image](https://www.researchgate.net/publication/392469058/figure/fig3/AS%3A11431281517111781%401750895592279/ROC-curve-comparing-the-performance-of-Random-Forest-and-XGBoost-models-Receiver.tif)

---

## 📉 Precision-Recall Curve

Better when:

* Dataset is imbalanced (which yours is)

---

## ⚠️ Key Difference

| Metric  | Use Case            |
| ------- | ------------------- |
| ROC-AUC | Balanced datasets   |
| PR-AUC  | Imbalanced datasets |

---

# 📏 5. Calibration Metrics (Almost Everyone Ignores This)

This is where most “high accuracy models” fail clinically.

---

## 🧠 What is Calibration?

If model says:

* “80% fracture”

👉 Is it actually correct 80% of the time?

---

## 📊 Metrics

### 🔹 Brier Score

[
Mean (predicted - actual)^2
]

* Lower is better

---

### 🔹 Expected Calibration Error (ECE)

Measures mismatch between:

* predicted probability vs actual outcome

---

### 🔹 Reliability Diagram

![Image](https://www.researchgate.net/publication/382687385/figure/fig3/AS%3A11431281263877923%401722370219890/Reliability-diagram-for-evaluating-the-calibration-of-a-classification-model-Points.png)

![Image](https://www.researchgate.net/publication/259805161/figure/fig2/AS%3A279058769301520%401443544203720/Calibration-plot-Observed-vs-predicted-probability-of-survival-in-the-derivation-blue.png)

![Image](https://www.researchgate.net/publication/358177121/figure/fig2/AS%3A11431281275329776%401725377952249/A-sample-sketch-of-the-reliability-diagram-shows-perfectly-calibrated-overconfident.tif)

---

### 🧠 Interpretation

| Pattern        | Meaning        |
| -------------- | -------------- |
| Below diagonal | Overconfident  |
| Above diagonal | Underconfident |

---

# 🧩 6. Detection Metrics (Bounding Boxes)

Used if you localize fractures.

---

## 📏 IoU (Intersection over Union)

[
\frac{Overlap}{Union}
]

---

## 🎯 mAP (Mean Average Precision)

* Measures detection performance across thresholds

---

### 🧠 Interpretation

* mAP@0.5 → loose localization
* mAP@0.75 → strict

---

# 📐 7. Segmentation Metrics (Pixel-Level)

---

## 🎯 Dice Coefficient

[
\frac{2TP}{2TP + FP + FN}
]

👉 Most used in medical imaging

---

## 📏 IoU (Jaccard Index)

[
\frac{TP}{TP + FP + FN}
]

---

## 🧠 Difference

* Dice is more forgiving
* IoU is stricter

---

# 🩻 Segmentation Example

![Image](https://www.researchgate.net/publication/365411986/figure/fig2/AS%3A11431281115074422%401674750673231/From-left-to-right-Ground-truth-gt-versus-prediction-pred-area-of-union-gt.png)

![Image](https://www.researchgate.net/publication/370398196/figure/fig1/AS%3A11431281154377145%401682779562255/Comparison-of-real-image-mask-and-predicted-mask-in-DiceCoefficient-and-Jaccard-Index.ppm)

![Image](https://www.researchgate.net/publication/380460349/figure/fig22/AS%3A11431281242126554%401715357657067/X-ray-images-of-a-hand-with-predicted-bone-instance-segmentation-masks-and-boxes.png)

---

# 🔍 8. Localization Metrics (Weak Supervision / Grad-CAM)

---

## 📌 Pointing Game

* Check if peak activation lies inside fracture region

---

## 📏 Localization Accuracy

* % of heatmaps overlapping annotated region

---

## 🎯 Energy-Based Metrics

* How much heatmap mass lies inside ground truth

---

# 🧠 9. Explainability Metrics (Grad-CAM & Beyond)

---

## 📌 What to Evaluate

### 1. Faithfulness

* Does heatmap reflect model decision?

### 2. Localization

* Does it align with pathology?

### 3. Robustness

* Does explanation change drastically with small noise?

---

## ⚠️ Reality

* Grad-CAM is **not reliable alone**
* Compare with:

  * Grad-CAM++
  * Score-CAM
  * Eigen-CAM

---

# 🧪 10. Statistical Evaluation (You NEED this for PhD level)

---

## 📊 Confidence Intervals

* Report uncertainty in metrics

---

## 🧠 Bootstrapping

* Resample dataset → compute metric distribution

---

## 🧪 Hypothesis Testing

* Compare models (e.g., AUC difference)

---

# 📉 11. Error Analysis Metrics

---

## 📌 Per-Class Metrics

* Fracture types (wrist, hip, etc.)

---

## 📌 Hard Case Analysis

* Small fractures
* Low contrast
* Occlusions

---

## 📌 Subgroup Analysis

* Age groups
* Imaging views

---

# 📊 12. Training Metrics (Model Behavior)

---

## 📉 Loss Curve

* Optimization progress

---

## 📈 Accuracy Curve

* Learning behavior

---

## 🚨 Red Flags

| Pattern       | Problem      |
| ------------- | ------------ |
| Train ↑ Val ↓ | Overfitting  |
| Both low      | Underfitting |

---

# 🧠 13. Clinical Metrics (Rarely Used but Powerful)

---

## 📌 NPV (Negative Predictive Value)

[
\frac{TN}{TN + FN}
]

👉 “If model says normal, can we trust it?”

---

## 📌 PPV (Same as Precision)

---

## 📌 Decision Curve Analysis

* Measures clinical usefulness

---

# 📦 14. Generalization Metrics

---

## 📌 Cross-Dataset Performance

Train → MURA
Test → FracAtlas

👉 Measures robustness

---

## 📌 Domain Shift Sensitivity

* Performance drop across datasets

---

# 🧠 15. What a Complete Evaluation Looks Like

---

## ✔ MUST HAVE

* Confusion matrix metrics
* ROC + PR curves
* Calibration analysis
* Error breakdown

---

## ✔ STRONG PAPER

* Localization metrics
* Explainability validation
* Cross-dataset testing
* Statistical significance

---

## ✔ TOP-TIER RESEARCH

* Clinical validation
* Human vs AI comparison
* Failure mode taxonomy

---

# 🚫 Final Reality Check

If your evaluation:

* Ignores calibration → not clinically usable
* Ignores FN → dangerous
* Ignores localization → not trustworthy
* Ignores generalization → not publishable

---

# 🧠 Bottom Line

* **Metrics ≠ understanding**
* Each metric answers a different failure mode
* Real research = connecting them into a **coherent story**

