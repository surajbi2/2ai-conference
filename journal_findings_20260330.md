# Journal Extension Results — Analysis Report
**Run Timestamp:** `journal_results_20260330_150903`  
**Code:** [journal_extension_final.py](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py)  
**Source Anatomy:** XR_WRIST → Target Anatomies: XR_ELBOW, XR_HAND, XR_SHOULDER, XR_FINGER  
**Model:** DenseNet-121 | Seeds: [42, 123, 456, 789, 2024]

---

## 🔬 Experiment Overview

The code trains a DenseNet-121 model on **MURA XR_WRIST** (source anatomy) and studies how well it transfers to other radiograph anatomies (cross-anatomy transfer learning), using external validation on **FracAtlas**. Five evaluation phases are run.

---

## Phase 0 — MURA Source + Transfer Baseline ([mura_summary.csv](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/mura_summary.csv))

| Anatomy | Mean AUC | Std AUC |
|---|---|---|
| **XR_WRIST** (source) | **0.872** | ±0.010 |
| XR_ELBOW | 0.747 | ±0.026 |
| XR_FINGER | 0.749 | ±0.062 |
| XR_HAND | 0.706 | ±0.013 |
| XR_SHOULDER | **0.580** | ±0.022 |

**Key Finding:**
- The wrist-trained model achieves its highest AUC on **XR_WRIST (0.872)** — strong within-anatomy performance.
- Transfer degrades with anatomical distance: Elbow/Finger retain ~0.75 AUC, Hand drops slightly to 0.706, and **Shoulder suffers most at 0.580** (barely above chance).
- **XR_FINGER** shows the highest variance (±0.062), suggesting unstable transfer.

---

## Phase 1 — Grad-CAM Mechanistic Analysis ([gradcam/](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#618-740))

> *Code: [run_gradcam_analysis()](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#618-740) — hooks into `features.denseblock4` of DenseNet-121, computes per-anatomy attention intensity on fracture vs. normal images.*

| Anatomy | Mean CAM (Fracture) | Mean CAM (Normal) | Centrality (Fracture) | Centrality (Normal) |
|---|---|---|---|---|
| XR_WRIST | 0.1787 | 0.1770 | 0.830 | 0.652 |
| XR_ELBOW | **0.2255** | 0.1204 | 0.765 | 0.767 |
| XR_HAND | 0.2230 | 0.1390 | 0.752 | 0.573 |
| XR_SHOULDER | **0.2670** | 0.2484 | **0.890** | 0.811 |
| XR_FINGER | 0.1529 | 0.1264 | 0.606 | 0.514 |

**Key Findings:**
- **XR_WRIST (source):** Very small gap between fracture and normal CAM intensity (0.179 vs 0.177), suggesting the model already learned a domain-specific representation but doesn't strongly differentiate.
- **XR_ELBOW & XR_HAND:** Healthy fracture-vs-normal CAM gap (~0.08–0.09). The model still attends to meaningful regions on these anatomies.
- **XR_SHOULDER:** Has the **highest absolute activation (0.267)** and very high centrality (0.890), but this is accompanied by poor AUC (0.580). This implies the model is **highly activated but confusedly so** — it attends to central regions but cannot discriminate fractures on shoulders.
- **XR_FINGER:** Lowest overall activation (0.153); model has weakest learned representation for this anatomy, consistent with its high AUC variance.
- Grad-CAM per-anatomy visualizations saved to: `gradcam_XR_*.png`

---

## Phase 2 — Feature Space Analysis ([feature_analysis/](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#830-993))

> *Code: [run_feature_analysis()](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#830-993) — extracts 1024-dim penultimate features, computes inter- vs intra-anatomy centroid distances (PCA + t-SNE).*

| Anatomy | Inter-Anatomy Dist | Intra-Anatomy Dist | Distance Ratio | N Samples |
|---|---|---|---|---|
| XR_WRIST (source) | 0.000 | 27.485 | **0.000** | 400 |
| XR_ELBOW | 9.252 | 18.517 | 0.500 | 400 |
| XR_HAND | 10.325 | 18.933 | 0.545 | 389 |
| XR_SHOULDER | **11.805** | 18.192 | **0.649** | 400 |
| XR_FINGER | 5.315 | 20.666 | **0.257** | 400 |

**Key Findings:**
- **XR_WRIST distance ratio = 0.0** by design (it IS the source — no inter-anatomy distance to reference anatomy).
- **XR_SHOULDER has the highest distance ratio (0.649):** Its feature centroid is most distant from XR_WRIST in the learned embedding space. This directly explains its worst AUC — the features are misaligned.
- **XR_FINGER has the lowest distance ratio (0.257):** Despite a relatively large intra-class spread (20.67), its centroid is close to wrist features, meaning the model applies somewhat similar representations, yet still shows high AUC variance.
- The **distance ratio strongly correlates with AUC degradation**: Shoulder > Hand > Elbow > Finger ordinal ordering aligns (with Finger being an outlier due to instability).
- The t-SNE plot ([tsne_feature_space.png](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/feature_analysis/tsne_feature_space.png)) should visually confirm cluster separation.

---

## Phase 3 — Confidence Calibration Analysis ([calibration/](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#1069-1163))

> *Code: [run_calibration_analysis()](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#1069-1163) — computes Expected Calibration Error (ECE) via [compute_ece()](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#1039-1067), generates reliability diagrams per anatomy.*

| Anatomy | ECE ↓ | High-Conf Error Rate ↓ | AUC |
|---|---|---|---|
| **XR_WRIST** | **0.124** | **0.131** | **0.879** |
| XR_FINGER | 0.232 | 0.230 | 0.782 |
| XR_ELBOW | 0.296 | 0.275 | 0.760 |
| XR_HAND | 0.279 | 0.274 | 0.716 |
| **XR_SHOULDER** | **0.414** | **0.416** | **0.547** |

**Key Findings:**
- **XR_WRIST is well-calibrated (ECE = 0.124):** Confidence scores closely match real accuracy on the source domain.
- **Calibration degrades sharply for transferred anatomies:** Elbow and Hand show ECE ~0.28–0.30, while Shoulder reaches **ECE = 0.414** — extremely poor calibration.
- **High-Confidence Error Rate mirrors ECE:** On Shoulder, 41.6% of high-confidence predictions are **wrong** — clinically dangerous in a medical AI context.
- The calibration hierarchy matches the AUC hierarchy almost perfectly: better AUC → better calibration.
- **Implication for paper:** The model doesn't just perform poorly on out-of-anatomy data — it is *confidently wrong*, which is a critical safety concern for deployment.

---

## Phase 4 — FracAtlas External Validation ([fracatlas_validation/](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#1210-1386))

> *Code: [run_fracatlas_validation()](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#1210-1386) — applies MURA-wrist model to FracAtlas (a separate fracture dataset with multi-body-part images), including body-part-stratified analysis.*

### Performance by Subset

| Dataset Subset | AUC | Sensitivity | Specificity | PPV | NPV | FN Rate |
|---|---|---|---|---|---|---|
| **All** | **0.596** | 0.476 | 0.661 | 0.230 | 0.856 | **52.4%** |
| Hand | 0.534 | 0.455 | 0.583 | 0.296 | 0.735 | 54.5% |
| Leg | 0.628 | 0.500 | 0.698 | 0.172 | 0.917 | 50.0% |
| Hip | 0.611 | 0.538 | 0.736 | 0.333 | 0.867 | 46.2% |
| **Shoulder** | **0.315** | **0.214** | **0.492** | **0.088** | **0.732** | **78.6%** |

### False Negative (FN) Analysis

| Subset | Total Fractures | False Negatives | FN Rate | High-Conf FNs |
|---|---|---|---|---|
| All | 143 | 75 | **52.4%** | 68 |
| Hand | 88 | 48 | 54.5% | 43 |
| Leg | 50 | 25 | 50.0% | 24 |
| Hip | 13 | 6 | 46.2% | 6 |
| **Shoulder** | 14 | **11** | **78.6%** | **10** |

**Key Findings:**
- **Overall FracAtlas AUC = 0.596** — substantially below the MURA wrist AUC of 0.872. The model generalizes poorly to this unseen, multi-anatomy dataset.
- **Shoulder is catastrophically bad (AUC = 0.315, below chance):** This is consistent with MURA shoulder findings. The wrist-trained model actively misclassifies shoulder fractures.
- **52.4% of all fractures are missed (False Negatives)**, with **68 out of 75 FNs being high-confidence errors** — confirming the calibration analysis: the model is confidently missing fractures.
- **NPV is relatively high (0.856)** but driven by high specificity of negatives, not true fracture detection competence.
- **Leg & Hip** show moderate performance (AUC ~0.61–0.63), suggesting the single-anatomy model captures some general fracture morphology for non-wrist extremities.
- **PPV is very low across all subsets (0.09–0.30):** High false positive burden in the positive class.

> [!CAUTION]
> 78.6% of shoulder fractures in FracAtlas were **missed with high confidence**. This is a critical safety finding about direct deployment of source-anatomy models.

---

## Phase 5 — Domain Adaptation (`domain_adaptation/`)

> *Code: `run_domain_adaptation()` — tests Fine-Tuning, DANN, and MMD on XR_ELBOW as target, using 50/200/500 labeled samples.*

| Method | N Samples | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| **Zero-shot** | 0 | 0.760 | **0.878** | 0.353 |
| Fine-tune | 50 | 0.833 | 0.670 | 0.855 |
| DANN | 50 | 0.759 | 0.743 | 0.583 |
| MMD | 50 | 0.819 | 0.648 | 0.817 |
| Fine-tune | 200 | **0.859** | 0.743 | 0.843 |
| DANN | 200 | 0.772 | 0.761 | 0.587 |
| MMD | 200 | 0.854 | 0.817 | 0.740 |
| Fine-tune | 500 | 0.858 | 0.717 | **0.885** |
| DANN | 500 | 0.669 | 0.713 | 0.532 |
| MMD | 500 | **0.866** | **0.826** | 0.728 |

**Key Findings:**
- **Zero-shot transfer gives decent AUC (0.760) but terrible specificity (35.3%):** High sensitivity but too many false positives — not clinically viable.
- **Fine-tuning is the most consistently effective method** across all sample sizes (AUC: 0.833 → 0.859 → 0.858).
- **MMD is competitive with fine-tuning at large N:** At 500 samples, MMD achieves AUC **0.866** (highest overall), with a better sensitivity-specificity balance than fine-tune.
- **DANN consistently underperforms all other methods** and actually *degrades* at 500 samples (AUC 0.669) — the adversarial training may be destabilizing the feature representations with this dataset size.
- **Adaptation saturates at ~200 samples** for fine-tuning: performance gain from 200→500 is minimal (0.859 → 0.858).
- **Sensitivity-Specificity trade-off:** Fine-tune sacrifices sensitivity for specificity. MMD achieves a better overall balance at 500 samples.

---

## Summary of Key Findings

| Theme | Finding |
|---|---|
| **Transfer Performance** | Source AUC 0.872 → degrades to 0.580 (Shoulder). Anatomical proximity matters. |
| **Feature Misalignment** | Distance ratio highest for Shoulder (0.649) → explains worst AUC. |
| **Overconfidence** | ECE up to 0.414 (Shoulder); 41.6% of high-confidence predictions are wrong. |
| **External Validity** | FracAtlas AUC = 0.596; 52.4% fractures missed; Shoulder AUC = 0.315 (below chance). |
| **False Negatives** | 68/75 FNs on FracAtlas are high-confidence — clinically dangerous. |
| **Best Adaptation** | MMD at 500 samples (AUC 0.866); Fine-tuning best at low data (50 samples). |
| **DANN Warning** | DANN degrades at 500 samples; not recommended for this setting. |

---

## Artifacts Generated

| Directory | Files |
|---|---|
| Root | [config.json](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/config.json), [mura_summary.csv](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/mura_summary.csv), [JOURNAL_SUMMARY_FIGURE.png](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/JOURNAL_SUMMARY_FIGURE.png) |
| [calibration/](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#1069-1163) | [calibration_results.csv](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/calibration/calibration_results.csv), [ece_comparison.png](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/calibration/ece_comparison.png), [reliability_diagrams.png](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/calibration/reliability_diagrams.png) |
| `domain_adaptation/` | [domain_adaptation_results.csv](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/domain_adaptation/domain_adaptation_results.csv), [domain_adaptation_comparison.png](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/domain_adaptation/domain_adaptation_comparison.png) |
| [feature_analysis/](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#830-993) | [feature_distances.csv](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/feature_analysis/feature_distances.csv), [tsne_feature_space.png](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/feature_analysis/tsne_feature_space.png), [distance_ratio.png](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/feature_analysis/distance_ratio.png) |
| [fracatlas_validation/](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#1210-1386) | [fracatlas_results.csv](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/fracatlas_validation/fracatlas_results.csv), [fracatlas_fn_analysis.csv](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/fracatlas_validation/fracatlas_fn_analysis.csv), [fracatlas_comparison.png](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/fracatlas_validation/fracatlas_comparison.png) |
| [gradcam/](file:///c:/Users/ramag/Desktop/2ai/journal_extension_final.py#618-740) | `gradcam_XR_*.png` (×5), [gradcam_attention_stats.csv](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/gradcam/gradcam_attention_stats.csv), [gradcam_summary.png](file:///c:/Users/ramag/Desktop/2ai/journal_results_20260330_150903/gradcam/gradcam_summary.png) |
