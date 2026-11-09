# Part 1 — Audit report: Rohit's data, then the models that earn their place

Author: Sikharam Satya Sri Harsha Phani. Every number below comes from a file in `results/` written by a script in this repository (`python run_all.py`); figures are in `figures/`. Seeds 0, 1, 2 are logged in `results/seed_log.txt`. Validation = fold 0 of the new split; the locked test set was never used for a decision.

---

## 1. Inventory and pixel statistics (`audit.py` → `results/audit_summary.md`)

| original split | NORMAL | PNEUMONIA | total | % pneumonia |
|---|---:|---:|---:|---:|
| train | 1341 | 3875 | 5216 | 74.3 |
| test | 234 | 390 | 624 | 62.5 |
| val | 8 | 8 | 16 | 50.0 |
| all | 1583 | 4273 | 5856 | 73.0 |

* **Greyscale.** 5,573 files are mode `L`; 283 are stored as RGB, but all 5,856 have identical channels, so all are true greyscale content.
* **Size and shape differ strongly by class.** Median height 1,323 px (NORMAL) vs 776 px (PNEUMONIA); median aspect ratio 1.24 vs 1.49; median area about 2.18 M vs 0.90 M pixels.
* **Brightness does not differ.** Mean brightness 0.481 vs 0.482. Contrast (std) differs slightly: 0.240 vs 0.217.

Single-feature separability (ROC-AUC of one number alone; 0.5 = nothing, far from 0.5 either way = shortcut):

| feature | AUC | | feature | AUC |
|---|---:|---|---|---:|
| mean brightness | 0.508 | | width | 0.111 |
| contrast (std) | 0.302 | | aspect ratio | 0.840 |
| height | 0.085 | | area | 0.090 |

**Implication.** Height alone separates the classes with AUC 0.915 (0.085 flipped), area 0.910. The two classes were evidently assembled from different sources or crops, so a network can score well by recognising the crop/scanner format instead of lung tissue. At the camp, framing and size are set by different hardware and such cues will not transfer. (My initial expectation was that brightness would be the shortcut; the data say otherwise.) Figures: `figures/audit_class_histograms.png`, `figures/audit_pixel_hist.png`.

## 2. Duplicates and leakage (`results/audit_summary.md`, `results/near_duplicates.csv`)

* **Exact duplicates (file bytes):** 30 groups, 32 redundant files; **none span two original splits**.
* **Near-duplicates (pHash, Hamming distance ≤ 4 of 64):** 239 pairs — 210 within train, 8 within test, **21 across train/test**. 15 of 624 test images (2.4%) have a train neighbour. The mean pixel correlation of the cross pairs is only 0.876.
* **Worst five train/test pairs** (`figures/audit_worst5_train_test_pairs.png`): all have distance 4. None is a copy. Four are two NORMAL images with the same standard frontal framing but different anatomy (rib spacing, clavicles, heart outline) and a differently placed "R" marker; the fifth is a PNEUMONIA training image against a NORMAL test image with visibly different brightness. A 4-bit threshold therefore flags look-alike chest X-rays, so **2.4% is an upper bound on duplicate leakage**. (Visual reading; verify at full resolution.)
* **Patient overlap.** A naive file-name parser reports 264 shared IDs, but these are collisions (e.g. `person100_virus` in train vs `person100_bacteria` in test). With person+pathogen keys (PNEUMONIA) and series-aware keys (NORMAL), **0 IDs appear in both train and test**. File names cannot prove patient-level separation for NORMAL images, so this is absence of *detectable* overlap only.

**Consequence for the 98.4%.** Leakage by duplication is at most modest and cannot by itself explain it. What the audit does show are risks that inflate any single number from this data: a 16-image validation set, a class mix that shifts between splits (74% → 62.5% → 50% pneumonia), and strong non-lung shortcuts. The 98.4% cannot be reproduced (no code, split or weights) and I do not claim to know how it was obtained.

## 3. The validation set and its replacement (`split.py` → `results/split_summary.md`)

* **Confidence interval.** For 16/16 correct the exact Clopper–Pearson 95% interval is **[0.794, 1.000]**: the lower bound is 0.025^(1/16) = 0.794. A perfect validation score is compatible with a true accuracy near 79%. (Rohit's 98.4% on 624 test images ≈ 10 errors; CI [0.971, 0.992] from sampling noise alone.)
* **Replacement design.** All 5,856 images were pooled and re-split by **group** (patient ID merged with near-duplicate clusters; 2,778 groups, none shared across splits — asserted in the script): a **locked test set** of 837 images (226 NORMAL / 611 PNEUMONIA, 73.0% pneumonia) and **5 stratified folds** of ~1,004 images (72.9–73.0% pneumonia) on the remaining pool. Fold 0 is the validation set for every decision. 1,254 images changed split.
* **Why both, and what I rejected.** A locked test set gives one unbiased number; the folds give mean ± std and a validation set large enough to tune on. *Rejected: a single train/val/test split alone* — every decision overfits one slice and the estimate stays noisy. *Rejected: k-fold alone* — no untouched test set, so reported scores are biased by hyper-parameter and threshold selection.

## 4. Shortcut hunt (`shortcut.py` → `results/shortcut_summary.md`)

Crippled models trained on the pool, evaluated on the locked test set (majority-class accuracy 0.73):

| model | acc | bal. acc | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| 16×16 pixels + logistic regression | 0.925 | 0.901 | 0.975 | 0.991 |
| 16×16 pixels + random forest | 0.912 | 0.871 | 0.968 | 0.987 |
| **lung field masked (border only)** + logistic regression | 0.888 | 0.844 | **0.923** | 0.959 |
| metadata only (mean, aspect, area) + logistic regression | 0.841 | 0.775 | 0.916 | 0.970 |
| metadata only + random forest | 0.858 | 0.809 | 0.922 | 0.971 |

**What it means.** A 16×16 thumbnail already reaches AUC 0.975, close to the real model's 0.989 — but it still shows coarse lung opacity, so that alone is not proof of a shortcut. The cleaner evidence: with the lung field **masked out**, a linear model still gets AUC 0.923, and three metadata numbers alone get 0.916–0.922. **Information outside the lungs identifies the class**, and the real model can use it too. High accuracy on this dataset therefore does not show that a model reads lung tissue. Example inputs: `figures/shortcut_masked_examples.png`.

## 5. Baselines (`baselines.py` → `results/baselines_summary.md`)

| model | split | acc | bal. acc | precision | recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| majority class | test | 0.730 | 0.500 | 0.730 | 1.000 | 0.844 | 0.500 | 0.730 |
| HOG + logistic regression | val | 0.943 | 0.929 | 0.962 | 0.960 | 0.961 | 0.988 | 0.995 |
| HOG + logistic regression | test | 0.939 | 0.932 | 0.968 | 0.948 | 0.958 | 0.981 | 0.993 |
| LBP + logistic regression | test | 0.912 | 0.889 | 0.941 | 0.938 | 0.939 | 0.959 | 0.984 |

**The bar for deep learning:** HOG + logistic regression already reaches test AUC 0.981 and balanced accuracy 0.932. The best deep model improves AUC by under two points.

## 6. Transfer baseline (`train.py`, 3 seeds, mean ± std; `results/summary_runs.md`)

**Expectation (written before running; `reports/hypotheses.md`, commit `4f614ff`, 2026-11-08 15:01 +0530):** ImageNet low-level filters (edges, textures) transfer to radiographs; high-level object features do not. So pretrained backbones should be at least as good as a from-scratch network, without a wide margin, and converge in fewer epochs.

| model | params | val ROC-AUC | test ROC-AUC | test PR-AUC | test bal. acc | test sens. | test spec. |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 (ImageNet) | 11.18 M | 0.998 ± 0.001 | 0.995 ± 0.000 | 0.998 ± 0.000 | 0.968 ± 0.007 | 0.981 ± 0.001 | 0.954 ± 0.016 |
| MobileNetV3-small (ImageNet) | 0.93 M | 0.997 ± 0.001 | 0.994 ± 0.001 | 0.998 ± 0.000 | 0.962 ± 0.007 | 0.981 ± 0.011 | 0.944 ± 0.024 |

**Did the numbers agree?** Yes. Both pretrained models beat ArogyaNet (test AUC 0.988) — by +0.007 and +0.006 AUC and +0.022 / +0.017 balanced accuracy — and ResNet18 reached validation AUC 0.993 in its first epoch. The gap is small in AUC, larger in balanced accuracy. Practical note: ResNet18 is about 45 MB in float32 (over the 25 MB channel); MobileNetV3-small is about 4 MB.

## 7. ArogyaNet (`arogyanet.py`, `arch_report.py` → `results/arch_table.md`)

**7a. Architecture (input 1×224×224; 242,319 parameters; limit 2 M; no pretrained weights).**

| layer | output shape | params |
|---|---|---:|
| input BN | 1×224×224 | 2 |
| stem (3×3, stride 2) | 16×112×112 | 176 |
| block 1 (2× depthwise-separable, SE, residual) | 32×56×56 | 3,060 |
| block 2 | 64×28×28 | 10,728 |
| block 3 | 128×14×14 | 39,888 |
| block 4 | 192×7×7 | 100,120 |
| block 5 | 192×7×7 | 88,152 |
| global average pool | 192 | 0 |
| dropout + linear | 1 | 193 |
| **total** | | **242,319** |

Diagram: `figures/arogyanet_diagram.png`. Deliberate decisions: **D1** residual shortcut in every block; **D2** squeeze-and-excitation in every block (also: depthwise-separable convolutions for CPU speed, global-average-pool head instead of flatten+dense).

**7b. Hypotheses, written before the results** (`reports/hypotheses.md`):
* **D1:** residual connections keep gradients flowing through the five stacked blocks; removing them should cost roughly 0–1 pp of validation ROC-AUC, likely inside seed noise.
* **D2:** SE re-weights channels using global context so channels that respond to diffuse opacity are boosted; removing it should lower PR-AUC / specificity by about 1 pp, possibly inside the noise.

**7c. Ablations** (3 seeds each; Δ = variant − reference; rule: *inside noise* if |Δ| < √(std_ref² + std_var²), a crude heuristic with n = 3):

| change | split | ΔROC-AUC | ΔPR-AUC | Δbal. acc | verdict |
|---|---|---:|---:|---:|---|
| remove residuals (D1) | val | −0.002 ± 0.002 | −0.000 ± 0.002 | −0.019 ± 0.016 | AUC marginal, PR-AUC noise, bal. acc worse |
| remove residuals (D1) | test | −0.004 ± 0.001 | −0.001 ± 0.001 | −0.013 ± 0.007 | **small, consistent loss** |
| remove SE (D2) | val | −0.001 ± 0.002 | 0.000 ± 0.002 | −0.007 ± 0.006 | noise (bal. acc marginal) |
| remove SE (D2) | test | 0.000 ± 0.001 | 0.000 ± 0.000 | −0.001 ± 0.009 | **inside noise** |

**Reading.** D1: a small but consistent benefit on the test set (AUC −0.004 beyond the ±0.001 spread), weaker on validation — roughly in line with the 0–1 pp expectation, somewhat larger than "noise" on test. **D2 failed:** squeeze-and-excitation gave no measurable benefit in any metric, and it costs 24k parameters (about 10%). The hypothesis was not supported and is reported as such.

## 8. The two-branch experiment (`arogyanet_meta`)

Mean intensity, aspect ratio and log-area go through a small dense branch (BN → 16 → 16) and are concatenated with the 192-d image vector before the classifier (242,677 parameters).

| model | val ROC-AUC | test ROC-AUC | test bal. acc |
|---|---:|---:|---:|
| image only | 0.993 ± 0.002 | 0.988 ± 0.001 | 0.945 ± 0.006 |
| image + metadata | 0.994 ± 0.001 | 0.990 ± 0.001 | 0.949 ± 0.007 |

**Result:** the differences (+0.001 val, +0.002 test AUC; +0.004 balanced accuracy) are all inside noise. *Hypothesis (written before): the branch helps internally — not supported.*

**Connection to task 4.** The metadata is informative on its own (AUC 0.916–0.922), so the branch adds nothing the image branch cannot already read (framing, borders, intensity layout). I therefore cannot claim the model learned something specific about *these patients* — only that its signal is not distinguishable from such cues. **At the camp this would not hold:** these numbers depend on how images were cropped and exposed at one source (here, height and aspect ratio separate the classes), and the external hospital's images are all 299×299 (KS statistic for width = 1.0). A model leaning on them degrades when equipment changes.

## 9. Two ablations only (`results/summary_runs.md`)

| change | split | ΔROC-AUC | Δbal. acc | verdict |
|---|---|---:|---:|---|
| resolution 224 → 96 px | val | −0.001 ± 0.002 | −0.005 ± 0.007 | noise |
| resolution 224 → 96 px | test | −0.003 ± 0.003 | −0.012 ± 0.006 | small loss on balanced accuracy |
| augmentation: safe → none | val | −0.002 ± 0.002 | −0.014 ± 0.007 | AUC noise, bal. acc worse |
| augmentation: safe → none | test | −0.006 ± 0.001 | −0.011 ± 0.008 | **worse** |
| augmentation: safe → naive (with h-flip) | val | −0.001 ± 0.002 | −0.008 ± 0.008 | noise |
| augmentation: safe → naive (with h-flip) | test | 0.000 ± 0.002 | +0.001 ± 0.007 | **inside noise** |

* **Resolution:** 96 px loses little (a cheaper model is viable).
* **Augmentation:** small rotations/shifts/zoom help a little compared with none.
* **The medically wrong augmentation is the horizontal flip** (and the vertical flip). It places the heart on the right — dextrocardia, a rare and, for this population, implausible anatomy — and swaps the meaning of the left/right laterality marker, so the image no longer corresponds to a real patient. **Precisely:** the pneumonia label itself is unchanged by a flip; what is corrupted is the anatomical context the image was labelled in. In our data the flip did **not** hurt measurably (inside noise on every metric): the pneumonia-vs-normal metric is blind to laterality, so this is an argument from anatomy, not an experimental finding. The "safe" policy therefore excludes flips (rotation ≤ 7°, shift ≤ 3%, zoom 0.9–1.1).

---

### Reproducibility
`pip install -r requirements.txt`, then `python run_all.py` (or `bash run_all.sh`). Part 1 scripts: `audit.py`, `split.py`, `shortcut.py`, `baselines.py`, `arch_report.py`, `train.py`, `aggregate.py`. Hypotheses are committed at `4f614ff` (see `reports/hypotheses_proof.txt`); the first real result file was written the same day at 21:43, after the 15:01 commit.
