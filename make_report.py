"""Builds reports/Technical_Report.docx and reports/CEO_Memo.docx from results/*.md and figures/*.png.
Tables are copied from the auto-generated result files (never typed by hand); the prose cites those files.
Run:  pip install python-docx   then   python make_report.py"""
import re, glob, subprocess
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from common import ROOT, RESULTS, FIGS

REP = ROOT / "reports"; REP.mkdir(exist_ok=True)


# ------------------------------------------------------------------ helpers
def md_tables(name):
    f = RESULTS / name
    if not f.exists(): return []
    tabs, cur = [], []
    for line in open(f, encoding="utf-8"):
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip().replace("**", "") for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells if c) and any(cells): continue
            cur.append(cells)
        elif cur: tabs.append(cur); cur = []
    if cur: tabs.append(cur)
    return tabs


def pick(rows, cols=None, match=None, key=0, drop=None):
    """keep header + rows whose `key` column satisfies match(); keep only named columns"""
    h, body = rows[0], rows[1:]
    if match: body = [r for r in body if match(r[key] if len(r) > key else "")]
    if drop: body = [r for r in body if not drop(r)]
    if cols:
        idx = [h.index(c) for c in cols if c in h]; h = [h[i] for i in idx]; body = [[r[i] for i in idx] for r in body]
    return [h] + body


def new_doc(font=9.5):
    d = Document(); st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(font)
    for s in d.sections: s.left_margin = s.right_margin = Inches(0.7); s.top_margin = s.bottom_margin = Inches(0.6)
    return d


def H(d, t, lvl=1): d.add_heading(t, lvl)


def P(d, t, bold=False, italic=False):
    p = d.add_paragraph(); r = p.add_run(t); r.bold, r.italic = bold, italic; p.paragraph_format.space_after = Pt(4); return p


def B(d, items):
    for t in items:
        p = d.add_paragraph(style="List Bullet"); p.paragraph_format.space_after = Pt(1)
        if "::" in t: a, b = t.split("::", 1); p.add_run(a).bold = True; p.add_run(b)
        else: p.add_run(t)


def TODO(d, t):
    r = d.add_paragraph().add_run("TODO (write this yourself after looking at the figure): " + t); r.font.color.rgb = RGBColor(0xC0, 0, 0); r.bold = True


def T(d, rows, size=8, caption=None):
    if not rows or len(rows) < 2: P(d, "[table missing - run the script that produces it]", italic=True); return
    if caption: P(d, cap(caption), italic=True)
    t = d.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            if j >= len(rows[0]): continue
            cell = t.cell(i, j); cell.text = ""; run = cell.paragraphs[0].add_run(c); run.font.size = Pt(size); run.bold = (i == 0)
    d.add_paragraph().paragraph_format.space_after = Pt(2)


def F(d, name, caption, w=4.4):
    f = FIGS / name
    if f.exists(): d.add_picture(str(f), width=Inches(w)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    else: P(d, f"[missing figure figures/{name}]", italic=True)
    c = P(d, cap(caption), italic=True); c.alignment = WD_ALIGN_PARAGRAPH.CENTER


APP = []
CNT = {"Table": 0, "Figure": 0}


def cap(c):
    m = re.match(r"^(Table|Figure) \d+\.", c)
    if not m: return c
    CNT[m.group(1)] += 1; return re.sub(r"^(Table|Figure) \d+\.", f"{m.group(1)} {CNT[m.group(1)]}.", c)


def git_line():
    try:
        o = subprocess.check_output(["git", "log", "-1", "--format=%h, %ci", "--", "reports/hypotheses.md"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return o or "no commit found - commit reports/hypotheses.md"
    except Exception:
        return "git not available - add the commit hash and date yourself"


def deploy_text():
    f = RESULTS / "deploy.md"; return open(f, encoding="utf-8").read() if f.exists() else "[results/deploy.md missing]"


# ------------------------------------------------------------------ technical report
def technical():
    d = new_doc(); d.add_heading("MediScan Audit - Technical Report", 1)
    P(d, "Arogya Diagnostics - paediatric chest X-ray pneumonia screening. Author: [YOUR NAME]. "
         "Every table below is copied from a file in results/ produced by a script in this repository (python run_all.py); "
         "figures are in figures/. Seeds are logged in results/seed_log.txt. Environment: Python 3.12, PyTorch 2.14.1 (CUDA 12.6), RTX 4050 laptop GPU.", italic=True)

    H(d, "1. Summary")
    B(d, ["The wrong instruction::  Instruction 2, \"99% accuracy\", is the wrong target. At the camp's 9:1 NORMAL:PNEUMONIA mix a tool that flags nobody is 90% accurate, "
          "and a 99%-accurate tool can still miss 10 of 100 sick children (cost 200 per 1,000 children) or raise 10 false alarms (cost 10): accuracy cannot tell these apart (Section 5). "
          "Instructions 1 (evidence) and 3 (200 ms, 25 MB) were reasonable and are met (Section 10).",
          "What the audit found::  the original data has a 16-image validation set (an observed 16/16 is compatible with a true accuracy as low as 79.4%), "
          "a class mix that shifts between splits (74% pneumonia in train, 62.5% in test, 50% in val), strong non-lung shortcuts (image size/aspect ratio), "
          "and modest near-duplicate leakage (15 of 624 test images, 2.4%). We found no patient overlap between train and test by file-name ID. "
          "The 98.4% cannot be reproduced or verified (no code, split or weights).",
          "What we built::  ArogyaNet, a 242,319-parameter CNN trained from scratch. On a patient-disjoint locked test set it reaches ROC-AUC 0.989 (mean of 3 seeds), "
          "sensitivity 94.9% and specificity 95.6% at the cost-optimal threshold. At 10% prevalence that is about 5 missed and 40 falsely flagged children per 1,000.",
          "How far it generalises::  on an external hospital (adult-dominated, PA view) ROC-AUC fell from 0.989 to 0.899 and specificity from 95.6% to 72.6% at the same threshold. "
          "The CLAHE fix did not help (AUC 0.852).",
          "Honest limitations::  a pretrained MobileNetV3 and ResNet18 beat ArogyaNet in-distribution; the squeeze-excitation hypothesis was not supported; "
          "the sensitivity target of 95% is not demonstrably met (95% CI 93-97%)."])

    H(d, "2. Data audit (Part 1, tasks 1-3)")
    a = md_tables("audit_summary.md")
    P(d, "The primary data are 5,856 JPEGs (1,583 NORMAL, 4,273 PNEUMONIA); 283 are stored as RGB but all 5,856 have identical channels (true greyscale content). "
         "The two classes differ strongly in image geometry, not brightness:")
    if len(a) > 0: T(d, a[0], caption="Table 1. Inventory in Rohit's original split (results/audit_summary.md).")
    if len(a) > 2: T(d, a[2], caption="Table 2. Can one metadata number separate the classes? Single-feature ROC-AUC (0.5 = no information; values far from 0.5 in either direction = shortcut).")
    B(d, ["Median height is 776 px for PNEUMONIA vs 1,323 px for NORMAL, and median aspect ratio 1.49 vs 1.24 - the images were assembled from different sources/crops.",
          "Height alone gives AUC 0.085 (i.e. 0.915 with the sign flipped), area 0.090, width 0.111, aspect ratio 0.84. Mean brightness is NOT a shortcut here (AUC 0.508, my initial expectation was wrong); contrast (std) is a weaker one (0.302).",
          "Implication: a network can score well by recognising crop/scanner format. At the camp, framing and size are set by different hardware, so such cues will not transfer."])
    F(d, "audit_class_histograms.png", "Figure 1. Brightness, contrast and size distributions by class.")
    P(d, "Duplicates and leakage. ", bold=True)
    B(d, ["30 groups of byte-identical files (32 redundant files), none spanning two original splits.",
          "Perceptual hash (pHash, Hamming distance <= 4 of 64): 239 near-duplicate pairs, 218 within a split and 21 across splits; 15 of 624 test images (2.4%) have a near-duplicate in train. "
          "The mean pixel correlation of the cross-split pairs is only 0.876, so some are similar-looking X-rays rather than copies (Figure 2).",
          "Patient overlap: a naive file-name parser reports 264 shared IDs, but these are collisions (e.g. person100_virus in train vs person100_bacteria in test). "
          "Using person+pathogen keys for PNEUMONIA and series-aware keys for NORMAL, 0 IDs appear in both train and test. File names cannot prove patient-level separation for NORMAL images."])
    if len(a) > 3: T(d, a[3], caption="Table 3. Near-duplicate pairs by split.")
    F(d, "audit_worst5_train_test_pairs.png", "Figure 2. The five closest train/test image pairs.", w=2.6)
    TODO(d, "state for each of the 5 pairs whether it is a true copy or just two similar X-rays.")
    P(d, "The validation set. ", bold=True)
    P(d, "The original val/ folder has 16 images (8 per class). An observed 16/16 = 100% has an exact (Clopper-Pearson) 95% CI of [0.794, 1.000]; "
         "Rohit's 98.4% on 624 test images corresponds to about 10 errors (CI [0.971, 0.992] from sampling noise alone).")
    P(d, "Replacement design. All 5,856 images were pooled and re-split by group (patient ID merged with near-duplicate clusters; 2,778 groups, none shared across splits - asserted in split.py): "
         "a locked test set of 837 images (226 NORMAL / 611 PNEUMONIA, 73.0% pneumonia) and 5 stratified folds of about 1,004 images (72.9-73.0% pneumonia) on the remaining pool. "
         "Fold 0 is the validation set for every decision; the five folds give mean ± std for Part 2. 1,254 images changed split. "
         "Rejected alternatives: a single train/val/test split alone (every decision overfits one slice) and k-fold alone (no untouched test set, so selection bias).")

    H(d, "3. Shortcut hunt and classical baselines (tasks 4-5)")
    T(d, pick(md_tables("shortcut_summary.md")[0]) if md_tables("shortcut_summary.md") else [], caption="Table 4. Deliberately crippled models, trained on the pool, evaluated on the locked test set.")
    B(d, ["A 16x16 thumbnail (256 numbers) with logistic regression reaches test AUC 0.975, close to ArogyaNet's 0.989. A thumbnail still shows coarse lung opacity, so this alone is not proof of a shortcut.",
          "The stronger evidence: with the lung field masked out (border pixels only) AUC is 0.923, and the three metadata numbers (mean brightness, aspect ratio, area) alone give AUC 0.916-0.922. Information outside the lungs identifies the class.",
          "Consequence: high accuracy on this dataset does not show that a model reads lung tissue."])
    APP.append(lambda: F(d, "shortcut_masked_examples.png", "Figure B1. Examples of the lung-masked inputs.", w=3.0))
    b = md_tables("baselines_summary.md")
    if b: T(d, b[0], caption="Table 5. Baselines (trained on folds 1-4, validated on fold 0, then locked test).")
    P(d, "Bar for deep learning: HOG + logistic regression already reaches test AUC 0.981 and balanced accuracy 0.932, so the best deep model improves AUC by under 2 points.")

    H(d, "4. Models and ablations (tasks 6-9)")
    r = md_tables("summary_runs.md")
    keep = ["resnet18", "mobilenetv3", "arogyanet", "arogyanet_nores", "arogyanet_nose", "arogyanet_meta", "arogyanet_res96",
            "arogyanet_aug_none", "arogyanet_aug_naive", "arogyanet_tuned", "arogyanet_tuned_cw", "arogyanet_tuned_clahe"]
    if r:
        T(d, pick(r[0], ["tag", "n_seeds", "params", "test_roc_auc", "test_pr_auc", "test_bal_acc", "test_recall", "test_specificity"], match=lambda x: x in keep),
          caption="Table 6. Mean ± std over 3 seeds, locked test set, threshold 0.5 (decisions were made on validation fold 0; test numbers are for reporting).")
    P(d, "Transfer baseline. ", bold=True)
    P(d, "Expectation (written before running, reports/hypotheses.md, commit " + git_line() + "): ImageNet low-level filters (edges, textures) transfer to radiographs, "
         "high-level object features do not, so pretrained backbones should be at least as good as ArogyaNet without a wide margin. Result: ResNet18 test AUC 0.995 and MobileNetV3 0.994 vs ArogyaNet 0.988; "
         "balanced accuracy 0.968 / 0.962 vs 0.945. The expectation held; the gap is small in AUC but larger in balanced accuracy. "
         "ResNet18 has 11.2 M parameters (about 45 MB as float32, over the 25 MB channel); MobileNetV3-small has 0.93 M (about 4 MB).")
    P(d, "ArogyaNet. ", bold=True)
    P(d, "242,319 parameters (limit 2 M), no pretrained weights. Stem + five blocks of two depthwise-separable convolutions, each with squeeze-and-excitation (D2) and a residual shortcut (D1), then global average pooling and a linear head "
         "(architecture table: results/arch_table.md, diagram below).")
    F(d, "arogyanet_diagram.png", "Figure 4. ArogyaNet. Green arcs: residual shortcuts. Each block: 2x(depthwise 3x3 + pointwise 1x1) -> SE -> add -> ReLU6.")
    sc = [["Hypothesis (written before results)", "Observed (test, 3 seeds)", "Verdict"],
          ["D1 residual connections: small gain, probably inside noise", "Removing them: AUC -0.004 (spread ±0.001), balanced accuracy -0.013 (±0.007); on validation mostly inside noise", "Small but consistent effect on test; weak on validation"],
          ["D2 squeeze-excitation: about +1 pp PR-AUC / specificity", "Removing it: every metric inside noise (AUC 0.988 vs 0.988); costs 24k parameters", "NOT supported"],
          ["Pretrained >= ArogyaNet, not by a wide margin", "ResNet18/MobileNetV3 AUC +0.007 / +0.006, balanced accuracy +0.022 / +0.017", "Supported"],
          ["Metadata branch helps internally", "AUC +0.002 (±0.002), all metrics inside noise", "NOT supported"],
          ["Flip augmentation changes nothing internally", "All metrics inside noise", "Supported"]]
    T(d, sc, size=8, caption="Table 7. Hypotheses vs results. Noise rule: |difference| < sqrt(std_a² + std_b²); with only 3 seeds this rule is crude.")
    ab = md_tables("summary_runs.md")
    if len(ab) > 1:
        T(d, pick(ab[1], ["ablation", "metric", "reference", "variant", "delta", "spread", "verdict"], match=lambda x: x == "test", key=ab[1][0].index("split")),
          caption="Table 8. Ablations on the locked test set (all metrics: Appendix A).", size=7)
    P(d, "Two-branch experiment. ", bold=True)
    P(d, "Adding mean intensity, aspect ratio and area through a small dense branch did not improve results beyond noise (test AUC 0.990 vs 0.988). This does not show the metadata is useless: "
         "alone it separates the classes with AUC about 0.92 (Table 4), so the image branch can already read equivalent cues (framing, borders, intensity layout). "
         "We therefore cannot claim the model learned something specific about patients - only that its signal is not distinguishable from these cues. "
         "At the camp such cues would not hold: the external hospital's images are all 299x299 (KS statistic for width = 1.0), and size and framing depend on the scanner.")
    P(d, "Resolution and augmentation. ", bold=True)
    B(d, ["96 px vs 224 px: validation inside noise; on test AUC -0.003 (±0.003) and balanced accuracy -0.012 (±0.006). A cheaper 96 px model loses little.",
          "No augmentation vs small rotations/shifts/zoom: no augmentation is worse on test (AUC -0.006 ±0.001, PR-AUC -0.003, balanced accuracy -0.011).",
          "The medically wrong augmentation is the horizontal flip (and vertical flip). It places the heart on the right (an anatomy that is rare or implausible, dextrocardia) and swaps the meaning of the left/right laterality marker. "
          "Strictly, the pneumonia label itself is unchanged by a flip - what is corrupted is the anatomical context the image was labelled in. "
          "In our data the flip did not hurt measurably (inside noise on every metric): the benchmark metric is blind to laterality, so this is an argument from anatomy, not an experimental finding."])

    H(d, "5. Imbalance, threshold and cost (Part 2.1)")
    tc = md_tables("threshold_cost.md")
    P(d, "Class weights vs threshold moving. Class weights produced a different network (relative weight distance 0.25-0.28, validation probability rank correlation 0.91-0.97, test AUC 0.984 vs 0.989). "
         "Moving the threshold leaves the network and its AUC unchanged and only changes the confusion matrix. At a validation-chosen \"95% sensitivity\" threshold, test sensitivity was only 0.91-0.94: "
         "a sensitivity target set on validation does not transfer exactly.")
    if tc: APP.append(lambda: T(d, tc[0], caption="Table B1. Class-weight (training-side) vs threshold-moving comparison, per seed.", size=7))
    P(d, "Cost matrix: a missed pneumonia costs 20, a false alarm 1; camp prevalence 10% pneumonia. Expected cost per child = 0.1*20*(1-TPR) + 0.9*1*FPR. "
         "The cost-optimal threshold chosen on validation is 0.515 - close to 0.5. This is expected: the 20x miss cost pushes the threshold down, while moving from the training prevalence (73%) to the camp (10%) pushes it up; "
         "the analytic prior-shift value (0.548) agrees. The model's probabilities are very confident, so the cost curve is flat around the optimum.")
    if len(tc) > 1: T(d, tc[1], caption="Table 10. Test-set confusion and camp-prevalence metrics at three thresholds.", size=7)
    ca = md_tables("ceo_arithmetic.md")
    if ca: T(d, pick(ca[0], ["scenario", "sick_missed", "healthy_flagged", "accuracy", "sensitivity", "PPV", "NPV", "cost"]), size=7, caption="Table 11. Per 1,000 children at the camp (100 sick, 900 healthy): why accuracy hides the cost of missed cases (results/ceo_arithmetic.md).")
    F(d, "threshold_cost_roc_pr.png", "Figure 5. Cost vs threshold (validation), ROC and precision-recall (test) with the chosen point; PR also re-weighted to camp prevalence.")

    H(d, "6. Evaluation and cross-validation (Part 2.2)")
    cv = md_tables("cv_table.md")
    if len(cv) > 1: T(d, cv[1], caption="Table 11. 5-fold stratified, group-aware CV at the fixed threshold 0.515 vs the single locked-test model.")
    P(d, "Locked test confusion matrix at the chosen threshold: TN = 216, FP = 10, FN = 31, TP = 580. Cross-validated sensitivity (0.977 ± 0.004) is higher than the locked-test value (0.949); "
         "the three seeds' test sensitivities range from 0.941 to 0.964 and the reported seed was selected on validation, so part of the gap is seed variation; the threshold was also chosen on fold 0, a small optimistic bias for CV. "
         "Why accuracy alone is the wrong headline: at 9:1 prevalence a model that never flags anyone is 90% accurate while missing every sick child, and accuracy weighs a miss like a false alarm although a miss is about 20x costlier.")

    H(d, "7. Tuning and regularisation (Part 2.3)")
    tu = md_tables("tuning.md")
    P(d, "Budget: random search, 12 trials x 8 epochs, one seed, validation fold 0; learning rate (log-uniform 1e-4 to 3e-3), batch size {16, 32, 64}, dropout {0, 0.1, 0.3, 0.5}, weight decay (log-uniform 1e-6 to 1e-2); selection by lowest validation loss. "
         "Chosen: lr 8.4e-4, batch 32, dropout 0.1, weight decay about 1e-6.")
    if tu: APP.append(lambda: T(d, tu[0], caption="Table B2. Top five search trials.", size=7))
    P(d, "The best trials differ very little (validation loss 0.097-0.102), the chosen weight decay is almost zero, and the tuned model is not better than the default beyond noise (test AUC 0.989 vs 0.988), so the search is best read as confirming the defaults.")
    F(d, "curves_unreg_vs_reg.png", "Figure 6. Unregularised (no augmentation, dropout 0, weight decay 0, no early stopping) vs regularised run.")
    P(d, "Overfitting begins right after epoch 4 in the unregularised run (validation loss 0.112, rising to 0.175 by epoch 30 while training loss falls to 0.0004). The regularised run reached its best epoch at 9 (validation loss 0.083) and early stopping ended it at epoch 14 of 30, "
         "saving 16 epochs and avoiding a validation-loss rise of 0.063. The two runs differ in several settings at once, so the saving cannot be credited to early stopping alone.")

    H(d, "8. Ensembles (Part 2.4)")
    en = md_tables("ensemble_summary.md")
    if en:
        T(d, pick(en[0], ["model", "val_roc", "test_roc", "test_pr", "test_bal_acc", "test_sens", "test_spec"],
                  match=lambda x: x in ("arogyanet_s0", "resnet18_s0", "mobilenetv3_s0") or x.startswith("SOFT") or "on CNN features" in x, drop=lambda r: "all members" in r[0]),
          caption="Table 13. Single members, soft-voting ensembles and tree models on CNN features (full list in results/ensemble_summary.md; the 'all members' row is mathematically identical to the family average and omitted).")
    P(d, "The three-family soft vote (test AUC 0.9955) equals its best member (ResNet18, 0.9952) within seed noise; member probabilities correlate at 0.96 and member errors at 0.52, so averaging removes little variance. "
         "Random forest and AdaBoost on the penultimate-layer features were worse than the network's own head (AUC 0.980 and 0.981 vs 0.989), consistent with fitting on embeddings the CNN has already fitted.")

    H(d, "9. Feature-space analysis (Part 2.5)")
    em = md_tables("embeddings_summary.md")
    if em: T(d, em[0], caption="Table 14. Errors vs correct predictions on the locked test set (model arogyanet_tuned, seed 0).", size=7)
    P(d, "k-means with k = 2 on the penultimate embeddings matches the labels well (adjusted Rand index 0.828, purity 0.957 vs 0.730 for the majority class). "
         "Missed pneumonia cases (31) are 61% viral and 39% bacterial, roughly twice the viral share among all pneumonia cases (about 31%), and are confidently wrong (mean probability 0.20), so a lower threshold would not recover most of them. "
         "The 10 false alarms are all NORMAL images with mean probability 0.84 and slightly higher brightness and area; with n = 10 this is only suggestive.")
    F(d, "embeddings_pca_tsne.png", "Figure 7. PCA and t-SNE of test embeddings by true class and by correct/incorrect.", w=4.2)
    APP.append(lambda: F(d, "misclassified_examples.png", "Figure B2. Examples of misclassified test images.", w=5.0))
    TODO(d, "look at figures/misclassified_examples.png (Figure B2, Appendix B) and describe what these images have in common (exposure, rotation, small opacity, etc.).")

    H(d, "10. External-hospital generalisation, run once (Part 2.6)")
    ex = md_tables("external_summary.md")
    P(d, "The external folder holds 11,537 images (10,192 Normal, 1,345 Viral Pneumonia; 11.7% pneumonia), all used. It is touched only by external_eval.py, which refuses a second run. Thresholds were fixed on internal validation data.")
    if ex: T(d, ex[0], caption="Table 15. Internal test vs external hospital.", size=7)
    B(d, ["Drop: ROC-AUC 0.989 -> 0.899 (-0.090); specificity 0.956 -> 0.726; balanced accuracy 0.953 -> 0.840. Sensitivity stayed 0.953, but the external positives are viral pneumonia in adults, so this does not show that sensitivity generalises.",
          "At camp prevalence PPV fell from 0.70 to 0.28 and expected cost per child rose from 0.14 to 0.34 - still better than flagging nobody (2.0) or everybody (0.9). PR-AUC is 0.423 against a chance level of about 0.12.",
          "Distribution mismatch (pixel statistics from Part 1): every external image is 299x299 whereas internal images average about 1,400x1,700 (NORMAL) and 820x1,200 (PNEUMONIA) pixels (KS statistics: height 0.987, width 1.0); the contrast distribution differs moderately (KS 0.238) and mean brightness little (0.122). "
          "Internally, geometry alone separated the classes (Table 2), a cue that cannot exist in a uniformly sized external set. Other differences are population (children 1-5 years vs adults), projection (AP vs PA) and scanner/processing. These are consistent explanations; we did not test them causally."])
    F(d, "external_pixel_shift.png", "Figure 9. Per-image mean and contrast: internal vs external.", w=4.0)
    P(d, "Fix attempted: CLAHE preprocessing, retrained on internal data. It did not help: external AUC 0.852 (vs 0.899), specificity 0.668 (vs 0.726). Internally it was inside noise. Not adopted.")

    H(d, "11. Shipping (Part 2.7)")
    bd = md_tables("band.md")
    P(d, "Decision band (set_band.py, thresholds from validation data only): NORMAL below 0.461, LOW CONFIDENCE from 0.461 to 0.885 (refer to a clinician), PNEUMONIA suspected from 0.885. Only the NORMAL zone clears a child.")
    if bd: T(d, bd[0], caption="Table 16. Zones on the locked test set, per 1,000 camp children.")
    P(d, "Validation had suggested 98% sensitivity for the NORMAL cut; on the test set the referred group has sensitivity 0.951 (95% CI 0.931-0.967). At camp prevalence the NORMAL zone has NPV 0.994, 13.9% of children are referred and the PNEUMONIA zone has PPV 0.848.")
    P(d, deploy_text().replace("# ", "").replace("Decision rule: p < 0.515 -> NORMAL;  0.515 <= p < 0.515 -> LOW CONFIDENCE, refer to clinician;  p >= 0.515 -> PNEUMONIA suspected.", "(Decision band: see the zone table above.)"), italic=True)
    rj = md_tables("rejection_test.md")
    P(d, "Non-X-ray inputs. A first version let a greyscale dog photo through as NORMAL (embedding distance 22.7 vs the then threshold of 25.0, with real X-rays reaching 24.0). Lowering the threshold to 20.0, chosen from validation distances (real-X-ray 99th percentile 17.4-19.3), rejects it; "
         "the cost is that roughly 1% of real X-rays may be rejected and need to be retaken. This is a mitigation, not a guarantee.")
    if rj: T(d, rj[0], caption="Table 17. Rejection test (results/rejection_test.md): no non-X-ray received a medical answer; the six real X-ray controls did.", size=7)
    shots = sorted(glob.glob(str(REP / "screenshots" / "*.png")) + glob.glob(str(REP / "screenshots" / "*.jpg")))
    if shots:
        t = d.add_table(rows=(len(shots) + 1) // 2, cols=2)
        for i, s in enumerate(shots):
            c = t.cell(i // 2, i % 2); c.paragraphs[0].add_run().add_picture(s, width=Inches(2.2))
        P(d, cap("Figure 10. Demo screenshots (app.py): real X-rays, a puppy photo and a blank image."), italic=True)
    else: P(d, "[no screenshots found in reports/screenshots/]", italic=True)
    P(d, "Must not be used for: diagnosis; adults; PA or non-AP views; other scanners or hospitals without re-validation; children with symptoms who should be examined regardless; any setting without a clinician in the loop.")

    H(d, "12. Limitations and honest notes")
    B(d, ["Selection: hyper-parameters, the threshold and the band were chosen on validation fold 0 only; the test set was not used for any decision but was evaluated for every run.",
          "The deployed model is ArogyaNet because the external test was run on it. MobileNetV3 is stronger in-distribution (AUC 0.994) and small enough to ship, but its external performance is unknown and the external set cannot be reused.",
          "Three seeds give crude noise estimates; the \"inside noise\" rule is a heuristic, not a significance test.",
          "File-name IDs are weak evidence about patients; the 0-overlap finding is a lower bound on leakage risk.",
          "The 98.4% of the previous model cannot be reproduced; we do not claim to know how it was obtained."])

    H(d, "13. Reproducibility")
    P(d, "pip install -r requirements.txt; python run_all.py runs every script in order (audit, split, shortcut, baselines, arch_report, training, aggregate, tune, threshold_cost, ceo_numbers, cv_eval, ensemble, embeddings, make_final, reject_test, external_eval, build_report). "
         "set_band.py and make_report.py are run afterwards; the out-of-distribution threshold in models/final_meta.json was set to 20.0 after ood_check.py. "
         "Seeds 0, 1, 2 are logged in results/seed_log.txt.")

    d.add_page_break(); H(d, "Appendix A. All ablations (validation and test)")
    if len(ab) > 1: T(d, pick(ab[1], ["ablation", "split", "metric", "reference", "variant", "delta", "spread", "verdict"]), size=7)
    if APP:
        H(d, "Appendix B. Supporting tables and figures")
        for fn in APP: fn()
    d.save(REP / "Technical_Report.docx"); print("wrote reports/Technical_Report.docx")


# ------------------------------------------------------------------ CEO memo
def memo():
    d = new_doc(9.5); d.add_heading("Memo to the CEO - MediScan", 1)
    P(d, "From: [YOUR NAME], ML engineer.   Re: Rohit's model, the rebuild, and why \"99% accuracy\" is the wrong target.", italic=True)
    H(d, "What the 98.4% really was", 2)
    P(d, "It cannot be verified: there is no code, split record or weights. What we can show is why any single number from this data deserves distrust: the validation set had 16 images (a perfect score there is compatible with 79% true accuracy); "
         "the test set had a different disease mix from training (62.5% vs 74% pneumonia); 15 of 624 test images have a near-duplicate in training; and image size and shape alone separate pneumonia from normal images with ROC-AUC about 0.9, "
         "so a model can score well without looking at lungs. The pilot - about 7 in 10 healthy children flagged - is what such a model does on different equipment.")
    H(d, "What the rebuilt model achieves", 2)
    P(d, "On a locked test set of 837 children's X-rays, split by patient: ROC-AUC 0.989, sensitivity 94.9% and specificity 95.6% at the operating point chosen by cost (a missed pneumonia counts 20x a false alarm). "
         "Per 1,000 children at the camp (10% sick): about 95 of 100 sick children are flagged, about 5 are missed and about 40 healthy children are referred unnecessarily; 3 in 10 referrals are healthy. "
         "It runs in about 130 ms per image on a laptop CPU (limit 200 ms) and the file is 1.05 MB (limit 25 MB). Both operating requirements are met.")
    t = md_tables("ceo_arithmetic.md")
    if t: T(d, pick(t[0], ["scenario", "sick_missed", "healthy_flagged", "accuracy", "sensitivity", "PPV", "cost"]), size=8, caption="Per 1,000 children at the camp (cost = 20 per missed child + 1 per false alarm).")
    H(d, "Why \"99% accuracy\" is the wrong target", 2)
    P(d, "At the camp only about 1 child in 10 is sick. A tool that flags nobody is already 90% accurate and misses every sick child. A tool at 99% accuracy makes 10 errors per 1,000 children: if all are missed cases the cost is 200, "
         "if all are false alarms the cost is 10 - a 20-fold difference that accuracy does not show. Our model's accuracy is lower than 99% (95.5%) yet its cost (141) beats the 'all missed' 99% tool. "
         "Accuracy also moves with who is screened: the same model scores differently at 73% and 10% prevalence.")
    P(d, "Ask for this instead: a minimum sensitivity (for example 95%, judged by the lower end of a 95% confidence interval), a maximum share of healthy children referred, and the PPV and NPV at 9:1 prevalence - measured on patient-disjoint data from other hospitals and, before roll-out, on the camp's own devices.", bold=True)
    H(d, "What it must not be used for", 2)
    B(d, ["Diagnosis, or as a reason to send home a child who has symptoms.", "Adults, non-AP views, or X-ray machines and hospitals it has not been re-tested on: on an external hospital the AUC fell from 0.989 to 0.899 and healthy-child specificity from 96% to 73%.",
          "Any use without a clinician reviewing the \"LOW CONFIDENCE\" and \"PNEUMONIA\" results.", "Photos of screens, other images, or non-X-rays: the tool rejects most of these, but this is a safeguard, not a guarantee."])
    P(d, "Your three instructions: (1) evidence on what Rohit did wrong - delivered (technical report, Sections 2-3); (2) 99% accuracy - rejected, for the reasons above; (3) 200 ms and 25 MB - met (about 130 ms, 1.05 MB).", italic=True)
    d.save(REP / "CEO_Memo.docx"); print("wrote reports/CEO_Memo.docx")


if __name__ == "__main__":
    technical(); memo()