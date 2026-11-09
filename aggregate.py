"""Collect every run into mean ± std tables and the ablation difference table (with an explicit noise verdict)."""
import json, glob
import numpy as np, pandas as pd
from common import *

MET = ["roc_auc", "pr_auc", "bal_acc", "recall", "specificity"]
PAIRS = [("arogyanet", "arogyanet_nores", "D1 residual connections REMOVED"),
         ("arogyanet", "arogyanet_nose", "D2 squeeze-excitation REMOVED"),
         ("arogyanet", "arogyanet_meta", "metadata branch ADDED (two-branch)"),
         ("arogyanet", "arogyanet_res96", "input resolution 224 -> 96"),
         ("arogyanet", "arogyanet_aug_none", "augmentation safe -> none"),
         ("arogyanet", "arogyanet_aug_naive", "augmentation safe -> naive (with horizontal flip)"),
         ("resnet18", "arogyanet", "ResNet18 (pretrained) vs ArogyaNet"),
         ("arogyanet_tuned", "arogyanet_tuned_cw", "class weights (training-side treatment)"),
         ("arogyanet_tuned", "arogyanet_tuned_clahe", "CLAHE preprocessing (internal data)")]


def main():
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "runs" / "*" / "seed*.json"))):
        r = json.load(open(f)); row = dict(tag=r["tag"], seed=r["seed"], params=r["params"], best_epoch=r["best_epoch"])
        for sp in ("val", "test"):
            for m in MET: row[f"{sp}_{m}"] = r[sp][m]
        rows.append(row)
    if not rows: print("no runs found"); return
    df = pd.DataFrame(rows); df.to_csv(RESULTS / "all_runs.csv", index=False)
    g = df.groupby("tag"); mean, std, n = g.mean(numeric_only=True), g.std(numeric_only=True, ddof=1).fillna(0), g.size()
    cols = [f"{sp}_{m}" for sp in ("val", "test") for m in MET]
    summ = pd.DataFrame({c: [f"{mean.loc[t, c]:.3f} ± {std.loc[t, c]:.3f}" for t in mean.index] for c in cols}, index=mean.index)
    summ.insert(0, "n_seeds", n); summ.insert(1, "params", mean["params"].astype(int))
    summ.to_csv(RESULTS / "summary_runs.csv")
    txt = ["# All runs, mean ± std over seeds (thresholds fixed at 0.5 here; threshold work is in threshold_cost.md)\n",
           summ.to_markdown(), "\n# Ablations (variant minus reference). Rule: a difference is 'INSIDE NOISE' if |Δ| < sqrt(std_ref² + std_var²).\n"]
    ab = []
    for ref, var, name in PAIRS:
        if ref not in mean.index or var not in mean.index: continue
        for sp in ("val", "test"):
            for m in ("roc_auc", "pr_auc", "bal_acc"):
                c = f"{sp}_{m}"; d = mean.loc[var, c] - mean.loc[ref, c]; s = np.hypot(std.loc[ref, c], std.loc[var, c])
                ab.append(dict(ablation=name, split=sp, metric=m, reference=f"{mean.loc[ref, c]:.3f}", variant=f"{mean.loc[var, c]:.3f}",
                               delta=f"{d:+.3f}", spread=f"±{s:.3f}", verdict="INSIDE NOISE" if abs(d) <= s else ("better" if d > 0 else "worse")))
    if ab:
        A = pd.DataFrame(ab); A.to_csv(RESULTS / "ablations.csv", index=False); txt.append(A.to_markdown(index=False))
    open(RESULTS / "summary_runs.md", "w").write("\n".join(txt)); print("\n".join(txt))


if __name__ == "__main__":
    main()