"""Part 1, task 5: majority class and classical HOG / LBP + logistic regression."""
import numpy as np, pandas as pd
from skimage.feature import hog, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import *


def hog_feats(a): return np.stack([hog(x, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)) for x in a])


def lbp_feats(a, grid=4):
    out = []
    for x in a:
        l = local_binary_pattern(x, 8, 1, "uniform"); h = x.shape[0] // grid; f = []
        for i in range(grid):
            for j in range(grid):
                f.append(np.bincount(l[i*h:(i+1)*h, j*h:(j+1)*h].astype(int).ravel(), minlength=10) / (h * h))
        out.append(np.concatenate(f))
    return np.stack(out)


def main():
    set_seed(SEED); df = load_splits(); ix = get_idx(df, 0); y = df.label.values
    a = load_array(df.path.tolist(), 128)
    F = {}
    for name, fn in (("hog", hog_feats), ("lbp", lbp_feats)):
        f = CACHE / f"feat_{name}_{len(df)}.npy"
        if f.exists(): F[name] = np.load(f)
        else: F[name] = fn(a); np.save(f, F[name])
    rows = []
    def add(name, p_val, p_te):
        for s, i, p in (("val", ix["val"], p_val), ("test", ix["test"], p_te)):
            rows.append(dict(model=name, split=s, **compute_metrics(y[i], p)))
    prior = y[ix["train"]].mean()
    add("majority class", np.full(len(ix["val"]), prior), np.full(len(ix["test"]), prior))
    # NB: with constant scores thr=0.5 predicts the majority class (prior>0.5 -> all PNEUMONIA)
    for n in ("hog", "lbp"):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(F[n][ix["train"]], y[ix["train"]])
        add(f"{n.upper()} + LogReg", m.predict_proba(F[n][ix["val"]])[:, 1], m.predict_proba(F[n][ix["test"]])[:, 1])
    t = pd.DataFrame(rows)[["model", "split", "acc", "bal_acc", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]].round(3)
    t.to_csv(RESULTS / "baselines_table.csv", index=False)
    txt = "# Baselines (train folds 1-4, validate fold 0, locked test)\n" + t.to_markdown(index=False) + "\n"
    open(RESULTS / "baselines_summary.md", "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
