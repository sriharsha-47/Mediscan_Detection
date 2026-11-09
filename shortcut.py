"""Part 1, task 4: deliberately crippled models. If they beat chance by a lot, there is a shortcut."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import *


def main():
    set_seed(SEED); df = load_splits(); ix = get_idx(df, 0); y = df.label.values
    tr = np.concatenate([ix["train"], ix["val"]]); te = ix["test"]       # train on whole pool, evaluate on locked test
    paths = df.path.tolist()
    x16 = load_array(paths, 16).reshape(len(df), -1).astype(np.float32) / 255
    x64 = load_array(paths, 64)
    masked = x64.copy(); masked[:, 8:58, 8:56] = 0                         # blank out the central lung field
    xm = np.stack([m[::2, ::2] for m in masked]).reshape(len(df), -1).astype(np.float32) / 255
    meta = meta_features(df)
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for a, k in zip(ax, [0, len(df) // 2, len(df) - 1]):
        a.imshow(masked[k], cmap="gray"); a.axis("off"); a.set_title(CLASSES[y[k]])
    plt.savefig(FIGS / "shortcut_masked_examples.png", dpi=120); plt.close()

    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    rf = lambda: RandomForestClassifier(300, random_state=SEED, n_jobs=-1)
    rows = []
    maj = np.full(len(te), y[tr].mean())
    rows.append(dict(model="majority class", **compute_metrics(y[te], maj)))
    for name, X, mk in [("16x16 pixels + LogReg", x16, lr), ("16x16 pixels + RandForest", x16, rf),
                        ("lung field masked (border only) + LogReg", xm, lr),
                        ("metadata only (mean, aspect, area) + LogReg", meta, lr),
                        ("metadata only (mean, aspect, area) + RandForest", meta, rf)]:
        m = mk().fit(X[tr], y[tr]); rows.append(dict(model=name, **compute_metrics(y[te], m.predict_proba(X[te])[:, 1])))
    t = pd.DataFrame(rows)[["model", "acc", "bal_acc", "recall", "specificity", "roc_auc", "pr_auc"]].round(3)
    t.to_csv(RESULTS / "shortcut_table.csv", index=False)
    txt = ("# Shortcut hunt (trained on pool, evaluated on locked test)\n" + t.to_markdown(index=False) +
           f"\n\nTest-set pneumonia share = {y[te].mean():.3f}. Chance balanced accuracy = 0.5, chance ROC-AUC = 0.5.\n"
           "Read: if the crippled models are far above 0.5 balanced accuracy / AUC, brightness, framing or border "
           "information (not lung texture) already separates the classes.\n")
    open(RESULTS / "shortcut_summary.md", "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
