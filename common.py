"""Shared helpers: paths, seeding, manifest, image loading, metrics, threshold selection."""
import os, re, sys, json, random, hashlib
from pathlib import Path
import numpy as np, pandas as pd, cv2

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MEDISCAN_DATA", ROOT / "data")).resolve()
RESULTS, FIGS, MODELS, CACHE = ROOT / "results", ROOT / "figures", ROOT / "models", ROOT / ".cache"
for _d in (RESULTS, FIGS, MODELS, CACHE, RESULTS / "runs"):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
CLASSES = ["NORMAL", "PNEUMONIA"]          # label 0 / 1  (PNEUMONIA = positive)
IMG_EXT = {".jpeg", ".jpg", ".png"}
CAMP_PREV = 0.10                           # camp prevalence of pneumonia (9:1 NORMAL)
C_FN, C_FP = 20.0, 1.0                     # cost of missed pneumonia vs false alarm


# ------------------------------------------------------------------ seeding
def set_seed(seed, log=True):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    if log:
        with open(RESULTS / "seed_log.txt", "a") as f:
            f.write(f"{Path(sys.argv[0]).name}\tseed={seed}\n")


# ------------------------------------------------------------------ primary dataset manifest
def primary_root():
    for c in [DATA / "chest_xray", DATA / "chest_xray" / "chest_xray", DATA / "chest-xray-pneumonia" / "chest_xray"]:
        if (c / "train").exists():
            return c
    raise FileNotFoundError(f"Could not find chest_xray/train under {DATA}. Put the Kaggle download in data/chest_xray/ "
                            f"(train/ test/ val/ each with NORMAL/ PNEUMONIA/) or set MEDISCAN_DATA.")


def patient_id(name):
    """Kaggle file names: person123_bacteria_456.jpeg  /  IM-0115-0001.jpeg  /  NORMAL2-IM-0373-0001.jpeg"""
    m = re.match(r"(person\d+)_", name)
    if m: return m.group(1)
    m = re.search(r"IM-(\d{4})", name)
    if m: return "IM-" + m.group(1)
    return name


def phash(gray):
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return format(int("".join("1" if b else "0" for b in bits), 2), "016x")


def hex_bits(hs):
    return np.array([[int(c) for c in format(int(h, 16), "064b")] for h in hs], dtype=np.float32)


def hamming_matrix(hs):
    B = hex_bits(hs)
    return (64 - (B @ B.T + (1 - B) @ (1 - B).T)).astype(np.int16)


def _file_info(p):
    from PIL import Image
    with Image.open(p) as im:
        mode = im.mode
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    true_gray = True
    if mode not in ("L", "I;16", "1"):
        c = cv2.imread(str(p), cv2.IMREAD_COLOR).astype(np.int16)
        true_gray = bool(np.abs(c[..., 0] - c[..., 1]).max() <= 2 and np.abs(c[..., 1] - c[..., 2]).max() <= 2)
    h, w = g.shape
    return dict(mode=mode, true_gray=true_gray, h=h, w=w, aspect=w / h, area=h * w,
                mean=float(g.mean() / 255), std=float(g.std() / 255),
                md5=hashlib.md5(open(p, "rb").read()).hexdigest(), phash=phash(g))


def build_manifest():
    root = primary_root(); rows = []
    for split in ("train", "test", "val"):
        for ci, cname in enumerate(CLASSES):
            for p in sorted((root / split / cname).glob("*")):
                if p.suffix.lower() not in IMG_EXT: continue
                r = dict(path=p.relative_to(DATA).as_posix(), orig_split=split, label=ci, label_name=cname,
                         patient=patient_id(p.name))
                r.update(_file_info(p)); rows.append(r)
                if len(rows) % 500 == 0: print(f"  manifest: {len(rows)} images", flush=True)
    return pd.DataFrame(rows)


def load_manifest(force=False):
    f = RESULTS / "manifest.csv"
    if f.exists() and not force:
        return pd.read_csv(f, dtype={"phash": str})
    df = build_manifest(); df.to_csv(f, index=False); return df


def load_splits():
    f = RESULTS / "splits.csv"
    if not f.exists(): raise FileNotFoundError("Run split.py first.")
    return pd.read_csv(f, dtype={"phash": str})


def get_idx(df, val_fold=0, holdout_fold=None):
    """pool = train+val+test images re-split by patient. 'test' is locked. Folds 0-4 live inside the pool."""
    pool = (df.split == "pool").values; fold = df.fold.values
    val = pool & (fold == val_fold)
    hold = pool & (fold == holdout_fold) if holdout_fold is not None else np.zeros(len(df), bool)
    train = pool & ~val & ~hold
    return dict(train=np.where(train)[0], val=np.where(val)[0], test=np.where(df.split.values == "test")[0],
                hold=np.where(hold)[0])


# ------------------------------------------------------------------ image loading
def preprocess_gray(g, res, preproc="none"):
    x = cv2.resize(g, (res, res), interpolation=cv2.INTER_AREA)
    if preproc == "clahe": x = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(x)
    elif preproc == "hist": x = cv2.equalizeHist(x)
    return x


def read_gray(path):
    p = Path(path)
    if not p.is_absolute(): p = DATA / p
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if g is None: raise IOError(f"cannot read {p}")
    return g


def load_array(paths, res, preproc="none", key="primary"):
    h = hashlib.md5("|".join(map(str, paths)).encode()).hexdigest()[:8]
    f = CACHE / f"{key}_{res}_{preproc}_{len(paths)}_{h}.npy"
    if f.exists(): return np.load(f)
    arr = np.stack([preprocess_gray(read_gray(p), res, preproc) for p in paths]).astype(np.uint8)
    np.save(f, arr); return arr


def meta_features(df):
    """Three handcrafted numbers: mean intensity, aspect ratio, log image area."""
    return np.c_[df["mean"].values, df["aspect"].values, np.log(df["area"].values)].astype(np.float32)


# ------------------------------------------------------------------ metrics
def compute_metrics(y, p, thr=0.5):
    from sklearn import metrics as M
    y = np.asarray(y).astype(int); p = np.asarray(p, float); pred = (p >= thr).astype(int)
    tn, fp, fn, tp = M.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sd = lambda a, b: a / b if b else float("nan")
    rec, spec = sd(tp, tp + fn), sd(tn, tn + fp)
    return dict(thr=float(thr), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                acc=(tp + tn) / len(y), bal_acc=np.nanmean([rec, spec]), precision=sd(tp, tp + fp),
                recall=rec, specificity=spec, f1=sd(2 * tp, 2 * tp + fp + fn),
                roc_auc=M.roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
                pr_auc=M.average_precision_score(y, p))


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def cost_curve(y, p, prev=CAMP_PREV, cfn=C_FN, cfp=C_FP):
    """Expected cost per child at a given prevalence. TPR/FPR do not depend on prevalence, so the
    camp prevalence can be imposed on a dataset with a different class mix."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    cost = prev * cfn * (1 - tpr) + (1 - prev) * cfp * fpr
    return fpr, tpr, thr, cost


def choose_threshold(y, p, prev=CAMP_PREV, cfn=C_FN, cfp=C_FP):
    fpr, tpr, thr, cost = cost_curve(y, p, prev, cfn, cfp)
    i = int(np.argmin(cost)); t = float(thr[i])
    if not np.isfinite(t): t = float(np.max(p)) + 1e-6
    return t, float(cost[i])


def fmt(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


# ------------------------------------------------------------------ run bookkeeping
def run_seeds(tag):
    import glob
    return sorted(int(Path(f).stem[4:]) for f in glob.glob(str(RESULTS / "runs" / tag / "seed*.json")))


def load_run(tag, seed):
    d = RESULTS / "runs" / tag
    return json.load(open(d / f"seed{seed}.json")), np.load(d / f"seed{seed}.npz")


def best_seed(tag):
    """Seed with the highest VALIDATION ROC-AUC (never selected on the test set)."""
    return max(run_seeds(tag), key=lambda s: load_run(tag, s)[0]["val"]["roc_auc"])