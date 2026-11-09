"""Training engine + CLI. Used for every deep model in the project (transfer baselines, ArogyaNet, ablations, CV, tuning)."""
import argparse, json, time, copy, sys
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from common import *
from arogyanet import build_model, count_params

DEFAULTS = dict(model="arogyanet", epochs=20, bs=32, lr=1e-3, wd=1e-4, dropout=0.2, res=224, aug="safe", preproc="none",
                pos_weight="none", early_stop=True, patience=5, workers=0, max_train=0, pretrained=True)


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def make_aug(name):
    """none | safe (small rotation, shift, zoom) | naive (= safe + HORIZONTAL FLIP, the medically wrong one) | intensity (safe + brightness/contrast)"""
    if name == "none": return None
    t = []
    if name == "naive": t.append(v2.RandomHorizontalFlip(0.5))
    t.append(v2.RandomAffine(degrees=10 if name == "naive" else 7, translate=(0.03, 0.03), scale=(0.9, 1.1)))
    if name == "intensity": t.append(v2.ColorJitter(brightness=0.4, contrast=0.4))
    return v2.Compose(t)


class DS(Dataset):
    def __init__(self, imgs, meta, y, idx, aug=None):
        self.imgs, self.meta, self.y, self.idx, self.aug = imgs, meta, y, idx, aug
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; x = torch.from_numpy(self.imgs[j]).float().div_(255).unsqueeze(0)
        if self.aug is not None: x = self.aug(x)
        return x, torch.from_numpy(self.meta[j]), torch.tensor(float(self.y[j]))


@torch.no_grad()
def predict_probs(model, imgs, meta, idx, device, bs=128):
    model.eval(); out = []
    for k in range(0, len(idx), bs):
        j = idx[k:k + bs]
        x = torch.from_numpy(imgs[j]).float().div_(255).unsqueeze(1).to(device)
        out.append(torch.sigmoid(model(x, torch.from_numpy(meta[j]).to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


@torch.no_grad()
def embed_all(model, imgs, meta, idx, device, bs=128):
    model.eval(); out = []
    for k in range(0, len(idx), bs):
        j = idx[k:k + bs]
        x = torch.from_numpy(imgs[j]).float().div_(255).unsqueeze(1).to(device)
        out.append(model.embed(x, torch.from_numpy(meta[j]).to(device)).cpu().numpy())
    return np.concatenate(out)


def load_ckpt(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location=device, weights_only=False); cfg = ck["cfg"]
    m = build_model(cfg["model"], cfg["dropout"], pretrained=False); m.load_state_dict(ck["state"]); return m.to(device).eval(), cfg


def run_one(cfg, seed, tag, val_fold=0, holdout_fold=None, save=True, verbose=True):
    cfg = {**DEFAULTS, **cfg}; set_seed(seed); device = get_device()
    df = load_splits(); y = df.label.values
    imgs = load_array(df.path.tolist(), cfg["res"], cfg["preproc"]); meta = meta_features(df)
    ix = get_idx(df, val_fold, holdout_fold); tr = ix["train"]
    if cfg["max_train"]: tr = np.random.RandomState(seed).choice(tr, min(cfg["max_train"], len(tr)), replace=False)
    model = build_model(cfg["model"], cfg["dropout"], cfg["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg["epochs"], 1))
    pw = None
    if cfg["pos_weight"] == "auto":
        npos = y[tr].sum(); pw = torch.tensor([(len(tr) - npos) / max(npos, 1)], dtype=torch.float32, device=device)
    crit, crit_eval = nn.BCEWithLogitsLoss(pos_weight=pw), nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(DS(imgs, meta, y, tr, make_aug(cfg["aug"])), batch_size=cfg["bs"], shuffle=True, drop_last=True,
                    num_workers=cfg["workers"], generator=g)
    yv = y[ix["val"]]; hist = []; best, best_ep, best_state, bad = 1e9, -1, None, 0; t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train(); tl, n = 0.0, 0
        for x, m_, yb in dl:
            x, m_, yb = x.to(device), m_.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(x, m_), yb); loss.backward(); opt.step()
            tl += loss.item() * len(yb); n += len(yb)
        sched.step()
        pv = predict_probs(model, imgs, meta, ix["val"], device)
        vl = float(crit_eval(torch.logit(torch.tensor(pv).clamp(1e-6, 1 - 1e-6)), torch.tensor(yv, dtype=torch.float32)))
        mv = compute_metrics(yv, pv)
        hist.append(dict(epoch=ep + 1, train_loss=tl / max(n, 1), val_loss=vl, val_roc_auc=mv["roc_auc"],
                         val_pr_auc=mv["pr_auc"], val_bal_acc=mv["bal_acc"]))
        if verbose: print(f"[{tag} s{seed}] ep{ep+1:02d} train {tl/max(n,1):.4f} val {vl:.4f} auc {mv['roc_auc']:.3f} bal {mv['bal_acc']:.3f}", flush=True)
        if vl < best - 1e-5: best, best_ep, bad = vl, ep + 1, 0; best_state = copy.deepcopy(model.state_dict())
        else: bad += 1
        if cfg["early_stop"] and bad >= cfg["patience"]: break
    if cfg["early_stop"] and best_state is not None: model.load_state_dict(best_state)
    probs = {k: predict_probs(model, imgs, meta, ix[k], device) for k in ("val", "test", "hold")}
    res = dict(tag=tag, seed=seed, cfg=cfg, params=count_params(model), best_epoch=best_ep, epochs_run=len(hist),
               history=hist, train_seconds=time.time() - t0, val_fold=val_fold, holdout_fold=holdout_fold,
               val=compute_metrics(yv, probs["val"]), test=compute_metrics(y[ix["test"]], probs["test"]))
    if save:
        d = RESULTS / "runs" / tag; d.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(d / f"seed{seed}.json", "w"), indent=1, default=float)
        np.savez(d / f"seed{seed}.npz", val_idx=ix["val"], val_prob=probs["val"], test_idx=ix["test"],
                 test_prob=probs["test"], hold_idx=ix["hold"], hold_prob=probs["hold"])
        torch.save(dict(state=model.state_dict(), cfg=cfg), MODELS / f"{tag}_s{seed}.pt")
    res["probs"], res["ix"] = probs, ix
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="arogyanet"); ap.add_argument("--tag"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--val_fold", type=int, default=0); ap.add_argument("--hparams"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no_early_stop", action="store_true"); ap.add_argument("--no_pretrained", action="store_true")
    for k, v in DEFAULTS.items():
        if k not in ("model", "early_stop", "pretrained"): ap.add_argument("--" + k, type=type(v), default=None)
    a = ap.parse_args(); cfg = dict(model=a.model)
    if a.hparams:
        hp = json.load(open(a.hparams)); cfg.update({k: hp[k] for k in ("lr", "bs", "dropout", "wd") if k in hp})
    for k in DEFAULTS:
        if getattr(a, k, None) is not None and k not in ("model",): cfg[k] = getattr(a, k)
    if a.no_early_stop: cfg["early_stop"] = False
    if a.no_pretrained: cfg["pretrained"] = False
    if a.quick: cfg.update(epochs=2, max_train=300)
    tag = a.tag or a.model
    for s in a.seeds: run_one(cfg, s, tag, a.val_fold)


if __name__ == "__main__":
    main()
