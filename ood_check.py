"""Diagnostic: compare the OOD (Mahalanobis) score of probe images with real X-rays, so the guard threshold can be set from data."""
import json, numpy as np
from common import *
from predict import predict_path, _load

def main():
    S = _load(); thr = S["info"]["ood_maha_thr"]; df = load_splits()
    real = {}
    for name, part in (("val fold0", df[(df.split == "pool") & (df.fold == 0)]), ("locked test", df[df.split == "test"])):
        sub = part.sample(min(150, len(part)), random_state=SEED)
        real[name] = np.array([predict_path(DATA / p).get("maha", np.nan) for p in sub.path])
    print(f"current guard threshold: {thr:.1f}")
    for k, v in real.items():
        print(f"real X-rays ({k}): median {np.nanmedian(v):.1f}  p95 {np.nanpercentile(v,95):.1f}  p99 {np.nanpercentile(v,99):.1f}  max {np.nanmax(v):.1f}")
    print("probe images:")
    for f in sorted((ROOT / "tests" / "probes").glob("*.*")):
        r = predict_path(f); print(f"  {f.name:28s} status={r['status']:8s} maha={r.get('maha', float('nan')):8.1f}  label={r['label']}")

if __name__ == "__main__":
    main()