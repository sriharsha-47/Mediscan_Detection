"""Usage: python predict.py path/to/xray.png   ->  class + probability, with input guards and a refer-to-clinician band."""
import sys, json
import numpy as np, cv2, torch
from common import MODELS, preprocess_gray
from arogyanet import build_model

_STATE = {}


def _load():
    if _STATE: return _STATE
    info = json.load(open(MODELS / "final_meta.json")); ck = torch.load(MODELS / "final.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]; m = build_model(cfg["model"], cfg["dropout"], False); m.load_state_dict(ck["state"]); m.eval()
    o = np.load(MODELS / "ood_stats.npz"); _STATE.update(info=info, model=m, mu=o["mu"], P=o["P"]); return _STATE


def predict_path(path):
    S = _load(); info, model = S["info"], S["model"]
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None: return dict(status="rejected", label="INVALID INPUT", prob=None, message="Cannot read this file as an image.")
    c = bgr.astype(np.int16); colour = max(np.abs(c[..., 0] - c[..., 1]).mean(), np.abs(c[..., 1] - c[..., 2]).mean())
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if g.std() / 255 < info["min_std"]:
        return dict(status="rejected", label="INVALID INPUT", prob=None, message="Blank / uniform image - not a chest X-ray.")
    if colour > info["max_colour_diff"]:
        return dict(status="rejected", label="INVALID INPUT", prob=None, message="Colour photograph - not a chest X-ray.")
    h, w = g.shape
    x = torch.from_numpy(preprocess_gray(g, info["res"], info["preproc"])).float().div(255)[None, None]
    meta = torch.tensor([[g.mean() / 255, w / h, np.log(h * w)]], dtype=torch.float32)
    with torch.no_grad():
        z = model.embed(x, meta); p = float(torch.sigmoid(model.fc(z)).item())
    d = z.numpy()[0] - S["mu"]; maha = float(np.sqrt(d @ S["P"] @ d))
    if maha > info["ood_maha_thr"]:
        return dict(status="rejected", label="INVALID INPUT", prob=p, maha=maha, message=f"Image does not look like the X-rays this tool was built for (OOD score {maha:.1f}).")
    if p < info["thr_low"]: label, msg = "NORMAL", "No pneumonia flagged. This is a screening aid, not a diagnosis."
    elif p < info["thr_high"]: label, msg = "LOW CONFIDENCE", "LOW CONFIDENCE - refer to a clinician."
    else: label, msg = "PNEUMONIA (suspected)", "Refer for clinical confirmation."
    return dict(status="ok", label=label, prob=p, maha=maha, message=msg)


if __name__ == "__main__":
    r = predict_path(sys.argv[1])
    print(f"{r['label']}" + (f"  probability={r['prob']:.3f}" if r["prob"] is not None else "") + f"\n{r['message']}")