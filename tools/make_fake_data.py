"""Synthetic stand-in data in data_fake/ so the whole pipeline can be smoke-tested in minutes. Results are meaningless."""
import numpy as np, cv2
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "data_fake"; rng = np.random.RandomState(0)

def img(sick, h, w):
    y, x = np.mgrid[0:h, 0:w]; im = 60 + 30 * np.exp(-(((x - w / 2) / (w * .5)) ** 2 + ((y - h / 2) / (h * .5)) ** 2))
    for cx in (.3, .7): im += (70 if sick else 40) * np.exp(-(((x - cx * w) / (w * .12)) ** 2 + ((y - .5 * h) / (h * .25)) ** 2))
    return np.clip(im + rng.randn(h, w) * 8 + (12 if sick else 0), 0, 255).astype(np.uint8)

for split, n in (("train", 70), ("test", 30), ("val", 4)):
    for cls in ("NORMAL", "PNEUMONIA"):
        d = R / "chest_xray" / split / cls; d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            sick = cls == "PNEUMONIA"; h, w = (rng.randint(500, 900), rng.randint(700, 1200)) if sick else (rng.randint(700, 1200), rng.randint(900, 1500))
            name = f"person{rng.randint(1, 400) if sick else i}_bacteria_{i}.jpeg" if sick else f"IM-{rng.randint(1, 1000):04d}-{i:04d}.jpeg"
            cv2.imwrite(str(d / name), img(sick, h // 4, w // 4))
for cls, sub in (("Normal", "images"), ("Viral Pneumonia", "images")):
    d = R / "covid" / cls / sub; d.mkdir(parents=True, exist_ok=True)
    for i in range(40): cv2.imwrite(str(d / f"{cls.split()[0]}-{i}.png"), img(cls != "Normal", 299, 299))
print("fake data in", R)
