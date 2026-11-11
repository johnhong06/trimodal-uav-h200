"""로컬 도구: thermal · event 채널의 평균 · 표준편차를 저자 train 7,800 장에서 잰다 → stats/te_stats.json (저장소에 커밋).

논문 4.1 절: "thermal and event channels used statistics computed from the training split". 저자 코드는 이 값 대신 0.5 / 0.5 를 쓴다.
값은 0–1 범위 (픽셀 / 255). 작업 안에서는 src/stats.py 가 같은 식으로 다시 재서 이 파일과 같은지 확인한다.

사용: python -I tools/make_stats.py <data.zip>
"""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
z = zipfile.ZipFile(sys.argv[1])
train = [r["stem"] for r in csv.DictReader(open(ROOT / "splits" / "author_split.csv", encoding="utf-8")) if r["split"] == "train"]

s = np.zeros(2)
s2 = np.zeros(2)
n = 0
for stem in train:
    a = np.load(io.BytesIO(z.read(f"data/images/{stem}.npy")), allow_pickle=False)[..., 3:5].astype(np.float64) / 255.0
    s += a.sum(axis=(0, 1))
    s2 += (a * a).sum(axis=(0, 1))
    n += a.shape[0] * a.shape[1]
mean = s / n
std = np.sqrt(s2 / n - mean * mean)
out = {
    "_note": "thermal (채널 3) · event (채널 4) 평균 · 표준편차, 0–1 범위, 저자 train 7,800 장의 모든 픽셀. tools/make_stats.py",
    "n_images": len(train), "n_pixels": int(n),
    "thermal_mean": round(float(mean[0]), 6), "thermal_std": round(float(std[0]), 6),
    "event_mean": round(float(mean[1]), 6), "event_std": round(float(std[1]), 6),
}
(ROOT / "stats").mkdir(exist_ok=True)
(ROOT / "stats" / "te_stats.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
