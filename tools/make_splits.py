"""로컬 도구: 저자 분할을 그대로 재현하고, 설정 고르기용 검증 세트를 학습 쪽에서 뗀다 → splits/ (저장소에 커밋).

저자 코드 (trimodaldet/data/dataset.py) 와 같은 순서:
  images/ 의 .npy 를 이름순 정렬 → 라벨 파일이 있고 비어 있지 않은 것만 → train_test_split(test_size=0.2, random_state=42)
검증 세트 = 저자 train 7,800 장에서 낮·밤 (접두어) 마다 10 % 를 무작위로 (시드 0). test 와 train 의 관계 (무작위 프레임) 를 흉내 낸다.
작업 안에서는 src/data.py 가 저자 Dataset 으로 다시 나눠 이 목록과 같은지 확인한다.

사용: python -I tools/make_splits.py <data.zip>
"""
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent.parent / "splits"
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
npy = sorted(Path(n).name for n in names if n.startswith("data/images/") and n.endswith(".npy"))
lab = {Path(n).name: n for n in names if n.startswith("data/labels/") and n.endswith(".txt")}
valid = [f for f in npy if f.replace(".npy", ".txt") in lab and z.read(lab[f.replace(".npy", ".txt")]).decode("utf-8", "replace").strip()]
train, test = train_test_split(valid, test_size=0.2, random_state=42)

rng = np.random.default_rng(0)
val = set()
for pre in ("frame_", "nframe_"):
    g = sorted(f for f in train if f.startswith(pre))
    val |= set(rng.choice(g, size=round(len(g) * 0.1), replace=False).tolist())

OUT.mkdir(exist_ok=True)
with open(OUT / "author_split.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["stem", "split", "val"])
    for f in valid:
        s = "train" if f in set(train) else "test"
        w.writerow([f[:-4], s, int(f in val)])


def digest(files):
    return hashlib.sha256("\n".join(sorted(f[:-4] for f in files)).encode()).hexdigest()


info = {
    "_note": "tools/make_splits.py 가 data.zip 에서 만든 목록. train/test = 저자 분할 (random_state=42), val = train 안에서 낮·밤마다 10 % (시드 0).",
    "n_valid": len(valid), "n_train": len(train), "n_test": len(test), "n_val": len(val),
    "n_val_day": sum(f.startswith("frame_") for f in val), "n_val_night": sum(f.startswith("nframe_") for f in val),
    "sha256_train": digest(train), "sha256_test": digest(test), "sha256_val": digest(val),
}
(OUT / "split.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(info, ensure_ascii=False))
