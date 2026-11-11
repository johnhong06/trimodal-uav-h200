"""데이터: 저자 Dataset (trimodaldet.data.dataset.NpyYoloDataset) 을 그대로 쓰고 분할만 고른다.

분할
  full  저자 train 7,800 장 (논문과 같음)
  test  저자 test 1,950 장 (논문과 같음 — 최종 숫자는 여기서만)
  sub   full 에서 val 을 뺀 7,020 장 (설정 고르기용 학습)
  val   full 안의 780 장 (splits/author_split.csv 의 val=1, 낮·밤마다 10 %)
저자 Dataset 은 픽셀을 0–255 float 로 주고 (/255 없음), 정규화는 Faster R-CNN 의 image_mean · image_std 가 한다.

  python src/data.py check   # 저자 Dataset 으로 다시 나눈 결과가 splits/author_split.csv 와 같은지 (작업 안 확인)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import Dataset  # noqa: E402

from common import REPO, author, need  # noqa: E402


def _author_ds(mode):
    d = need("TRI_DATA")
    return author().data.dataset.NpyYoloDataset(os.path.join(d, "images"), os.path.join(d, "labels"),
                                                 mode=mode, test_size=0.2, random_state=42)


def split_table():
    with open(os.path.join(REPO, "splits", "author_split.csv"), encoding="utf-8") as f:
        return {r["stem"]: (r["split"], r["val"] == "1") for r in csv.DictReader(f)}


class Named(Dataset):
    """저자 Dataset 의 일부 + 사진 이름 (낮·밤 구분 · 예측 저장용)."""

    def __init__(self, base, idx):
        self.base, self.idx = base, idx
        self.stems = [base.image_files[i][:-4] for i in idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


def make(split, limit=0):
    if split == "test":
        base = _author_ds("test")
        idx = list(range(len(base)))
    else:
        base = _author_ds("train")
        if split == "full":
            idx = list(range(len(base)))
        else:
            tab = split_table()
            idx = [i for i, f in enumerate(base.image_files) if tab[f[:-4]][1] == (split == "val")]
    if limit:
        idx = idx[:limit]
    return Named(base, idx)


def collate(batch):
    return tuple(zip(*batch))


def check():
    tab = split_table()
    tr, te = _author_ds("train"), _author_ds("test")
    got = {f[:-4]: "train" for f in tr.image_files} | {f[:-4]: "test" for f in te.image_files}
    want = {k: v[0] for k, v in tab.items()}
    nval = sum(v[1] for v in tab.values())
    ok = got == want
    print(f"[split] 저자 Dataset 재분할: train {len(tr.image_files):,} · test {len(te.image_files):,} · (val {nval} 은 train 안) — "
          + ("splits/author_split.csv 와 같음" if ok else "!!! 다름"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check() if sys.argv[1:] == ["check"] else 2)
