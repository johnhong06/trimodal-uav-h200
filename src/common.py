"""공통: 경로 · 저자 코드 불러오기 · 시드.

run.sh 가 정하는 환경 변수:
  TRI_CODE  저자 저장소 (setup/fetch_code.sh 가 받은 radlab-sketch/trimodal-uav-det @ e1d2dede)
  TRI_DATA  images/ · labels/ 가 있는 폴더 (setup/get_data.py 가 푼 곳)
  TRI_MIT   nvidia/mit-b1 의 pytorch_model.bin (paper 를 뺀 모든 팔)
"""
import json
import os
import random
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 391, 301                       # 모든 사진이 301×391 (2026-11-11 전수 확인)
PAPER = {"map": 84.24, "map_50": 98.95, "params_m": 60.01}   # 논문 표 1 · 표 5 의 MiT-B1 3-모달 MAGE+BiTE


def need(k):
    v = os.environ.get(k)
    if not v:
        raise SystemExit(f"!!! 환경 변수 {k} 가 없음 (run.sh 가 정한다)")
    return v


def author():
    """저자 패키지 trimodaldet 를 불러온다 (고치지 않은 원본)."""
    code = need("TRI_CODE")
    if code not in sys.path:
        sys.path.insert(0, code)
    import trimodaldet
    return trimodaldet


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def te_stats():
    with open(os.path.join(REPO, "stats", "te_stats.json"), encoding="utf-8") as f:
        return json.load(f)


def is_day(stem):
    return stem.startswith("frame_")


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
