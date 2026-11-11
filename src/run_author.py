"""저자 학습 스크립트 (scripts/train.py) 를 고치지 않고 그대로 돌린다 — 시드만 먼저 정한다.
우리 학습 루프 (src/train.py --arm paper) 가 저자 루프와 같은 학습을 하는지 보는 대조 (A/A) 용이다.

저자 Trainer: 저자 분할 train 7,800 장 · SGD 0.02 · warmup 500 + cosine · 15 epoch · 배치 16 · DataLoader 작업자 0.
끝나면 --model 에 state_dict 를 저장한다 → src/evaluate.py --arm paper 로 평가 (지표는 저자 utils/metrics.py 와 같은 torchmetrics).

사용: python src/run_author.py --seed 0 --data <images·labels 폴더> --model <저장 경로> [--epochs 15]
"""
import argparse
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import need, seed_all  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--epochs", type=int, default=15)
a = ap.parse_args()
script = os.path.join(need("TRI_CODE"), "scripts", "train.py")
seed_all(a.seed)
sys.argv = [script, "--data", a.data, "--epochs", str(a.epochs), "--batch-size", "16", "--lr", "0.02",
            "--backbone", "mit_b1", "--model", a.model]
runpy.run_path(script, run_name="__main__")
