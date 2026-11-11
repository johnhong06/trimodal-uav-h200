#!/usr/bin/env bash
# 로컬 CPU 시험: 준비 (저자 코드 · 가중치) → 데이터 (로컬 data.zip) → 부품 시험 → 아주 작은 smoke.
# GPU 는 쓰지 않는다 (CUDA_VISIBLE_DEVICES 비움). CPU 는 nice 19 · 스레드 6 개. 작업 폴더에 약 8 GB 를 쓴다 (풀린 데이터 6.2 GB + DINOv2-L 1.2 GB).
#
# ★ 모든 단계를 메모리 한도를 건 systemd 범위 (MemoryMax) 안에서 돌린다. 2026-11-11 사고: VSCode 터미널에서 바로 띄운 CPU smoke 의
#   inject 단계 (DINOv2-L 을 CPU 로, 사진 4 장) 가 20 GB 까지 올라 전역 OOM 이 났고, 같은 범위 (VSCode) 의 편집기 프로세스가 함께 죽었다.
#   범위를 따로 두면 한도를 넘어도 이 시험만 끝난다. inject 는 사진 1 장으로 따로 돈다 (12 GB 안).
# 사용: bash tests/run_local.sh <작업 폴더> <파이썬 (envs/train.txt 의 판, CPU torch 면 됨)> <data.zip>
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:?작업 폴더}; P=${2:?파이썬}; Z=${3:?data.zip}
box() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
          env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 nice -n 19 "$@"; }
box bash run.sh setup WORK="$W" PY="$P" JOB=t_setup | grep -aE '^\[(code|weights|env)|^!!!'
box bash run.sh data WORK="$W" PY="$P" JOB=t_data DATA_ZIP="$Z" | grep -aE '^\[(data|split)|^!!!'
export HF_HOME=$W/hf TRI_CODE=$W/code/trimodal-uav-det TRI_DATA=$W/data
TRI_MIT=$(python3 -I setup/get_weights.py --dir mit_b1)/pytorch_model.bin; export TRI_MIT
box "$P" -I -B tests/test_parts.py 2>&1 | grep -aE '^(PASS|FAIL|부품)'
box bash run.sh smoke WORK="$W" PY="$P" JOB=t_smoke ARMS=author,paper,select,base LIMIT=4 DATA_ZIP="$Z" | grep -aE '^\[(train|eval|select|판정|report|zip)|^!!!'
box bash run.sh smoke WORK="$W" PY="$P" JOB=t_smoke_inject ARMS=inject OPT=sgd LIMIT=1 DATA_ZIP="$Z" | grep -aE '^\[(train|eval|판정|report|zip)|^!!!'
