# trimodal-uav-h200

RGB · 열화상 · 이벤트 카메라 세 가지로 UAV 영상에서 차량을 찾는 논문, *Tri-Modal Fusion Transformers for UAV-based Object Detection* (arXiv 2604.16630) 을
**논문과 같은 데이터 · 같은 분할**로 다시 학습해서 논문 수치 (MiT-B1, mAP 84.24) 를 넘을 수 있는지 보는 실험 묶음이다.
Aerodrone H200 에서 돌리려고 만든 공개 저장소다.

이 저장소에는 우리 코드와 작은 목록 (분할 · 통계 · 체크섬) 만 있다. 저자 코드 · 데이터셋 · 사전학습 가중치는 작업이 시작할 때 원래 자리에서 받는다.

## 왜 하나

논문은 열화상 · 이벤트 두 채널을 한 줄기로 묶고 RGB 줄기와 백본 네 stage 에서 섞는다 (MAGE + BiTE). 같은 데이터에서 융합 방식 · 위치 · 백본 크기를 61 가지로 비교했다.
저자 코드 (radlab-sketch/trimodal-uav-det @ e1d2dede) 를 읽어 보면 기준선이 기본 설정부터 약하다.

- 백본을 사전학습 없이 처음부터 학습한다. 이 코드의 융합 모듈 원본인 CMX 는 ImageNet 사전학습 가중치를 쓴다.
- 입력을 255 로 나누지 않는다. 0–255 값을 0–1 용 평균 · 표준편차로 정규화해서 RGB 는 최대 약 1,100 까지 들어간다.
- Faster R-CNN 기본값 때문에 사진 (301×391) 을 약 2.66 배 키워서 학습 · 평가한다. 논문에는 원본 해상도라고 적혀 있다.
- 데이터 증강이 없다.

또 mAP50 은 98.95 인데 mAP (IoU 0.50–0.95) 는 84.24 다. 차는 거의 다 찾고 점수는 엄격한 IoU 에서 박스 위치가 조금씩 어긋나는 데서 잃는다는 뜻으로 읽힌다 (실험 1 에서 확인한다).
그래서 기본 설정을 먼저 바로잡아 기준선을 세우고 그 위에서 우리 방법 (고정 인코더 특징의 게이트 주입, 이전 깊이 추정 프로젝트에서 효과를 본 방식) 을 붙여 본다.

## 공통 설계

- **분할은 저자와 같다.** 저자 코드 그대로 라벨이 있는 9,750 장을 `train_test_split(test_size=0.2, random_state=42)` 로 나눈다 → train 7,800 · test 1,950.
  작업 안에서 저자 Dataset 으로 다시 나눠 `splits/author_split.csv` 와 같은지 확인한다. **최종 숫자는 test 1,950 장에서만 낸다.**
- **설정은 검증 세트로만 고른다.** 저자 코드는 검증 세트 없이 test 로 평가한다. 우리는 train 안에서 낮 · 밤마다 10 % (780 장) 를 떼어 검증 세트로 쓴다. test 로는 아무것도 고르지 않는다.
- **학습 일정은 저자와 같다.** 15 epoch · 배치 16 · 500 스텝 warmup 뒤 cosine · AMP 없음 · 마지막 epoch 의 모델을 쓴다 (고르지 않음).
  저자 코드 (MiT · MAGE · BiTE · FPN · Faster R-CNN 설정) 는 고치지 않고 불러 쓴다. 바꾸는 것은 실험마다 아래 표에 적은 것뿐이다.
- **지표도 저자와 같다.** torchmetrics `MeanAveragePrecision` (COCO 방식). 전체 · 낮 · 밤을 따로 잰다. 숫자는 % 로 적는다.
- **시드 3 개 (0 · 1 · 2).** 논문 안의 차이가 1 점 안팎이라 한 번 돌린 숫자로는 비교할 수 없다.
- 데이터: 사진 10,489 장 중 라벨이 있는 9,750 장을 쓴다 (낮 3,339 · 밤 6,411). 차량 박스 30,634 개, 한 클래스.

| 팔 | 백본 초기화 | 입력 정규화 | 최적화 | 더한 것 |
|---|---|---|---|---|
| `paper` | 처음부터 (저자) | 저자 값 (0–255 에 0–1 용 값) | SGD lr 0.02 (저자) | — |
| `base` | ImageNet 사전학습 MiT-B1 (두 줄기 모두) | /255 + RGB 는 ImageNet 값, 열화상 · 이벤트는 train 통계 | 실험 2 에서 고름 | — |
| `inject` | `base` 와 같음 | `base` 와 같음 | `base` 와 같음 | 고정 DINOv2-L 특징을 게이트로 FPN 에 더함 |

## 실험 0 — 논문 재현

### 설계

- `author`: 저자 학습 스크립트 (`scripts/train.py`) 를 고치지 않고 그대로, 시드 0 으로 한 번.
- `paper`: 우리 학습 루프로 같은 설정, 시드 3 개. 이후 실험은 모두 이 루프를 쓴다.
- 둘을 비교해서 우리 루프가 저자 루프와 같은 학습을 하는지 본다.

### 판정 기준 (2026-11-11, 결과 전에 고정)

- **재현됨**: `paper` 시드 3 개 평균 mAP 가 84.24 ± 1.5 안.
- **같은 학습**: `author` mAP 가 `paper` 시드 3 개의 [최솟값 − 0.5, 최댓값 + 0.5] 안.
- 재현이 안 되면 이후 비교의 기준은 논문 숫자가 아니라 우리 `paper` 재현값으로 둔다 (논문 숫자도 같이 적는다).

### 결과

표 1 의 앞 세 줄.

### 진행 상황

- [ ] H200 `smoke`
- [ ] H200 `all`

## 실험 1 — 어디서 점수를 잃나 (오차 분해)

### 설계

test 정답 박스마다, 점수 0.5 이상 예측 가운데 IoU 0.5 · 0.75 · 0.9 에서 짝을 찾았는지와 가장 잘 맞는 예측의 IoU 를 본다.
낮 · 밤, 크기 (COCO small < 32² · medium), 화면 가장자리에 걸린 박스 (전체의 약 30 %) 와 안쪽 박스로 나눈다.
학습은 따로 하지 않고 다른 실험의 예측 (`preds.npz`) 을 쓴다.

### 판정 기준

분해는 설명용이라 합격선이 없다. 실험 2 · 3 에서 무엇이 바뀌었는지 읽는 데 쓴다.

### 결과

표 3.

## 실험 2 — 기본 설정 보강

### 설계

- 사전학습 가중치와 입력 정규화를 바로잡는다 (팔 `base`). 사전학습 백본에는 저자의 SGD 0.02 가 너무 클 수 있어서 최적화를 하나 고른다.
- 고르기: train 에서 검증 780 장을 뺀 7,020 장으로 SGD (lr 0.02 · momentum 0.9 · wd 1e-4) 와 AdamW (lr 1e-4 · wd 0.05) 를 시드 0 으로 한 번씩 학습하고 검증 mAP 를 본다.
- 고른 최적화로 train 7,800 장 전체에서 시드 3 개를 학습해 test 에서 잰다.

### 판정 기준 (2026-11-11, 결과 전에 고정)

- **고르기**: AdamW 의 검증 mAP 가 SGD 보다 0.3 이상 높으면 AdamW, 아니면 SGD (저자 최적화).
- **개선 판정** (실험 2 · 3 공통): Δ = 두 팔의 시드 평균 차, 폭 = 두 팔 각각의 시드 범위 (최댓값 − 최솟값) 중 큰 것.
  Δ > 폭 이면 **개선**, 0 < Δ ≤ 폭 이면 **약한 개선**, Δ ≤ 0 이면 **개선 없음**.
- `base` 는 `paper` 재현과 비교한다. 이것은 우리 방법의 성과가 아니라 기준선을 바로 세운 효과로 따로 적는다.

### 결과

표 1 의 `base` 줄, 표 2.

### 진행 상황

- [ ] H200 `all`

## 실험 3 — 우리 방법: 고정 인코더 특징의 게이트 주입

### 설계

- 사전학습된 큰 인코더 DINOv2-L (고정, 학습 안 함) 의 특징을 검출기 FPN 네 층에 더한다.
  `P_l ← P_l + g_l · Conv1x1([LN(RGB 특징) ; LN(열화상 특징)])`, 게이트 g_l 은 0 에서 시작한다.
  그래서 학습 첫 순간의 출력은 `base` 와 똑같고, 게이트는 학습하면서 도움이 되는 만큼만 열린다.
- DINOv2 층은 깊이의 1/4 · 2/4 · 3/4 · 끝 (6 · 12 · 18 · 24) 을 FPN stride 4 · 8 · 16 · 32 에 하나씩 맞춘다.
- RGB 와 열화상을 각각 넣는다. 밤에는 RGB 가 거의 검기 때문이다 (밝기 중앙값 18.9/255). 이벤트는 넣지 않는다 (밤 프레임의 99 % 가 거의 비어 있음).
- 주입 모듈만 따로 AdamW (wd 0): 투영 lr 1e-4, 게이트 1e-3. 나머지는 `base` 와 같다 (실험 2 에서 고른 최적화).
- 이 꼴은 depthvlm-silog 의 해결 실험 (DepthVLM 고정 + 깊이 인코더 특징을 게이트로 주입) 과 같다. 거기서는 SILog 가 10.91 → 7.39 로 내려갔다.

### 판정 기준 (2026-11-11, 결과 전에 고정)

- `inject` vs `base`: 우리 방법 자체의 효과 (위 개선 판정 규칙).
- `inject` vs `paper`: 기준선 보강까지 합친 효과.
- **논문을 넘음**: `inject` 시드 평균 mAP > 84.24 이고 `paper` 대비 판정이 개선. `inject` 가 못 넘으면 `base` 에 같은 규칙을 적용한다.
- 파라미터 수와 1 장 추론 시간도 같이 적는다. 고정 인코더는 학습 파라미터에 넣지 않고 따로 적는다.

### 결과

표 1 의 `inject` 줄.

### 진행 상황

- [ ] H200 `all`

## 결과 표

결과가 나오면 `results.zip` 의 `summary/tables.md` 로 채운다. 빈칸은 아직 돌리지 않은 것이다.

**표 1. test 1,950 장 (%, 시드 평균 ± 표준편차)**

| 설정 | 시드 | mAP | mAP50 | mAP75 | 낮 mAP | 밤 mAP | small mAP | medium mAP | 파라미터 (M) | 추론 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 논문 보고값 | 1 | 84.24 | 98.95 | — | — | — | — | — | 60.01 | — |
| 저자 스크립트 그대로 (`author`) | 1 | | | | | | | | | |
| 논문 설정 재현 (`paper`) | 3 | | | | | | | | | |
| 기본 설정 보강 (`base`) | 3 | | | | | | | | | |
| 우리 방법 (`inject`) | 3 | | | | | | | | | |

**표 2. 최적화 고르기 — 검증 780 장, 시드 0 (%)**

| 설정 | mAP | mAP50 |
|---|---|---|
| `base` + SGD | | |
| `base` + AdamW | | |

고른 최적화:

**표 3. 오차 분해 — test 정답 박스 단위, 점수 0.5 이상 예측, 시드 평균**

| 설정 | 부분 | 정답 수 | 찾음 @0.5 (%) | @0.75 | @0.9 | IoU 중앙값 |
|---|---|---|---|---|---|---|
| `paper` | 전체 | | | | | |
| `paper` | 밤 | | | | | |
| `paper` | 화면 가장자리에 걸림 | | | | | |
| `base` | 전체 | | | | | |
| `inject` | 전체 | | | | | |

(`summary/tables.md` 에는 낮 · small · medium · 안쪽 박스 줄과 이미지당 잘못 찾은 수도 있다.)

**판정**

| 비교 | 결과 |
|---|---|
| 재현 (`paper` vs 84.24 ± 1.5) | |
| 같은 학습 (`author` vs `paper`) | |
| `base` vs `paper` | |
| `inject` vs `base` | |
| `inject` vs `paper` | |
| 논문을 넘음 | |

## H200 실행

요청 이슈에 명령 한 줄을 적는다 (`bash run.sh <모드> KEY=값 …`). GPU 할당량은 `7` (H200 한 장).

| 순서 | 이슈 명령 | 할당량 | 하는 일 |
|---|---|---|---|
| 1 | `bash run.sh check` | `1` | 1–2 분. 경로 · 공간 · GPU · 외부 접속 확인 (Google Drive 포함). 아무것도 설치하지 않음 |
| 2 | `bash run.sh smoke` | `7` | 모든 단계를 아주 작게 (학습 64 장 · 평가 64 장 · 1 epoch). 환경 만들기 · 데이터 받기 포함 |
| 3 | `bash run.sh all` | `7` | 실험 0 · 2 · 3 전부 (학습 12 번) 와 표 · 판정 |

- 걸리는 시간은 아직 모른다. `smoke` 결과의 스텝 시간으로 `all` 을 어림한 뒤 여기에 적는다.
- 한 작업이 너무 길면 단계를 나눌 수 있다. 예: `bash run.sh all ARMS=author,paper` → `bash run.sh all ARMS=select,base,inject`.
  작업끼리는 파일이 이어지지 않으므로 앞 작업에서 고른 최적화는 `OPT=sgd|adamw` 로 넘긴다.
- 결과는 `/app/output/<작업>/results.zip` 하나다 (경로는 화면 첫 줄과 마지막 줄에 나온다). 이슈 번호와 이 경로를 적어 관리자에게 요청한다.

### 인자

| 키 | 기본값 | 뜻 |
|---|---|---|
| `ARMS` | `author,paper,select,base,inject` | 돌릴 단계 |
| `SEEDS` | `0,1,2` | 시드 (`smoke` 는 `0`) |
| `OPT` | — | `select` 를 빼고 `base` · `inject` 를 돌릴 때 쓸 최적화 |
| `EPOCHS` | 15 | `smoke` 는 1 |
| `LIMIT` | 0 | 학습 · 평가 장 수 제한 (`smoke` 는 64) |
| `TIME` | 200 | 첫 시드의 test 평가에서 1 장씩 추론 시간을 잴 장 수 |
| `CKPT` | `seed0` | `results.zip` 에 넣을 체크포인트 (`seed0` · `all` · `none`) |
| `RETRY` | 2 | 학습이 죽으면 `last.pt` 에서 다시 시작하는 횟수 |
| `STALL_MIN` | 40 | 기록이 이만큼 그대로면 멈춘 것으로 보고 끈다 |
| `DATA_ZIP` | — | `data.zip` 경로를 직접 줄 때 |
| `JOB` | `<모드>_<월일_시분초>` | 결과 폴더 이름 |

이슈 댓글로 오는 화면 출력은 공개라서 이름과 숫자만 찍는다. 전체 기록은 `logs/` 에 있다.

### 결과 묶음 (`results.zip`)

| 경로 | 내용 |
|---|---|
| `logs/` | 화면 전체 (`console.log`) · 단계마다 기록 |
| `runs/<이름>/` | `train_log.csv` · `done.json` · `eval_test/` (또는 `eval_val/`) 의 `metrics.json` · `preds.npz` |
| `summary/` | `tables.md` (위 표를 채울 값) · `summary.json` · `runs.csv` · `choice.json` |
| `ckpt/` | 첫 시드의 `final.pt` (팔마다 약 0.24 GB) |
| `meta/` | `pip_freeze.txt` |

실행 이름은 `<팔>_<최적화>_<분할>_s<시드>` 꼴이다 (예: `paper_sgd_full_s1`, `base_adamw_sub_s0`).

## 데이터와 가중치

모두 작업 안에서 받은 뒤 크기 · sha256 을 확인한다. 토큰이 필요한 것은 없다.

| 무엇 | 어디서 | 크기 |
|---|---|---|
| 데이터셋 `data.zip` | 저자 저장소 README 의 Google Drive 링크 (`setup/data.json`) | 3.56 GB (풀면 6.2 GB) |
| 저자 코드 | `codeload.github.com` 의 고정 커밋 압축본, 파일 35 개 sha256 대조 (`setup/code.sha256`) | 1 MB |
| MiT-B1 사전학습 | Hugging Face `nvidia/mit-b1` 고정 리비전 | 55 MB |
| DINOv2-L | Hugging Face `facebook/dinov2-large` 고정 리비전 | 1.2 GB |
| 파이썬 패키지 | PyPI · PyTorch 휠 색인 (`envs/train.txt`, 판 고정) | 수 GB |

컨테이너에서 Google Drive 가 막혀 있으면 관리자에게 `data.zip` 을 `/app/data` 아래에 올려 달라고 한다. 작업은 `/app/data` 에서 크기와 sha256 이 맞는 파일을 먼저 찾는다 ([docs/ADMIN_ko.md](docs/ADMIN_ko.md)).

저장소에 미리 넣어 둔 것:

- `splits/author_split.csv` — 저자 분할 (train · test) 과 검증 표시. `splits/split.json` 에 목록 해시
- `stats/te_stats.json` — 열화상 · 이벤트 채널의 train 평균 · 표준편차 (0–1 범위)
- `setup/data.json` · `setup/weights.json` · `setup/code.sha256` — 받을 것의 크기 · 해시

## 다음 후보 (이번 묶음에는 없음)

- 신뢰도 가중 증류: `inject` 같은 큰 모델을 교사로, 논문과 같은 크기의 학생을 학습 (크기 · 지연은 그대로, 정확도만).
- 밤 RGB 톤매핑: 어두운 RGB 를 밝게 보정해서 넣기.
- 둘 다 실험 1 · 3 의 결과를 보고 정한다.

## 구성

```
run.sh        진입점 (check · setup · data · smoke · all)
envs/         학습 환경 판 목록 (train.txt)
setup/        check.py · fetch_code.sh + code.sha256 · get_weights.py + weights.json · get_data.py + data.json
src/          common.py · data.py · model.py · inject.py · train.py · evaluate.py · run_author.py · report.py
splits/ stats/  저장소에 넣어 보내는 작은 목록
tools/        로컬 도구 (make_splits.py · make_stats.py — data.zip 에서 위 목록을 만듦)
tests/        CPU 시험 (test_parts.py · run_local.sh)
docs/         ADMIN_ko.md (관리자 안내)
```

## 로컬 시험 (CPU)

```bash
bash -n run.sh && bash run.sh check
bash tests/run_local.sh <작업 폴더> <파이썬> <data.zip>   # 준비 → 데이터 → 부품 시험 → 아주 작은 smoke (LIMIT=4, inject 는 1 장)
```

`tests/run_local.sh` 는 단계마다 메모리 한도 12 GB 의 systemd 범위 (`systemd-run --user --scope -p MemoryMax=12G`) 안에서 돈다.
2026-11-11 에 VSCode 터미널에서 바로 띄운 CPU smoke 가 inject 단계 (DINOv2-L 을 CPU 로) 에서 20 GB 까지 올라 전역 메모리 부족이 났고, 편집기까지 같이 꺼졌다.
몇 분 넘게 도는 로컬 시험은 이 스크립트처럼 한도를 건 범위 안에서 돌린다.

## 출처

- 논문: C. Iaboni, P. Abichandani, *Tri-Modal Fusion Transformers for UAV-based Object Detection*, arXiv:2604.16630, 2026.
- 저자 코드 · 데이터셋: https://github.com/radlab-sketch/trimodal-uav-det (README 에 MIT 라이선스라고 적혀 있음, 데이터셋 라이선스는 적혀 있지 않음).
- 제3자 코드 · 가중치 · 데이터의 라이선스와 재배포 여부: [NOTICE.md](NOTICE.md).
