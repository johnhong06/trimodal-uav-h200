# NOTICE — 제3자 코드 · 가중치 · 데이터

이 저장소에는 우리 코드와 작은 목록 (분할 · 통계 · 체크섬) 만 들어 있다. 아래 것들은 **저장소에 넣지 않고** 작업이 시작할 때 원래 자리에서 받는다.
라이선스 내용은 2026-11-11 에 각 출처에서 확인한 것이다. 법률 자문이 아니므로 쓰기 전에 원 출처를 다시 확인한다.

| 무엇 | 출처 · 판 | 라이선스 (확인한 내용) | 저장소에 넣나 |
|---|---|---|---|
| 저자 코드 `trimodaldet` | https://github.com/radlab-sketch/trimodal-uav-det @ `e1d2dede5bd3ed5bd8a85b47b813b404a5c667fe` | 저장소 README 에 "MIT License" 라고 적혀 있다. LICENSE 파일은 없다 | 안 넣음 — `setup/fetch_code.sh` 가 받아 파일마다 sha256 대조. 고치지 않고 불러 쓴다 |
| 데이터셋 (`data.zip`) | 저자 저장소 README 의 Google Drive 링크 | 공개된 라이선스 문구 없음 | 안 넣음 — `setup/get_data.py` 가 받음. 분할 목록 · 통계 (`splits/` · `stats/`) 는 파일 이름과 숫자만 담는다 |
| MiT-B1 사전학습 가중치 | Hugging Face `nvidia/mit-b1` @ `13ddceec4e8bdf401e7cd7acf5aebc526222518c` | 모델 카드가 NVlabs/SegFormer 의 LICENSE (NVIDIA Source Code License, 비상업 연구용) 를 가리킨다 | 안 넣음 — `setup/get_weights.py` |
| 파이썬 패키지 | PyPI · PyTorch 휠 색인 (`envs/train.txt`) | 각 패키지 라이선스 | 안 넣음 |

## 우리 코드가 다른 저장소에서 가져온 것

- `run.sh` 의 화면 줄이기 · 단계 실행기 (멈춤 감지) · uv 환경 만들기, `setup/` 의 받기 · 대조 방식은 우리의 다른 H200 실행 묶음 (`depthvlm-edge-h200`, 비공개) 의 같은 부분을 줄여 옮겼다.
- `src/model.py` 의 `map_mit` 은 transformers 의 Segformer 이름을 SegFormer 원본 (저자 코드와 같은) 이름으로 바꾼다. 대응은 두 코드를 보고 직접 적었고,
  `tests/test_parts.py` 가 네 stage 출력이 transformers `SegformerModel` 과 같음을 확인한다.
- `src/rescale.py` (A) 와 `src/edge.py` (B) 는 새로 썼다. 방법은 README '출처' 에 적은 논문들을 참고했고 그 코드는 가져오지 않았다.
  0 에서 여는 게이트로 처음 출력을 기준선과 같게 두는 방식은 우리 프로젝트 depthvlm-silog 에서 쓰던 것이다.
- `src/edge.py` 의 `EdgeRoIHeads` 는 torchvision 0.22.1 (BSD-3-Clause) 의 `RoIHeads` 를 물려받는다. `forward` 의 박스 부분은 원본과 같은 순서로 다시 쓰고 경계 맵 손실과 투표를 더했다.

## 우리 코드의 라이선스

아직 정하지 않았다. 공개 저장소지만 LICENSE 파일이 없으므로 다른 사람에게 코드를 다시 쓰거나 배포할 권리를 주지 않는다. 정하면 LICENSE 를 넣는다. 위 가중치 중 MiT-B1 이 비상업 조건이므로 이 저장소의 실험은 비상업 연구에만 쓴다.
