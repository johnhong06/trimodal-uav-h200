# H200 관리자 안내 — trimodal-uav-h200

H200 관리자께 드리는 안내입니다. 3-모달 UAV 차량 검출 실험 (`trimodal-uav-h200`) 을 Aerodrone H200 에서 돌리기 위해
관리자 쪽에서 해 주실 일을 적었습니다. 실험 내용과 실행 방법 전체는 [README.md](../README.md) 에 있습니다.

## 1. 저장소

- 공개 저장소라서 따로 권한을 드릴 필요가 없습니다.
- 요청 이슈의 GitHub 링크: `https://github.com/johnhong06/trimodal-uav-h200.git`
- 명령은 모두 한 줄이고 환경 변수 없이 `KEY=값` 인자만 씁니다.

| 명령 | GPU 할당량 | 내용 |
|---|---|---|
| `bash run.sh check` | `1` | 1–2 분. 경로 · 공간 · GPU · 외부 접속 확인. 아무것도 설치하지 않음 |
| `bash run.sh smoke` | `7` | 모든 단계를 아주 작게 (환경 만들기 · 데이터 받기 포함) |
| `bash run.sh all ARMS=…` | `7` | 본 실험. 단계를 나눠 여러 번 요청드립니다 (작업 하나에 학습 3–6 번, 6–11 시간. 끝까지 돌고 결과를 묶음) |

## 2. 올려 주실 것 — 보통은 없습니다

데이터셋 · 저자 코드 · 가중치는 모두 공개 주소에 있어서 작업이 시작할 때 직접 받습니다 (토큰 없음).

| 무엇 | 출처 | 크기 |
|---|---|---|
| 데이터셋 `data.zip` | 논문 저자의 Google Drive 공개 링크 | 3.56 GB |
| 저자 코드 | `codeload.github.com/radlab-sketch/trimodal-uav-det` (고정 커밋) | 1 MB |
| 사전학습 가중치 | Hugging Face `nvidia/mit-b1` (고정 리비전) | 55 MB |
| 파이썬 패키지 | PyPI · PyTorch 휠 색인 | 수 GB |

**컨테이너에서 Google Drive 가 막혀 있을 때만** `data.zip` 을 올려 주시면 됩니다 (`bash run.sh check` 화면의 `접속 Google Drive (data.zip)` 줄로 알 수 있습니다).

- 받는 곳: https://drive.google.com/drive/folders/1U2NJQMRz8uP8e-z9lCguRWVPWpXs4MmB 안의 `data.zip` 하나
- 크기 3,558,279,309 바이트, SHA-256 `6d503f3d706308043ac17acada57c987a06cf97b82ad82586b4d68679ec779ad`
- 압축을 풀지 말고 그대로 `/app/data` 아래 아무 폴더에나 올려 주세요 (예: `/app/data/trimodal_uav/data.zip`). 작업이 4 단계 안에서 크기와 SHA-256 이 맞는 파일을 찾습니다.

```bash
sha256sum data.zip      # 위 값과 같아야 합니다. macOS 는 shasum -a 256 data.zip
```

## 3. 비밀 정보와 공개 출력

- 저장소 · 이슈 어디에도 토큰 · 비밀번호 · 키가 없습니다. 저장소에는 데이터셋 · 가중치도 넣지 않았습니다 (작업 안에서 받음).
- 이슈 댓글로 돌아오는 화면 출력은 공개라서 이름과 숫자만 찍습니다. 전체 기록은 `/app/output/<작업>/logs/` 에 파일로 남습니다.

## 4. 공간 (참고)

- `/app/scratch` (작업마다 새로 쓰고 비워짐): 환경 약 10 GB + 풀린 데이터 6.2 GB + 가중치 0.1 GB + 체크포인트 (학습 하나에 약 0.5–0.8 GB, 작업 하나에 6 번까지 약 5 GB). 합쳐 25 GB 안팎입니다.
- `/app/output`: 결과만 씁니다 — 기록 · 지표 · 예측 + `results.zip` (단계마다 첫 시드 체크포인트 포함, 작업 하나에 0.2–0.5 GB).

## 5. 결과 요청

작업이 끝나면 사용자가 **이슈 번호와 `/app/output/<작업>/results.zip` 경로**를 적어 요청드립니다 (경로는 화면 출력의 첫 줄과 마지막 줄에 있습니다).

## 6. 사용 범위 · 삭제

- `data.zip` 을 올려 주셨다면 이 실험 전용입니다. 실험이 끝났다고 사용자가 알려 드리면 지워 주세요 (데이터셋 라이선스가 공개되어 있지 않습니다).
- `/app/output` 의 결과는 사용자가 받은 뒤 정리를 부탁드립니다.
