#!/usr/bin/env bash
# H200 진입점 (gpu-request 이슈의 실행 명령 한 줄) — 논문 "Tri-Modal Fusion Transformers for UAV-based Object Detection"
# (arXiv 2604.16630) 과 같은 데이터 · 같은 분할로 논문 성능 (MiT-B1 3-모달 mAP 84.24) 을 넘을 수 있는지 본다.
#   bash run.sh check                     # 1–2 분: 파이썬 · torch · GPU · CPU · RAM · 경로 여유 · 쓰기 가능 · data.zip 사본 · 외부 접속. 설치 안 함
#   bash run.sh smoke                     # 끝까지 아주 작게: 학습 64 장 · 평가 64 장 · 1 epoch · 시드 0, 모든 단계
#   bash run.sh all [ARMS=…] [SEEDS=…]    # 준비 → 데이터 → 단계마다 학습 · 평가 → 표 → results.zip (단계마다 다시 묶음)
#   bash run.sh setup [DRY_RUN=1] | data  # 한 단계만 (작업끼리 파일이 안 이어지므로 data 는 setup 도 먼저 한다)
# KEY=값 (이슈 한 줄 명령에는 환경 변수를 못 넣으므로 인자로):
#   ARMS=author,paper,select,base,inject  all 에서 돌릴 단계 (쉼표). 단계 뜻은 README '실험'
#   SEEDS=0,1,2   OPT=sgd|adamw (select 를 빼고 base · inject 를 돌릴 때 최적화)   EPOCHS=15   LIMIT=0 (학습 · 평가 장 수 제한)
#   TIME=200 (첫 시드의 test 평가에서 1 장씩 추론 시간을 잴 장 수, 0 = 안 잼)   CKPT=seed0|all|none (results.zip 에 넣을 final.pt)
#   STALL_MIN=40 (로그가 이만큼 그대로면 멈춘 것으로 보고 끔)   PROGRESS_SEC=1800   RETRY=2 (학습이 죽으면 last.pt 에서 다시)
#   JOB=<결과 폴더 이름>   DATA_ZIP=<data.zip 경로>   DRY_RUN=1 (setup: 큰 것을 받지 않음)
#   (로컬 시험) WORK=<작업 폴더>  PY=<쓸 파이썬>  APP_ROOT=<가짜 /app>
# 경로 (H200): /app/data 읽기 전용 (관리자가 올린 data.zip 이 있으면 씀) · /app/scratch/tri = 작업 폴더 (환경 · 캐시 · 코드 · 데이터 · 체크포인트, 작업마다 비워짐)
#   /app/output/<작업>/ = 결과만: logs/ · runs/ · summary/ · results.zip
# 화면 출력 = 짧은 요약 (앞 150 줄 + 끝 50 줄, 65,000 자 안). 이름과 숫자만 찍는다 (이슈 댓글은 공개). 토큰은 쓰지 않는다.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
REPO=$PWD
MODE=${1:-}
[ $# -gt 0 ] && shift
MODES="check setup data smoke all"
[[ " $MODES " == *" $MODE "* ]] && [ -n "$MODE" ] || { echo "!!! 모드: ${MODES// / | } (받은 값: '$MODE'). 사용법은 run.sh 맨 위"; exit 2; }
KEYS="ARMS SEEDS OPT EPOCHS LIMIT TIME CKPT STALL_MIN PROGRESS_SEC RETRY JOB DATA_ZIP DRY_RUN WORK PY APP_ROOT"
for k in $KEYS; do unset "$k"; done            # 컨테이너 환경 변수와 섞이지 않게 — 값은 KEY=값 인자로만
for a in "$@"; do
  k=${a%%=*}
  if [ "$k" = "$a" ] || [[ " $KEYS " != *" $k "* ]]; then echo "!!! 모르는 인자: $a (KEY=값 꼴, KEY = $KEYS)"; exit 2; fi
  printf -v "$k" '%s' "${a#*=}"
done
INT='^[0-9]+$'; NAME='^[A-Za-z0-9_.+-]+$'; LIST='^[a-z0-9,]*$'
chk() { [ -z "${!1:-}" ] || [[ ${!1} =~ $2 ]] || { echo "!!! $1=${!1} — 형식이 맞지 않음"; exit 2; }; }
chk ARMS "$LIST"; chk SEEDS "$LIST"; chk EPOCHS "$INT"; chk LIMIT "$INT"; chk TIME "$INT"; chk STALL_MIN "$INT"; chk PROGRESS_SEC "$INT"; chk RETRY "$INT"; chk JOB "$NAME"
case ${OPT:-} in ""|sgd|adamw) ;; *) echo "!!! OPT = sgd | adamw"; exit 2;; esac
case ${CKPT:-seed0} in seed0|all|none) ;; *) echo "!!! CKPT = seed0 | all | none"; exit 2;; esac
case ${DRY_RUN:-0} in 0) ;; 1) [ "$MODE" = setup ] || { echo "!!! DRY_RUN=1 은 setup 모드에만"; exit 2; };; *) echo "!!! DRY_RUN = 0 | 1"; exit 2;; esac
_a=${ARMS:-}; for x in ${_a//,/ }; do case $x in author|paper|select|base|inject) ;; *) echo "!!! ARMS 의 '$x' — author | paper | select | base | inject"; exit 2;; esac; done
if [ "$MODE" = smoke ]; then
  ARMS=${ARMS:-author,paper,select,base,inject}; SEEDS=${SEEDS:-0}; EPOCHS=${EPOCHS:-1}; LIMIT=${LIMIT:-64}; TIME=${TIME:-5}; CKPT=${CKPT:-none}
fi
ARMS=${ARMS:-author,paper,select,base,inject}; SEEDS=${SEEDS:-0,1,2}; OPT=${OPT:-}; EPOCHS=${EPOCHS:-15}; LIMIT=${LIMIT:-0}; TIME=${TIME:-200}
CKPT=${CKPT:-seed0}; STALL_MIN=${STALL_MIN:-40}; PROGRESS_SEC=${PROGRESS_SEC:-1800}; RETRY=${RETRY:-2}; DRY_RUN=${DRY_RUN:-0}
DATA_ZIP=${DATA_ZIP:-}; PY=${PY:-}
[ -n "$SEEDS" ] || { echo "!!! SEEDS 가 비어 있음"; exit 2; }
SEED0=${SEEDS%%,*}
STALL=$((STALL_MIN * 60))

# ───────────────────────── 경로 ─────────────────────────
APP=${APP_ROOT:-/app}
if [ -d "$APP" ]; then H200=1; else H200=0; fi
if [ -n "${WORK:-}" ]; then OUTROOT=$WORK/output
elif [ $H200 = 1 ]; then WORK=$APP/scratch/tri; OUTROOT=$APP/output
else WORK=$REPO/.work; OUTROOT=$REPO/results; fi
[ $H200 = 1 ] && OUTROOT=$APP/output
mkdir -p "$WORK" || { echo "!!! 작업 폴더를 못 만듦: $WORK"; exit 1; }
WORK=$(cd "$WORK" && pwd)
JOB=${JOB:-${MODE}_$(date +%m%d_%H%M%S)}
OUTJ=$OUTROOT/$JOB; LOGS=$OUTJ/logs; RUNS=$OUTJ/runs; SUMM=$OUTJ/summary
mkdir -p "$LOGS" || { echo "!!! 결과 폴더를 못 만듦: $OUTJ"; exit 1; }
CONSOLE=$LOGS/console.log
# 화면 = 앞 150 줄 (40,000 자 안) + 끝 50 줄 (줄마다 400 자) — 이슈 댓글 65,000 자 한도. 전체는 console.log (tee). (depthvlm-edge-h200 run.sh 와 같은 방식)
exec > >(tee -a "$CONSOLE" | awk -v HL=150 -v HC=40000 -v TL=50 -v W=400 '
  { if (length($0) > W) $0 = substr($0, 1, W) " …"
    if (!cut && n < HL && c + length($0) + 1 <= HC) { print; fflush(); n++; c += length($0) + 1; next }
    cut = 1; k++; buf[k % TL] = $0 }
  END { if (k > TL) print "[… " (k - TL) " 줄 생략 — 전체는 logs/console.log]"
        for (i = (k > TL ? k - TL + 1 : 1); i <= k; i++) print buf[i % TL] }') 2>&1
TEE=$!
CUR=""; STOP=0; T_START=$(date +%s)
finish() {
  local rc=$?
  echo "[run] $(date '+%F %T') 끝 — 모드 $MODE, 종료 코드 $rc, $(( ($(date +%s) - T_START) / 60 )) 분. 결과 $OUTJ"
  exec >&- 2>&-
  wait "$TEE" 2>/dev/null   # H200 의 bash 5.1 은 인자 없는 wait 가 tee 까지 기다려 멈춘다 → PID 를 준다
  exit "$rc"
}
trap finish EXIT
trap 'STOP=1; echo "[run] 멈춤 신호 — 지금 단계를 정리하고 결과를 묶는다"; [ -n "$CUR" ] && kill -TERM "$CUR" 2>/dev/null' TERM INT
SYSPY=$(command -v python3 || echo python3)
SETSID=$(command -v setsid || true)

export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$WORK/hf TORCH_HOME=$WORK/torch PIP_CACHE_DIR=$WORK/pip UV_CACHE_DIR=$WORK/uv/cache UV_PYTHON_INSTALL_DIR=$WORK/uv/python XDG_CACHE_HOME=$WORK/cache
export TRI_CODE=$WORK/code/trimodal-uav-det TRI_DATA=$WORK/data
CK=$WORK/ck; STATE=$WORK/state
[ "$MODE" = check ] || mkdir -p "$CK" "$STATE" "$RUNS" "$SUMM"
under() { local p q; p=$(realpath -m "$1"); q=$(realpath -m "$2"); [ "$p" = "$q" ] || [[ $p == "$q"/* ]]; }
for v in WORK HF_HOME UV_CACHE_DIR TRI_CODE TRI_DATA CK; do
  under "${!v}" "$OUTROOT" && { echo "!!! $v=${!v} 가 결과 전용 경로 ($OUTROOT) 아래 — 캐시 · 데이터 · 체크포인트는 작업 폴더에. 중단"; exit 2; }
done
echo "[run] $(date '+%F %T') 모드 $MODE · $([ $H200 = 1 ] && echo "H200 ($APP)" || echo 로컬) · 커밋 $(git rev-parse --short HEAD 2>/dev/null || echo '?')" \
     "· 작업 폴더 여유 $(df -h "$WORK" 2>/dev/null | awk 'NR==2 {print $4}') · 결과 $OUTJ"
[ "$MODE" = check ] || echo "[run] ARMS=$ARMS · SEEDS=$SEEDS · EPOCHS=$EPOCHS · LIMIT=$LIMIT${OPT:+ · OPT=$OPT}"

# ───────────────────────── 단계 실행기: 멈춤 감지 ─────────────────────────
brief_fail() {  # 실패한 로그에서 마지막 예외 한 줄과 위치 (파일:줄 (함수)) 만 — 코드 내용은 찍지 않는다 (공개 댓글)
  "$SYSPY" -I -B - "$1" <<'PY'
import re, sys
try:
    L = open(sys.argv[1], errors="replace").read().splitlines()
except OSError:
    sys.exit()
tb = [i for i, x in enumerate(L) if x.startswith("Traceback")]
if not tb:
    for x in [x for x in L if x.strip()][-3:]:
        print("  로그 끝: " + x.strip()[:300])
    sys.exit()
body = L[tb[-1] + 1:]
frames = [re.sub(r'^\s*File "(?:.*/)?([^/"]+)", line (\d+), in (.*)$', r"\1:\2 (\3)", x) for x in body if x.lstrip().startswith('File "')]
exc = [x for x in body if x and not x.startswith((" ", "\t"))]
print("  예외: " + (exc[-1][:300] if exc else "?"))
for f in frames[-4:]:
    print("  위치: " + f[:200])
PY
}
run_step() {  # 이름 로그 보이기(show|quiet) 명령… → 종료 코드 (124 = 멈춤 감지로 끔, 130 = 멈춤 신호). WATCH=<파일> 이면 그 크기도 '진행' 으로 본다
  local name=$1 log=$2 show=$3; shift 3
  local tp t=0 idle=0 last=-1 sz rc="" seen n lastshow=0 new
  [ "$STOP" = 1 ] && { echo "[$name] 건너뜀 (멈춤 신호)"; return 130; }
  touch "$log"; seen=$(wc -l < "$log")
  $SETSID "$@" >> "$log" 2>&1 < /dev/null &
  tp=$!; CUR=$tp
  while kill -0 "$tp" 2>/dev/null; do
    sleep 5; t=$((t + 5))
    # shellcheck disable=SC2086
    sz=$(stat -c %s "$log" ${WATCH:-} 2>/dev/null | awk '{ s += $1 } END { print s + 0 }')
    if [ "$sz" = "$last" ]; then idle=$((idle + 5)); else idle=0; last=$sz; fi
    n=$(wc -l < "$log")
    if [ "$n" -gt "$seen" ]; then
      new=$(tail -n "+$((seen + 1))" "$log" | head -n $((n - seen))); seen=$n
      [ "$show" = show ] && printf '%s\n' "$new" | grep -aE '^(\[(train|eval|data|split|select|code|weights|env|판정|표1|report)|!!!)' | cut -c1-400 && lastshow=$t
    fi
    if [ $((t - lastshow)) -ge "$PROGRESS_SEC" ]; then echo "[$name] $(date +%H:%M) 진행 중 $((t / 60)) 분 · 로그 변화 없음 $((idle / 60)) 분"; lastshow=$t; fi
    if [ "$STOP" = 1 ]; then rc=130; break; fi
    if [ "$idle" -ge "$STALL" ]; then
      echo "[$name] !!! 로그가 ${STALL_MIN} 분 동안 그대로 — 멈춘 것으로 보고 끈다"
      kill -TERM -- "-$tp" 2>/dev/null; kill -TERM "$tp" 2>/dev/null; sleep 30
      kill -KILL -- "-$tp" 2>/dev/null; kill -KILL "$tp" 2>/dev/null; rc=124; break
    fi
  done
  if [ "${rc:-}" = 130 ]; then
    for ((n = 0; n < 120; n += 5)); do kill -0 "$tp" 2>/dev/null || break; sleep 5; done
    kill -KILL -- "-$tp" 2>/dev/null; kill -KILL "$tp" 2>/dev/null
  fi
  wait "$tp" 2>/dev/null; n=$?
  CUR=""; rc=${rc:-$n}
  [ "$rc" = 0 ] || { echo "[$name] $(date +%H:%M:%S) 실패 — 종료 $rc ($((t / 60)) 분, 기록 ${log#"$OUTJ"/})"; brief_fail "$log"; }
  return "$rc"
}

# ───────────────────────── 준비: 환경 · 저자 코드 · 가중치 ─────────────────────────
UV_VER=0.12.13; PY_VER=3.12.14; PIP_VER=26.2.1   # depthvlm-edge-h200 과 같은 고정 판 (2026-10 H200 에서 확인)
uv_bin() {  # 고정 판 uv — PATH 의 uv 는 판이 같을 때만, 아니면 작업 폴더에 pip --target 으로 깐다 (이미지 파이썬에는 손대지 않음)
  local u
  for u in "$(command -v uv 2>/dev/null)" "$WORK/uv/pkg/bin/uv"; do
    [ -n "$u" ] && [ -x "$u" ] && [ "$("$u" --version 2>/dev/null | awk '{print $2}')" = "$UV_VER" ] && { echo "$u"; return 0; }
  done
  "$SYSPY" -m pip install -q --upgrade --target "$WORK/uv/pkg" "uv==$UV_VER" >&2 || { echo "!!! [env] uv==$UV_VER 설치 실패" >&2; return 1; }
  [ -x "$WORK/uv/pkg/bin/uv" ] && echo "$WORK/uv/pkg/bin/uv"
}
mkenv_body() {
  local req=$1 E=$2 h=$3 u
  echo "[env] 학습 환경 만들기 ($req)"
  u=$(uv_bin) || return 1
  rm -rf "$E"
  "$u" venv -q -p "$PY_VER" "$E" || { echo "!!! [env] uv venv (Python $PY_VER) 실패"; return 1; }
  if [ "$DRY_RUN" = 1 ]; then
    env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$u" pip install --dry-run -p "$E/bin/python" -r "$req" --index-strategy unsafe-best-match 2>&1 | tail -n 3
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "!!! [env] 판 해석 실패"; return 1; }
    echo "[env] DRY_RUN: 판 해석 통과 — 설치는 건너뜀"; return 0
  fi
  env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$u" pip install -q -p "$E/bin/python" "pip==$PIP_VER" || { echo "!!! [env] pip==$PIP_VER 실패"; return 1; }
  env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$E/bin/python" -m pip install -q --no-input -r "$req" || { echo "!!! [env] 패키지 설치 실패"; return 1; }
  "$E/bin/python" -m pip freeze > "$E/freeze.txt" 2>/dev/null
  grep -E '^(torch|torchvision)==' "$E/freeze.txt" > "$E/constraints.txt"
  echo "[env] $("$E/bin/python" -V 2>&1) · $(tr '\n' ' ' < "$E/constraints.txt")"
  echo "$h" > "$E/.ok"
}
TPY=""
setup_env() {
  local E=$WORK/envs/train h rc
  if [ -n "$PY" ]; then TPY=$PY; echo "[setup] 학습 파이썬 (주어진 것) $TPY"; return 0; fi
  h=$({ cat envs/train.txt; echo "uv $UV_VER python $PY_VER pip $PIP_VER"; } | sha256sum | cut -c1-16)
  if [ ! -f "$E/.ok" ] || [ "$(cat "$E/.ok")" != "$h" ]; then
    mkenv_body envs/train.txt "$E" "$h" >> "$LOGS/setup_env.log" 2>&1; rc=$?
    grep -aE '^\[env\]|^!!!' "$LOGS/setup_env.log" | tail -n 2
    [ $rc = 0 ] || return 1
    [ "$DRY_RUN" = 1 ] && return 0
  fi
  TPY=$E/bin/python
  [ -s "$E/constraints.txt" ] && export PIP_CONSTRAINT=$E/constraints.txt UV_CONSTRAINT=$E/constraints.txt
  mkdir -p "$OUTJ/meta"; cp "$E/freeze.txt" "$OUTJ/meta/pip_freeze.txt" 2>/dev/null
}
setup_all() {
  [ -f "$STATE/.ok_setup_$DRY_RUN" ] && { setup_env >/dev/null; export_weights; return 0; }
  setup_env || { echo "!!! [setup] 환경 실패 — logs/setup_env.log"; return 1; }
  run_step code "$LOGS/setup_code.log" show bash setup/fetch_code.sh "$TRI_CODE" || return 1
  if [ "$DRY_RUN" = 1 ]; then
    run_step weights "$LOGS/setup_weights.log" show "$SYSPY" -I -B setup/get_weights.py --dry_run || return 1
    echo "[setup] DRY_RUN: 패키지 설치 · 가중치 받기는 건너뜀"; touch "$STATE/.ok_setup_1"; return 0
  fi
  run_step weights "$LOGS/setup_weights.log" show "$TPY" -I -B setup/get_weights.py || \
    run_step weights "$LOGS/setup_weights.log" show env HF_HUB_DISABLE_XET=1 "$TPY" -I -B setup/get_weights.py || return 1
  export_weights || return 1
  local g=""; [ $H200 = 1 ] && g=--gpu
  run_step env_check "$LOGS/setup_check.log" show env HF_HUB_OFFLINE=1 "$TPY" -I -B setup/check.py env $g || return 1
  touch "$STATE/.ok_setup_0"
}
export_weights() {
  local m d
  m=$("$SYSPY" -I -B setup/get_weights.py --dir mit_b1) && d=$("$SYSPY" -I -B setup/get_weights.py --dir dinov2_large) || { echo "!!! [setup] 가중치가 캐시에 없음"; return 1; }
  export TRI_MIT=$m/pytorch_model.bin TRI_DINO=$d HF_HUB_OFFLINE=1
}
data_step() {
  local s=()
  [ -d "$APP/data" ] && s=(--search "$APP/data")
  [ -f "$STATE/.ok_data" ] && return 0
  WATCH="$WORK/dl/data.zip.part" run_step data "$LOGS/data.log" show "$SYSPY" -I -B -u setup/get_data.py --dest "$TRI_DATA" --dl "$WORK/dl" \
    ${DATA_ZIP:+--zip "$DATA_ZIP"} "${s[@]}" || return 1
  rm -f "$WORK/dl/data.zip"                                       # 풀었으면 압축본은 지운다 (작업 폴더 공간)
  run_step split "$LOGS/split.log" show "$TPY" -I -B -u src/data.py check || return 1
  touch "$STATE/.ok_data"
}

# ───────────────────────── 학습 · 평가 ─────────────────────────
train_eval() {  # 이름 팔 최적화 분할 시드 평가분할 — 끝난 것은 건너뜀 (같은 작업 안)
  local name=$1 arm=$2 opt=$3 split=$4 seed=$5 ev=$6 rd=$RUNS/$1 ck=$CK/$1 a
  mkdir -p "$rd" "$ck"
  if [ ! -f "$rd/done.json" ]; then
    for ((a = 0; a <= RETRY; a++)); do
      WATCH="$rd/train_log.csv" run_step "학습 $name" "$LOGS/train_$name.log" show "$TPY" -I -B -u src/train.py --arm "$arm" --opt "$opt" \
        --split "$split" --seed "$seed" --epochs "$EPOCHS" --limit "$LIMIT" --out "$rd" --ckpt "$ck" --resume
      [ -f "$rd/done.json" ] && break
      [ "$STOP" = 1 ] && return 130
      [ -f "$rd/fail.json" ] && return 1                       # loss 발산은 다시 해도 같음
      [ $a -lt "$RETRY" ] && echo "[학습 $name] 다시 시작 ($((a + 1))/$RETRY) — last.pt 에서"
    done
    [ -f "$rd/done.json" ] || { echo "!!! [학습 $name] 끝내지 못함"; return 1; }
  fi
  eval_one "$name" "$arm" "$ev" "$seed"
}
eval_one() {  # 이름 팔 평가분할 시드
  local rd=$RUNS/$1 tm=()
  [ -f "$rd/eval_$3/metrics.json" ] && return 0
  [ "$3" = test ] && [ "$4" = "$SEED0" ] && [ "$TIME" -gt 0 ] && tm=(--time "$TIME")
  run_step "평가 $1" "$LOGS/eval_$1_$3.log" show "$TPY" -I -B -u src/evaluate.py --arm "$2" --ckpt "$CK/$1/final.pt" --split "$3" \
    --limit "$LIMIT" --out "$rd/eval_$3" "${tm[@]}"
}
author_stage() {  # 저자 학습 스크립트 그대로 (시드 SEED0) → 우리 평가
  local name=author_sgd_full_s$SEED0 dd=$TRI_DATA f n
  mkdir -p "$RUNS/$name" "$CK/$name"
  if [ "$LIMIT" -gt 0 ]; then                                     # smoke: 라벨이 있는 앞 LIMIT×1.25 장만 담은 작은 폴더 (저자 분할이 80/20 으로 다시 나눔)
    dd=$WORK/data_small; rm -rf "$dd"; mkdir -p "$dd/images" "$dd/labels"; n=0
    for f in $(ls "$TRI_DATA/labels" | sort); do
      [ -s "$TRI_DATA/labels/$f" ] || continue
      ln -s "$TRI_DATA/labels/$f" "$dd/labels/$f"; ln -s "$TRI_DATA/images/${f%.txt}.npy" "$dd/images/${f%.txt}.npy"
      n=$((n + 1)); [ $n -ge $((LIMIT * 5 / 4)) ] && [ $n -ge 5 ] && break   # 저자 분할 80/20 이 되려면 5 장 이상
    done
  fi
  if [ ! -f "$RUNS/$name/done.json" ]; then
    rm -f "$CK/$name/final.pt"
    run_step "학습 $name (저자 스크립트)" "$LOGS/train_$name.log" quiet "$TPY" -I -B -u src/run_author.py --seed "$SEED0" --data "$dd" \
      --model "$CK/$name/final.pt" --epochs "$EPOCHS" || return 1
    [ -f "$CK/$name/final.pt" ] || { echo "!!! [학습 $name] 저자 스크립트가 체크포인트를 남기지 않음"; return 1; }
    echo "{\"by\": \"author scripts/train.py\", \"seed\": $SEED0, \"epochs\": $EPOCHS}" > "$RUNS/$name/done.json"
    echo "[train $name] 끝 (저자 스크립트)"
  fi
  eval_one "$name" paper test "$SEED0"
}
CHOSEN=""
opt_for() {  # base · inject 의 최적화: select 결과 → OPT= → 없으면 멈춤
  if [ -f "$SUMM/choice.json" ]; then CHOSEN=$("$SYSPY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["opt"])' "$SUMM/choice.json")
  elif [ -n "$OPT" ]; then CHOSEN=$OPT
  else echo "!!! base · inject 의 최적화를 정할 수 없음 — ARMS 에 select 를 넣거나 OPT=sgd|adamw 를 준다"; return 1; fi
  echo "[run] base · inject 최적화 = $CHOSEN"
}
zip_results() {  # /app/output/<작업>/results.zip — logs · runs (기록 · 지표 · 예측) · summary · meta + CKPT 에 따른 final.pt
  local cks=()
  if [ "$CKPT" != none ] && [ -d "$CK" ]; then
    for d in "$CK"/*/; do
      [ -f "$d/final.pt" ] || continue
      [ "$CKPT" = all ] || [[ $(basename "$d") == *_s$SEED0 ]] && cks+=("$d/final.pt")
    done
  fi
  "$SYSPY" -I -B - "$OUTJ/results.zip" "$OUTJ" "${cks[@]}" <<'PY'
import os, sys, zipfile
dst, root, cks = sys.argv[1], sys.argv[2], sys.argv[3:]
items = []
for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f)
        if not p.startswith(dst):
            items.append((p, os.path.relpath(p, root)))
items += [(c, os.path.join("ckpt", os.path.basename(os.path.dirname(c)) + ".pt")) for c in cks]
with zipfile.ZipFile(dst + ".tmp", "w", allowZip64=True) as z:
    for src, arc in items:
        z.write(src, arc, compress_type=zipfile.ZIP_STORED if arc.endswith(".pt") else zipfile.ZIP_DEFLATED)
os.replace(dst + ".tmp", dst)
print(f"[zip] results.zip — 파일 {len(items)} 개 (체크포인트 {len(cks)}), {os.path.getsize(dst) / 2**20:.1f} MiB")
PY
}
report() { run_step report "$LOGS/report.log" show "$TPY" -I -B -u src/report.py summary --runs "$RUNS" --out "$SUMM"; }
has() { [[ ",$ARMS," == *",$1,"* ]]; }
fail() { echo "!!! [$MODE] $1"; report; zip_results; exit 1; }

case "$MODE" in
  check) "$SYSPY" -I -B setup/check.py box --app "$APP";;
  setup) setup_all || fail "준비 실패";;
  data) setup_all || fail "준비 실패"; data_step || fail "데이터 실패";;
  smoke|all)
    setup_all || fail "준비 실패"
    data_step || fail "데이터 실패"
    if has author; then author_stage || fail "저자 스크립트 단계 실패"; zip_results; fi
    if has paper; then
      for s in ${SEEDS//,/ }; do train_eval "paper_sgd_full_s$s" paper sgd full "$s" test || fail "paper 시드 $s 실패"; done
      report; zip_results
    fi
    if has select; then
      train_eval "base_sgd_sub_s$SEED0" base sgd sub "$SEED0" val || fail "고르기 (sgd) 실패"
      train_eval "base_adamw_sub_s$SEED0" base adamw sub "$SEED0" val || fail "고르기 (adamw) 실패"
      run_step select "$LOGS/select.log" show "$TPY" -I -B -u src/report.py select --runs "$RUNS" --out "$SUMM/choice.json" || fail "고르기 실패"
      zip_results
    fi
    if has base || has inject; then opt_for || fail "최적화 미정"; fi
    if has base; then
      for s in ${SEEDS//,/ }; do train_eval "base_${CHOSEN}_full_s$s" base "$CHOSEN" full "$s" test || fail "base 시드 $s 실패"; done
      report; zip_results
    fi
    if has inject; then
      for s in ${SEEDS//,/ }; do train_eval "inject_${CHOSEN}_full_s$s" inject "$CHOSEN" full "$s" test || fail "inject 시드 $s 실패"; done
    fi
    report; zip_results;;
esac
