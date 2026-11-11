#!/usr/bin/env bash
# 저자 코드 받기 (H200·로컬 공용, git 없이): radlab-sketch/trimodal-uav-det 를 고정 커밋의 GitHub 압축본
# (codeload.github.com/<저장소>/tar.gz/<커밋>) 으로 받아 <dest> 에 풀고, 파일마다 setup/code.sha256 과 대조한다.
# 하나라도 다르거나 빠지면 실패. 이미 맞게 받았으면 건너뛴다 (.commit). 네트워크가 끊기면 30 초 쉬고 3 번까지.
# 저자 코드는 고치지 않는다 — 우리 코드 (src/) 가 이것을 불러 쓴다.
# 사용: bash setup/fetch_code.sh <dest>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:?"받을 폴더를 준다"}
REPO_NAME=radlab-sketch/trimodal-uav-det
COMMIT=e1d2dede5bd3ed5bd8a85b47b813b404a5c667fe
export PYTHONDONTWRITEBYTECODE=1

verify() {
  python3 -I -B - "$DEST" "$HERE/code.sha256" <<'PY'
import hashlib, os, sys
root, lst = sys.argv[1:]
want = [l.split(None, 1) for l in open(lst, encoding="utf-8") if l.strip() and not l.startswith("#")]
bad = []
for h, p in want:
    p = p.strip()
    f = os.path.join(root, p)
    if not os.path.isfile(f):
        bad.append("없음 " + p)
        continue
    with open(f, "rb") as fh:
        if hashlib.sha256(fh.read()).hexdigest() != h:
            bad.append("다름 " + p)
if bad:
    print(f"!!! [code] 대조 실패 {len(bad)} / {len(want)}: " + ", ".join(bad[:5]))
    sys.exit(1)
print(f"[code] 대조 통과: 파일 {len(want)} 개")
PY
}

if [ "$(cat "$DEST/.commit" 2>/dev/null)" = "$COMMIT" ] && verify; then exit 0; fi
for a in 1 2 3; do
  rm -rf "${DEST:?}"; mkdir -p "$DEST"
  python3 -I -B - "$DEST" "https://codeload.github.com/$REPO_NAME/tar.gz/$COMMIT" <<'PY' && break
import sys, tarfile, urllib.request
dst, url = sys.argv[1:]
with urllib.request.urlopen(url, timeout=300) as r, tarfile.open(fileobj=r, mode="r|gz") as t:
    for m in t:                          # 맨 위 폴더 (<저장소>-<커밋>/) 를 떼고 푼다. 파일 · 폴더만
        parts = m.name.split("/", 1)
        if len(parts) < 2 or not parts[1] or not (m.isfile() or m.isdir()):
            continue
        m.name = parts[1]
        t.extract(m, dst, filter="data") if hasattr(tarfile, "data_filter") else t.extract(m, dst)
PY
  [ "$a" = 3 ] && { echo "!!! [code] 받기 실패 3 번 ($REPO_NAME @ ${COMMIT:0:8})" >&2; exit 1; }
  echo "[code] 받기 실패 ($a/3) — 30 초 뒤 다시" >&2; sleep 30
done
verify
echo "$COMMIT" > "$DEST/.commit"
echo "[code] $REPO_NAME @ ${COMMIT:0:8} → $DEST"
