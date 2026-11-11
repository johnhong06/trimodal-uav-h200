"""데이터 준비: data.zip 을 찾거나 받아서 sha256 을 확인하고, 사진 · 라벨만 <dest>/images · <dest>/labels 에 푼다.

찾는 순서 (setup/data.json):
  1. --zip 으로 준 파일 (run.sh 의 DATA_ZIP=)
  2. --search 폴더들 (H200 = /app/data) 아래 4 단계 안에서 크기가 꼭 맞는 파일 — 관리자가 올려 둔 사본
  3. 없으면 Google Drive 공개 링크에서 받는다 (이어받기, 3 번까지)
어느 쪽이든 sha256 이 data.json 과 같아야 쓴다. 압축 안의 __MACOSX/ (맥 메타데이터) 는 버리고,
이름이 (n)frame_<번호>.npy|.txt 꼴인 파일만 푼다. 끝나면 개수를 확인하고 <dest>/.done 에 sha256 을 남긴다 (다시 부르면 건너뜀).
표준 라이브러리만 쓴다.

사용: python3 setup/get_data.py --dest <폴더> --dl <받을 폴더> [--zip <파일>] [--search /app/data]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "data.json"), encoding="utf-8"))
Z = SPEC["zip"]
NAME = re.compile(r"^data/(images|labels)/(n?frame_\d+)\.(npy|txt)$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while c := f.read(32 << 20):
            h.update(c)
    return h.hexdigest()


def search(roots, depth=4):
    for root in roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip("/").count("/")
        for d, subs, files in os.walk(root):
            if d.count("/") - base >= depth:
                subs[:] = []
            for f in files:
                p = os.path.join(d, f)
                try:
                    if os.path.getsize(p) == Z["size"]:
                        yield p
                except OSError:
                    pass


def download(dst):
    """Google Drive 공개 링크 → dst (이어받기). 받은 크기가 맞으면 True."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + ".part"
    for t in range(3):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(Z["url"], headers={"User-Agent": "trimodal-h200-setup", "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if "text/html" in (r.headers.get("Content-Type") or ""):
                    print("!!! [data] Drive 가 파일 대신 웹 페이지를 줌 (받기 한도 · 확인 페이지)")
                    return False
                mode = "ab" if r.status == 206 else "wb"
                got, t0, mark = (have if mode == "ab" else 0), time.time(), time.time()
                with open(part, mode) as f:
                    while c := r.read(16 << 20):
                        f.write(c)
                        got += len(c)
                        if time.time() - mark > 120:              # 2 분마다 한 줄 (멈춤 감지가 진행으로 보게)
                            print(f"[data] 받는 중 {got / 2**30:.2f} / {Z['size'] / 2**30:.2f} GiB", flush=True)
                            mark = time.time()
        except Exception as ex:
            print(f"[data] 받기 끊김 ({t + 1}/3, {type(ex).__name__}) — 30 초 뒤 이어받기", flush=True)
            time.sleep(30)
            continue
        if os.path.getsize(part) == Z["size"]:
            os.replace(part, dst)
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--dl", required=True)
    ap.add_argument("--zip", default="")
    ap.add_argument("--search", nargs="*", default=[])
    a = ap.parse_args()
    done = os.path.join(a.dest, ".done")
    if os.path.exists(done) and open(done).read().strip() == Z["sha256"]:
        print("[data] 이미 풀려 있음 (같은 작업 안)")
        return 0

    cands = ([a.zip] if a.zip else []) + list(search(a.search))
    src = None
    for c in cands:
        t0 = time.time()
        if os.path.isfile(c) and sha256(c) == Z["sha256"]:
            src = c
            print(f"[data] 올려 둔 사본 사용 (sha256 일치, {time.time() - t0:.0f} 초)")
            break
        print(f"[data] 후보가 맞지 않음 — 건너뜀 ({os.path.basename(c)})")
    if src is None:
        dst = os.path.join(a.dl, Z["name"])
        t0 = time.time()
        print(f"[data] Google Drive 에서 받기 ({Z['size'] / 2**30:.2f} GiB)", flush=True)
        if not download(dst):
            print("!!! [data] 받지 못함. 관리자가 data.zip 을 /app/data 아래에 올리면 그것을 쓴다 (docs/ADMIN_ko.md)")
            return 1
        if sha256(dst) != Z["sha256"]:
            print("!!! [data] 받은 파일의 sha256 이 다름")
            return 1
        print(f"[data] 받기 · 확인 끝 ({(time.time() - t0) / 60:.1f} 분)")
        src = dst

    t0 = time.time()
    shutil.rmtree(a.dest, ignore_errors=True)
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(a.dest, sub))
    n = {"images": 0, "labels": 0}
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            m = NAME.match(info.filename)
            if not m:
                continue
            with z.open(info) as fi, open(os.path.join(a.dest, m.group(1), m.group(2) + "." + m.group(3)), "wb") as fo:
                shutil.copyfileobj(fi, fo, 8 << 20)
            n[m.group(1)] += 1
    want = SPEC["counts"]
    ok = n["images"] == want["images"] and n["labels"] == want["labels"]
    print(f"[data] 풀기 {time.time() - t0:.0f} 초 — 사진 {n['images']:,} · 라벨 {n['labels']:,} " + ("(맞음)" if ok else f"!!! 기대 {want['images']:,} · {want['labels']:,}"))
    if not ok:
        return 1
    with open(done, "w") as f:
        f.write(Z["sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
