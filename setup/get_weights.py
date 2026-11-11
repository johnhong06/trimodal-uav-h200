"""Hugging Face 공개 가중치를 고정 리비전으로 HF_HOME 에 받고, 파일마다 크기와 해시를 확인한다 (목록 = setup/weights.json).
LFS 파일은 sha256, 작은 파일은 git blob sha1. 토큰은 쓰지 않는다. 받기가 실패하면 30 초 쉬고 3 번까지.

사용:
  python setup/get_weights.py [--only mit_b1,dinov2_large]   # 받기 + 확인 (학습 환경, huggingface_hub). 이미 받은 파일은 확인만
  python3 setup/get_weights.py --dry_run                     # 받지 않고 HF API 로 리비전 · 크기 · 해시만 대조 (표준 라이브러리)
  python3 setup/get_weights.py --dir mit_b1                  # 받은 스냅숏 폴더 경로를 찍는다 (네트워크 없음, 없으면 종료 1)
종료 코드: 0 통과, 1 실패.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = {k: v for k, v in json.load(open(os.path.join(HERE, "weights.json"), encoding="utf-8")).items() if not k.startswith("_")}


def hub_dir():
    if os.environ.get("HF_HUB_CACHE"):
        return os.path.expanduser(os.environ["HF_HUB_CACHE"])
    return os.path.join(os.path.expanduser(os.environ.get("HF_HOME") or "~/.cache/huggingface"), "hub")


def snap_dir(m):
    return os.path.join(hub_dir(), "models--" + m["repo"].replace("/", "--"), "snapshots", m["revision"])


def problem(path, spec):
    """파일 하나 확인 → 문제 문장 또는 None."""
    if not os.path.isfile(path):
        return "파일 없음"
    n = os.path.getsize(path)
    if n != spec["size"]:
        return f"크기 {n:,} ≠ {spec['size']:,}"
    h = hashlib.sha256() if "sha256" in spec else hashlib.sha1(b"blob %d\0" % n)
    with open(path, "rb") as f:
        while c := f.read(16 << 20):
            h.update(c)
    want = spec.get("sha256") or spec["git_sha1"]
    return None if h.hexdigest() == want else "해시 다름"


def api_problems(m):
    url = f"https://huggingface.co/api/models/{m['repo']}/revision/{m['revision']}?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "trimodal-h200-setup"}), timeout=30) as r:
        d = json.load(r)
    errs = [] if d.get("sha") == m["revision"] else ["리비전 다름"]
    if d.get("gated") or d.get("private"):
        errs.append("gated/private")
    sib = {s["rfilename"]: s for s in d.get("siblings", [])}
    for f, spec in m["files"].items():
        s = sib.get(f)
        got = None if s is None else ((s.get("lfs") or {}).get("sha256") if "sha256" in spec else s.get("blobId"))
        if s is None or s.get("size") != spec["size"] or got != (spec.get("sha256") or spec.get("git_sha1")):
            errs.append(f"{f} 다름")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SPEC))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dir")
    a = ap.parse_args()
    if a.dir:
        d = snap_dir(SPEC[a.dir])
        ok = all(os.path.isfile(os.path.join(d, f)) for f in SPEC[a.dir]["files"])
        print(d if ok else "")
        return 0 if ok else 1
    keys = [k for k in a.only.split(",") if k]
    bad = 0
    for k in keys:
        m = SPEC[k]
        if a.dry_run:
            try:
                e = api_problems(m)
            except Exception as ex:                       # 접속 실패도 문제로 센다
                e = [f"API 접속 실패 ({type(ex).__name__})"]
            print(f"[weights] {k} {m['repo']}@{m['revision'][:8]}: " + ("API 대조 통과" if not e else "!!! " + ", ".join(e)))
            bad += bool(e)
            continue
        from huggingface_hub import hf_hub_download
        for f, spec in m["files"].items():
            p = os.path.join(snap_dir(m), f)
            if problem(p, spec) is None:
                continue
            for t in range(3):
                try:
                    p = hf_hub_download(m["repo"], f, revision=m["revision"])
                    break
                except Exception as ex:
                    print(f"[weights] {k}/{f} 받기 실패 ({t + 1}/3, {type(ex).__name__}) — 30 초 뒤 다시", flush=True)
                    time.sleep(30)
            e = problem(p, spec)
            if e:
                print(f"!!! [weights] {k}/{f}: {e}")
                bad += 1
        if not bad:
            print(f"[weights] {k} {m['repo']}@{m['revision'][:8]}: 파일 {len(m['files'])} 개 확인")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
