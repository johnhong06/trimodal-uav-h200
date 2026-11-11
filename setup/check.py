"""점검 — 아무것도 설치하지 않는다.

  python3 setup/check.py box  --app /app    # 컨테이너: 파이썬 · torch (있으면) · GPU 메모리 · CPU · RAM · /dev/shm · 경로 여유 · 쓰기 가능 ·
                                            #   /app/data 에서 data.zip 크기의 파일 · 외부 접속 (HF · codeload · Google Drive · PyPI · PyTorch 휠)
  <학습 파이썬> setup/check.py env [--gpu]   # 학습 환경: 패키지 판 · CUDA · bf16 · 작은 GPU 연산
화면에는 이름과 숫자만 찍는다 (요청 이슈 댓글은 공개).
"""
import argparse
import concurrent.futures as cf
import json
import os
import platform
import shutil
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data.json"), encoding="utf-8"))
W = {k: v for k, v in json.load(open(os.path.join(HERE, "weights.json"), encoding="utf-8")).items() if not k.startswith("_")}
COMMIT = "e1d2dede5bd3ed5bd8a85b47b813b404a5c667fe"


def gib(n):
    return f"{n / 2**30:.1f} GiB"


def probe(name, url, headers=None, expect=None):
    req = urllib.request.Request(url, headers={"User-Agent": "trimodal-h200-check", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            note = ""
            if expect:
                note = expect(r)
            return f"  접속 {name}: OK ({r.status}){note}"
    except Exception as ex:
        return f"  접속 {name}: 실패 ({type(ex).__name__}: {str(ex)[:80]})"


def drive_note(r):
    cr = r.headers.get("Content-Range") or ""
    ct = r.headers.get("Content-Type") or ""
    if "text/html" in ct:
        return " — 파일이 아니라 웹 페이지 (받기 막힘)"
    return " — 크기 맞음" if cr.endswith("/" + str(DATA["zip"]["size"])) else f" — 크기 확인 안 됨 ({cr or '범위 응답 없음'})"


def box(app):
    print(f"[check] 파이썬 {platform.python_version()} · {platform.platform()}")
    try:
        import torch
        line = f"[check] torch {torch.__version__} · CUDA {torch.version.cuda} · GPU {torch.cuda.is_available()}"
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            line += f" · {torch.cuda.get_device_name(0)} · 메모리 {gib(free)} 비어 있음 / {gib(total)}"
        print(line)
    except Exception as ex:
        print(f"[check] 이미지 파이썬에 torch 없음 ({type(ex).__name__}) — 학습 환경은 setup 이 따로 만든다")
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count()
    quota = ""
    if os.path.exists("/sys/fs/cgroup/cpu.max"):
        q = open("/sys/fs/cgroup/cpu.max").read().split()
        quota = f" · cgroup {q[0]}/{q[1]}" if q and q[0] != "max" else " · cgroup 제한 없음"
    mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split()[0] in ("MemTotal:", "MemAvailable:")}
    shm = shutil.disk_usage("/dev/shm").total if os.path.isdir("/dev/shm") else 0
    print(f"[check] CPU {ncpu}{quota} · RAM {gib(mem.get('MemAvailable', 0) * 1024)} 비어 있음 / {gib(mem.get('MemTotal', 0) * 1024)} · /dev/shm {gib(shm)}")
    for sub in ("data", "output", "scratch"):
        p = os.path.join(app, sub)
        if not os.path.isdir(p):
            print(f"[check] {p}: 없음")
            continue
        u = shutil.disk_usage(p)
        try:
            with tempfile.NamedTemporaryFile(dir=p):
                w = "쓰기 가능"
        except OSError:
            w = "읽기 전용"
        print(f"[check] {p}: 여유 {gib(u.free)} / {gib(u.total)} · {w}")
    found = []
    d0 = os.path.join(app, "data")
    if os.path.isdir(d0):
        base = d0.count("/")
        for d, subs, files in os.walk(d0):
            if d.count("/") - base >= 4:
                subs[:] = []
            found += [f for f in files if os.path.getsize(os.path.join(d, f)) == DATA["zip"]["size"]]
    print(f"[check] /app/data 에서 data.zip 크기의 파일: {len(found)} 개" + (f" ({', '.join(found[:3])})" if found else " — 없으면 작업이 Google Drive 에서 받는다"))
    m = W["mit_b1"]
    jobs = [
        ("Hugging Face API", f"https://huggingface.co/api/models/{m['repo']}/revision/{m['revision']}", None, None),
        ("Hugging Face 파일", f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/config.json", {"Range": "bytes=0-0"}, None),
        ("codeload.github.com", f"https://codeload.github.com/radlab-sketch/trimodal-uav-det/tar.gz/{COMMIT}", {"Range": "bytes=0-0"}, None),
        ("Google Drive (data.zip)", DATA["zip"]["url"], {"Range": "bytes=0-0"}, drive_note),
        ("PyPI", "https://pypi.org/simple/pycocotools/", None, None),
        ("PyTorch 휠 색인", "https://download.pytorch.org/whl/cu128/torch/", None, None),
    ]
    with cf.ThreadPoolExecutor(len(jobs)) as ex:
        for line in ex.map(lambda j: probe(*j), jobs):
            print(line)


def env(gpu):
    import importlib
    vers = []
    for mod in ("torch", "torchvision", "transformers", "huggingface_hub", "safetensors", "torchmetrics", "pycocotools", "sklearn", "numpy"):
        try:
            vers.append(f"{mod} {getattr(importlib.import_module(mod), '__version__', '?')}")
        except Exception as ex:
            print(f"!!! [env] {mod} 불러오기 실패 ({type(ex).__name__})")
            return 1
    print("[env] " + " · ".join(vers))
    import torch
    if gpu:
        if not torch.cuda.is_available():
            print("!!! [env] GPU 를 못 봄")
            return 1
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        free, total = torch.cuda.mem_get_info()
        print(f"[env] {torch.cuda.get_device_name(0)} · bf16 {torch.cuda.is_bf16_supported()} · 메모리 {gib(free)} / {gib(total)} · 연산 확인 {y == y}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["box", "env"])
    ap.add_argument("--app", default="/app")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    sys.exit(box(a.app) or 0 if a.what == "box" else env(a.gpu))
