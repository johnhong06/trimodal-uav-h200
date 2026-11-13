"""모달마다 낮 · 밤에 정보가 얼마나 들어 있나 — 입력만 본다 (학습 없음, numpy 만 있으면 됨). 저자 분할 test 1,950 장.

tools/probe_fuse.py 에서 모델이 낮에는 열화상을, 밤에는 RGB · 이벤트를 거의 쓰지 않는 것으로 나왔다.
그것이 '쓸 정보가 없어서' 인지 '있는데 안 쓰는 것' 인지 가른다.

  1. 밝기: 사진마다 평균 · 표준편차 (0–255)
  2. 이벤트: 바탕값 (그 사진의 최빈값) 에서 5 넘게 벗어난 픽셀의 비율 — 화면 전체 · 정답 박스 안
  3. 박스 변 위의 경계 세기: 변에 수직인 방향의 밝기 차 (가운데 차분) 를 변 ±2 px 안에서 최대로 잡아 변을 따라 평균.
     같은 값을 화면 전체에서 평균한 것 (아무 데나 변을 놓았을 때의 값) 으로 나눈 비도 낸다. 화면 끝에 닿은 박스는 뺀다.

  python tools/modal_stats.py --zip <data.zip> [--out <txt>]
"""
import argparse
import csv
import io
import os
import zipfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 391, 301
BAND = 2
MODS = ("RGB", "열화상", "이벤트")
ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--out", default="")
A = ap.parse_args()
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def band_max(a, axis):
    """축을 따라 ±BAND 안의 최댓값 (화면 밖은 0)."""
    out = a.copy()
    for k in range(1, BAND + 1):
        for sh in (k, -k):
            b = np.roll(a, sh, axis)
            if axis == 0:
                (b[:k] if sh > 0 else b[-k:])[...] = 0
            else:
                (b[:, :k] if sh > 0 else b[:, -k:])[...] = 0
            out = np.maximum(out, b)
    return out


z = zipfile.ZipFile(A.zip)
with open(os.path.join(REPO, "splits", "author_split.csv"), encoding="utf-8") as f:
    test = [r["stem"] for r in csv.DictReader(f) if r["split"] == "test"]
img = {d: {k: [] for k in ("ev_base", "ev_act", "ev_act_in") + tuple(m + s for m in MODS for s in ("_mean", "_std"))} for d in (True, False)}
box = {d: {m: {"side": [], "bg": []} for m in MODS} for d in (True, False)}
for stem in test:
    a = np.load(io.BytesIO(z.read(f"data/images/{stem}.npy")), allow_pickle=False).astype(np.float32)
    assert a.shape == (H, W, 5), a.shape
    mod = {"RGB": a[..., :3].mean(-1), "열화상": a[..., 3], "이벤트": a[..., 4]}
    day = stem.startswith("frame_")
    boxes = []
    for line in z.read(f"data/labels/{stem}.txt").decode("utf-8", "replace").splitlines():
        p = line.split()
        if len(p) >= 5:
            cx, cy, w, h = (float(v) for v in p[1:5])
            boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    inside = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in boxes:
        inside[max(int(round(y1)), 0):int(round(y2)), max(int(round(x1)), 0):int(round(x2))] = True
    ev = mod["이벤트"]
    vals, cnt = np.unique(ev.round(), return_counts=True)
    act = np.abs(ev - vals[cnt.argmax()]) > 5
    d = img[day]
    d["ev_base"].append(vals[cnt.argmax()]); d["ev_act"].append(act.mean()); d["ev_act_in"].append(act[inside].mean() if inside.any() else np.nan)
    for k, m in mod.items():
        gx = np.zeros_like(m); gx[:, 1:-1] = np.abs(m[:, 2:] - m[:, :-2]) / 2          # 세로 변에 수직
        gy = np.zeros_like(m); gy[1:-1] = np.abs(m[2:] - m[:-2]) / 2                   # 가로 변에 수직
        bx, by = band_max(gx, 1), band_max(gy, 0)
        bg = (bx.mean() + by.mean()) / 2
        d[k + "_mean"].append(m.mean()); d[k + "_std"].append(m.std())
        for x1, y1, x2, y2 in boxes:
            if x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1:
                continue
            xa, xb, ya, yb = int(round(x1)), min(int(round(x2)), W - 1), int(round(y1)), min(int(round(y2)), H - 1)
            if xb - xa < 4 or yb - ya < 4:
                continue
            box[day][k]["side"].append(np.concatenate([bx[ya:yb, xa], bx[ya:yb, xb], by[ya, xa:xb], by[yb, xa:xb]]).mean())
            box[day][k]["bg"].append(bg)

med = lambda v: float(np.nanmedian(np.asarray(v, float)))
nd, nn = len(img[True]["ev_base"]), len(img[False]["ev_base"])
say(f"저자 분할 test {nd + nn} 장 (낮 {nd} · 밤 {nn}) — 모달마다 들어 있는 정보, 0–255 값")
say()
say("[1] 밝기 (사진마다 잰 값의 중앙값)")
say("  모달      낮 평균   낮 표준편차   밤 평균   밤 표준편차")
for k in MODS:
    say(f"  {k:6s} {med(img[True][k + '_mean']):8.1f} {med(img[True][k + '_std']):10.1f} {med(img[False][k + '_mean']):10.1f} {med(img[False][k + '_std']):10.1f}")
say()
say("[2] 이벤트 채널: 바탕값 (사진의 최빈값) 에서 5 넘게 벗어난 픽셀의 비율")
for nm, d in (("낮", img[True]), ("밤", img[False])):
    v = np.asarray(d["ev_act"])
    q = np.quantile(v, [0.5, 0.9, 0.99]) * 100
    say(f"  {nm}: 바탕값 중앙값 {med(d['ev_base']):.0f} · 화면 전체 중앙값 {q[0]:.2f} % (90 % 분위 {q[1]:.2f} · 99 % 분위 {q[2]:.2f}) · 정답 박스 안 중앙값 {med(d['ev_act_in']) * 100:.2f} % · "
        f"벗어난 픽셀이 화면의 1 % 도 안 되는 사진 {100 * (v < 0.01).mean():.1f} %")
say()
say(f"[3] 박스 변 위의 경계 세기 (변 ±{BAND} px 안 최대, 변을 따라 평균) — 화면 끝에 안 닿은 박스, 중앙값")
say("  모달      낮: 박스 변   화면 평균    비      밤: 박스 변   화면 평균    비     박스 수 (낮 · 밤)")
for k in MODS:
    r = []
    for d in (True, False):
        s, b = np.asarray(box[d][k]["side"]), np.asarray(box[d][k]["bg"])
        r.append((np.median(s), np.median(b), np.median(s / np.maximum(b, 1e-6)), len(s)))
    say(f"  {k:6s} {r[0][0]:10.1f} {r[0][1]:10.1f} {r[0][2]:7.2f} {r[1][0]:12.1f} {r[1][1]:10.1f} {r[1][2]:7.2f}     {r[0][3]} · {r[1][3]}")
say("  비가 1 이면 박스 변에 화면의 아무 곳과 같은 만큼의 경계만 있다는 뜻. 값 자체 (0–255 단위) 가 0 에 가까우면 비가 커도 정보는 없다.")
if A.out:
    with open(A.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
