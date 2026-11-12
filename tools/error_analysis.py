"""예측을 뜯어본다 — 점수를 어디서 잃는지, 박스 오차가 어떤 성격인지, 라벨이 얼마나 흔들리는지 (학습 없음, numpy 만 있으면 됨).
README 실험 1 의 '결과' 에 적은 숫자를 낸다. `src/report.py` 의 표 3 (찾은 비율) 보다 깊이 들어간 분석이다.

  1. IoU 문턱마다 AP (COCO 방식, 직접 계산 — metrics.json 의 값과 맞는지 먼저 확인)
  2. 변마다 오차 (예측 − 정답, 원본 px): 치우침 · 퍼짐
  3. 시드끼리 일치하는 정도 vs 정답과 일치하는 정도 (오차 = 시드가 공유하는 몫 + 시드마다 다른 몫)
  4. 라벨이 흔들리는 정도: test 프레임과 번호가 바로 옆인 train 프레임의 같은 차 박스 — 폭 · 높이 차이 (카메라가 움직여도 차 크기는 안 변한다)
  5. 옆 train 프레임의 정답을 그대로 베끼면 AP 가 얼마인가
  6. 천장 어림: 찾은 박스의 위치만 바꿔 AP 를 다시 잰다 (세 시드 평균 · 완벽 + 라벨 흔들림 · 완벽)
  7. IoU 0.9 에 못 미친 정답 박스는 어떤 박스인가

  python tools/error_analysis.py --runs <runs 폴더> --zip <data.zip> [--group paper_sgd_full] [--out <txt>]
    runs 폴더 = results.zip 을 푼 곳의 runs/ (그 안에 <group>_s0 · _s1 · _s2 / eval_test / preds.npz · metrics.json)
"""
import argparse
import csv
import json
import os
import zipfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap_ = argparse.ArgumentParser()
ap_.add_argument("--runs", required=True)
ap_.add_argument("--zip", required=True)
ap_.add_argument("--group", default="paper_sgd_full")
ap_.add_argument("--out", default="")
A = ap_.parse_args()
RUNS, ZIP = A.runs, A.zip
W, H = 391, 301
THS = np.round(np.arange(0.5, 0.96, 0.05), 2)
SEEDS = [f"{A.group}_s{k}" for k in (0, 1, 2)]
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def iou_pair(a, b):
    w = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return w * h / (ar(a) + ar(b) - w * h + 1e-9)


def ap_table(dets, gts, ths=THS):
    """dets[i] = (boxes (n, 4), scores (n,)), gts[i] = boxes (m, 4). → IoU 문턱마다 AP (COCO: 점수순 탐욕 짝짓기 · 101 점 보간 · 사진마다 100 개까지)."""
    out = []
    npos = sum(len(g) for g in gts)
    for t in ths:
        sc, tp = [], []
        for (b, s), g in zip(dets, gts):
            o = np.argsort(-s, kind="mergesort")[:100]
            b, s = b[o], s[o]
            used = np.zeros(len(g), bool)
            M = iou(b, g) if len(b) and len(g) else np.zeros((len(b), len(g)))
            for j in range(len(b)):
                m = np.where(used, -1.0, M[j]) if len(g) else np.array([])
                k = int(m.argmax()) if len(g) else -1
                hit = k >= 0 and m[k] >= t
                if hit:
                    used[k] = True
                tp.append(hit); sc.append(s[j])
        o = np.argsort(-np.array(sc), kind="mergesort")
        tpc = np.cumsum(np.array(tp, float)[o]); fpc = np.cumsum(1 - np.array(tp, float)[o])
        rec, prec = tpc / max(npos, 1), tpc / np.maximum(tpc + fpc, 1e-9)
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
        out.append(float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx])))
    return np.array(out)


def load(run):
    p = np.load(os.path.join(RUNS, run, "eval_test", "preds.npz"))
    stems = [str(s) for s in p["stems"]]
    dets = [(p["det_box"][p["det_img"] == i], p["det_score"][p["det_img"] == i]) for i in range(len(stems))]
    gts = [p["gt_box"][p["gt_img"] == i] for i in range(len(stems))]
    return stems, dets, gts


def best(dets, gts, score=0.5):
    """정답 박스마다 가장 잘 맞는 예측 (점수 ≥ score). → (예측 박스 (N, 4), IoU (N,)) — 예측이 없으면 nan · 0."""
    B, I = [], []
    for (b, s), g in zip(dets, gts):
        b = b[s >= score]
        if len(g) == 0:
            continue
        if len(b) == 0:
            B.append(np.full((len(g), 4), np.nan)); I.append(np.zeros(len(g))); continue
        M = iou(g, b)
        k = M.argmax(1)
        B.append(b[k]); I.append(M[np.arange(len(g)), k])
    return np.concatenate(B), np.concatenate(I)


# ───────────────────────── 자료 ─────────────────────────
data = {r: load(r) for r in SEEDS}
stems, _, gts = data[SEEDS[0]]
G = np.concatenate([g for g in gts if len(g)])
gimg = np.concatenate([np.full(len(g), i) for i, g in enumerate(gts)])
day = np.array([stems[i].startswith("frame_") for i in gimg])
gw, gh = G[:, 2] - G[:, 0], G[:, 3] - G[:, 1]
border = (G[:, 0] <= 1) | (G[:, 1] <= 1) | (G[:, 2] >= W - 1) | (G[:, 3] >= H - 1)
small = gw * gh < 32 ** 2
say("%s (시드 0 · 1 · 2) 의 test 예측 분석 — 원본 301×391 좌표, 정답 박스 %d 개 (낮 %d · 밤 %d), 사진 %d 장" % (A.group, len(G), day.sum(), (~day).sum(), len(stems)))
say("정답 박스 한 변 (√넓이) 중앙값 %.1f px · 폭 %.1f · 높이 %.1f · 화면 끝에 닿은 박스 %.1f %% · small (< 32²) %.1f %%" % (
    np.median(np.sqrt(gw * gh)), np.median(gw), np.median(gh), 100 * border.mean(), 100 * small.mean()))

# ── 1. IoU 문턱마다 AP ──
say("\n[1] IoU 문턱마다 AP (%, 시드 3 개 평균). 맨 오른쪽 = 열 평균 (= mAP)")
isday = np.array([s.startswith("frame_") for s in stems])
rows = {}
for name, sel in (("전체", np.ones(len(stems), bool)), ("낮", isday), ("밤", ~isday)):
    aps = []
    for r in SEEDS:
        _, d, g = data[r]
        aps.append(ap_table([d[i] for i in np.where(sel)[0]], [g[i] for i in np.where(sel)[0]]))
    rows[name] = np.mean(aps, 0)
    if name == "전체":
        ref = [json.load(open(os.path.join(RUNS, r, "eval_test", "metrics.json")))["all"] for r in SEEDS]
        mine = [(a.mean(), a[0], a[5]) for a in aps]
        say("  확인 (직접 계산 vs torchmetrics, 시드 0): mAP %.2f vs %.2f · mAP50 %.2f vs %.2f · mAP75 %.2f vs %.2f" % (
            100 * mine[0][0], 100 * ref[0]["map"], 100 * mine[0][1], 100 * ref[0]["map_50"], 100 * mine[0][2], 100 * ref[0]["map_75"]))
say("  IoU   " + " ".join(f"{t:6.2f}" for t in THS) + "    평균")
for name, a in rows.items():
    say(f"  {name:4s}" + " ".join(f"{100 * v:6.1f}" for v in a) + f"   {100 * a.mean():6.2f}")
a = rows["전체"]
say("  mAP 에서 잃는 %.1f 점 (100 − %.1f) 가운데: IoU 0.50–0.75 여섯 칸에서 %.1f 점, 0.80–0.95 네 칸에서 %.1f 점 (0.90 · 0.95 두 칸만 %.1f 점)" % (
    100 - 100 * a.mean(), 100 * a.mean(), (1 - a[:6]).sum() * 10, (1 - a[6:]).sum() * 10, (1 - a[8:]).sum() * 10))
say("  밤 − 낮 (점): " + " ".join(f"{100 * (n - d):6.1f}" for n, d in zip(rows["밤"], rows["낮"])) + f"   {100 * (rows['밤'].mean() - rows['낮'].mean()):6.2f}")

# ── 2. 변마다 오차 ──
say("\n[2] 변마다 오차 = 예측 − 정답 (px, 원본 좌표). 정답 박스마다 점수 ≥ 0.5 인 예측 가운데 가장 잘 맞는 것, IoU ≥ 0.5 인 짝만. 시드 셋을 모음")
P = {r: best(data[r][1], data[r][2]) for r in SEEDS}
E = np.concatenate([P[r][0] - G for r in SEEDS]); IO = np.concatenate([P[r][1] for r in SEEDS])
ok = IO >= 0.5
rep = lambda m: np.concatenate([m] * len(SEEDS))
say("  짝 %d 개 (정답의 %.1f %%). IoU 중앙값 %.3f · IoU ≥ 0.9 인 비율 %.1f %% · ≥ 0.95 %.1f %%" % (ok.sum(), 100 * ok.mean(), np.median(IO[ok]), 100 * (IO >= 0.9).mean(), 100 * (IO >= 0.95).mean()))
say("  변        치우침   퍼짐(std)  |오차| 중앙값   90 %% 분위")
for k, nm in enumerate(("왼 x1", "위 y1", "오른 x2", "아래 y2")):
    e = E[ok, k]
    say(f"  {nm:7s} {e.mean():+7.2f}  {e.std():8.2f}  {np.median(np.abs(e)):10.2f}  {np.percentile(np.abs(e), 90):9.2f}")
inner = ok & ~rep(border)
ae = np.abs(E[inner])
say("  화면 끝에 안 닿은 박스만: 네 변 |오차| 중앙값 %.2f px · 평균 %.2f px · RMS %.2f px" % (np.median(ae), ae.mean(), np.sqrt((E[inner] ** 2).mean())))
say("  박스 크기 차이 (예측 − 정답): 폭 %+.2f ± %.2f px · 높이 %+.2f ± %.2f px (화면 끝에 안 닿은 박스)" % (
    (E[inner, 2] - E[inner, 0]).mean(), (E[inner, 2] - E[inner, 0]).std(), (E[inner, 3] - E[inner, 1]).mean(), (E[inner, 3] - E[inner, 1]).std()))
for nm, m in (("낮", rep(day)), ("밤", rep(~day)), ("화면 끝에 닿음", rep(border)), ("안 닿음", rep(~border)), ("small", rep(small)), ("small 아님", rep(~small))):
    q = ok & m
    say("  %-9s 짝 %5d · IoU 중앙값 %.3f · IoU ≥ 0.9 %5.1f %% · 네 변 |오차| 중앙값 %.2f px" % (nm, q.sum(), np.median(IO[q]), 100 * (IO[m] >= 0.9).mean(), np.median(np.abs(E[q]))))
say("  참고: 한 변 52 px 정사각 박스에서 네 변이 모두 e px 씩 어긋나면 (안쪽 둘 · 바깥쪽 둘이 아니라 같은 쪽으로 밀리면) IoU = (52 − e)² / (52 + e)² — e = 1.4 px 에서 0.90, 0.7 px 에서 0.95")

# ── 3. 시드끼리 vs 정답 ──
say("\n[3] 시드끼리 일치하는 정도 vs 정답과 일치하는 정도 (세 시드 모두 IoU ≥ 0.5 로 찾은 정답 박스, 화면 끝에 안 닿은 것)")
B3 = np.stack([P[r][0] for r in SEEDS]); I3 = np.stack([P[r][1] for r in SEEDS])
m3 = (I3 >= 0.5).all(0) & ~border
pair = np.concatenate([iou_pair(B3[i][m3], B3[j][m3]) for i, j in ((0, 1), (0, 2), (1, 2))])
mean_box = B3[:, m3].mean(0)
say("  박스 %d 개. 시드 ↔ 시드 IoU 중앙값 %.3f (≥ 0.9 인 비율 %.1f %%)  ·  시드 ↔ 정답 IoU 중앙값 %.3f (≥ 0.9 %.1f %%)" % (
    m3.sum(), np.median(pair), 100 * (pair >= 0.9).mean(), np.median(I3[:, m3]), 100 * (I3[:, m3] >= 0.9).mean()))
ie = iou_pair(mean_box, G[m3])
say("  세 시드의 박스를 평균한 것 ↔ 정답: IoU 중앙값 %.3f (≥ 0.9 %.1f %%) — 시드마다 다른 몫을 지워도 이만큼만 오른다" % (np.median(ie), 100 * (ie >= 0.9).mean()))
shared = mean_box - G[m3]; own = B3[:, m3] - mean_box[None]
say("  변 오차의 제곱 평균을 둘로 나누면: 세 시드가 공유하는 몫 (평균 박스 − 정답) RMS %.2f px  ·  시드마다 다른 몫 RMS %.2f px  → 공유하는 몫이 분산의 %.0f %%" % (
    np.sqrt((shared ** 2).mean()), np.sqrt((own ** 2).mean()), 100 * (shared ** 2).mean() / ((shared ** 2).mean() + (own ** 2).mean())))

# ── 4 · 5. 라벨의 흔들림: test 프레임과 바로 옆 train 프레임 ──
z = zipfile.ZipFile(ZIP)
split = {r["stem"]: (r["stem"].rsplit("_", 1)[0], int(r["stem"].rsplit("_", 1)[1]), r["split"])
         for r in csv.DictReader(open(os.path.join(REPO, "splits", "author_split.csv"), encoding="utf-8"))}
by = {}
for s, (g, i, sp) in split.items():
    if sp == "train":
        by.setdefault(g, []).append(i)
by = {g: np.array(sorted(v)) for g, v in by.items()}


def labels(stem):
    b = []
    for ln in z.read(f"data/labels/{stem}.txt").decode("utf-8", "replace").splitlines():
        p = ln.split()
        if len(p) >= 5:
            cx, cy, w, h = (float(v) for v in p[1:5])
            b.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(b, np.float32).reshape(-1, 4)


nb_dets, gaps, dW, dH, dS, pio, pday = [], [], [], [], [], [], []
for i, s in enumerate(stems):
    g, idx, _ = split[s]
    a = by[g]
    k = a[np.abs(a - idx).argmin()]
    gaps.append(abs(int(k) - idx))
    nb = labels(f"{g}_{int(k):05d}")
    nb_dets.append((nb, np.ones(len(nb), np.float32)))
    t = gts[i]
    if len(t) and len(nb):
        M = iou(t, nb)
        j = M.argmax(1)
        keep = (M[np.arange(len(t)), j] >= 0.5) & ~((t[:, 0] <= 1) | (t[:, 1] <= 1) | (t[:, 2] >= W - 1) | (t[:, 3] >= H - 1))
        keep &= ~((nb[j][:, 0] <= 1) | (nb[j][:, 1] <= 1) | (nb[j][:, 2] >= W - 1) | (nb[j][:, 3] >= H - 1))
        if keep.any() and gaps[-1] == 1:
            a_, b_ = t[keep], nb[j][keep]
            dW.append((a_[:, 2] - a_[:, 0]) - (b_[:, 2] - b_[:, 0])); dH.append((a_[:, 3] - a_[:, 1]) - (b_[:, 3] - b_[:, 1]))
            d = a_ - b_
            if len(a_) >= 3:                                         # 사진 안 박스들이 같이 움직인 몫 (카메라) 을 빼고 남는 것
                dS.append(d - np.median(d, 0, keepdims=True))
            pio.append(M[np.arange(len(t)), j][keep]); pday.append(np.full(keep.sum(), s.startswith("frame_")))
gaps = np.array(gaps); dW, dH, pio, pday = (np.concatenate(v) for v in (dW, dH, pio, pday))
say("\n[4] 라벨이 흔들리는 정도 — test 프레임과 번호가 바로 옆인 train 프레임 (%.1f %% 가 번호 차 1) 의 같은 차 박스 %d 쌍 (화면 끝에 안 닿은 것, IoU ≥ 0.5)" % (100 * (gaps == 1).mean(), len(dW)))
say("  옆 프레임 정답 ↔ 이 프레임 정답: IoU 중앙값 %.3f (≥ 0.9 인 비율 %.1f %% · ≥ 0.95 %.1f %%)  — 카메라 · 차의 움직임이 섞인 값" % (np.median(pio), 100 * (pio >= 0.9).mean(), 100 * (pio >= 0.95).mean()))
say("  폭 차이 std %.2f px · 높이 차이 std %.2f px (|차이| 중앙값 %.2f · %.2f). 차의 크기는 한 프레임 사이에 안 변하므로 이건 라벨 붙이기의 흔들림이다" % (dW.std(), dH.std(), np.median(np.abs(dW)), np.median(np.abs(dH))))
sig = np.sqrt((dW.var() + dH.var()) / 2) / 2
say("  변 하나의 라벨 흔들림 σ ≈ std(폭 차이) / 2 = %.2f px (두 프레임 · 두 변의 흔들림이 독립이라고 보면 var(폭 차이) = 4σ²)" % sig)
say("  낮 %.2f px · 밤 %.2f px" % (np.sqrt((dW[pday].var() + dH[pday].var()) / 2) / 2, np.sqrt((dW[~pday].var() + dH[~pday].var()) / 2) / 2))
if dS:
    r = np.concatenate(dS)
    say("  (다른 방법) 박스 3 개 이상인 사진에서 박스들이 같이 움직인 몫을 빼고 남은 변 차이: std %.2f px → σ ≈ %.2f px (차가 따로 움직인 것도 섞임, %d 변)" % (r.std(), r.std() / np.sqrt(2), r.size))
say("  견줌: 모델의 변 오차 (화면 끝에 안 닿은 박스) RMS %.2f px, 그 가운데 시드가 공유하는 몫 RMS %.2f px" % (np.sqrt((E[inner] ** 2).mean()), np.sqrt((shared ** 2).mean())))

say("\n[5] 옆 train 프레임의 정답 박스를 그대로 베껴 예측으로 내면 (점수 모두 1)")
a = ap_table(nb_dets, gts)
say("  IoU   " + " ".join(f"{t:6.2f}" for t in THS) + "    평균")
say("  베낌  " + " ".join(f"{100 * v:6.1f}" for v in a) + f"   {100 * a.mean():6.2f}")
say("  모델  " + " ".join(f"{100 * v:6.1f}" for v in rows["전체"]) + f"   {100 * rows['전체'].mean():6.2f}")

# ── 6. 천장: 위치를 완벽하게 맞혀도 라벨이 흔들리는 만큼은 못 넘는다 ──
say("\n[6] 천장 어림 — 시드 0 의 예측에서 '정답과 짝이 된 박스' 의 좌표만 바꿔 AP 를 다시 잰다 (점수 · 못 찾은 것 · 잘못 찾은 것은 그대로)")
still_d = []                                                        # 거의 안 움직인 쌍 (박스 중심이 1 px 안): 변 차이가 곧 두 라벨의 흔들림 차이
for i, s_ in enumerate(stems):
    g, idx, _ = split[s_]
    a_ = by[g]; k = a_[np.abs(a_ - idx).argmin()]
    if abs(int(k) - idx) != 1 or len(gts[i]) == 0:
        continue
    nb = labels(f"{g}_{int(k):05d}")
    if len(nb) == 0:
        continue
    M = iou(gts[i], nb); j = M.argmax(1)
    t, n = gts[i], nb[j]
    c = np.abs((t[:, :2] + t[:, 2:]) / 2 - (n[:, :2] + n[:, 2:]) / 2).max(1)
    keep = (M[np.arange(len(t)), j] >= 0.5) & (c <= 1.0) & (t[:, 0] > 1) & (t[:, 1] > 1) & (t[:, 2] < W - 1) & (t[:, 3] < H - 1) \
        & (n[:, 0] > 1) & (n[:, 1] > 1) & (n[:, 2] < W - 1) & (n[:, 3] < H - 1)
    still_d.append((t - n)[keep])
still_d = np.concatenate(still_d)
say("  거의 안 움직인 쌍 (옆 프레임과 박스 중심 차 ≤ 1 px) %d 개: 변 차이 std %.2f px · |차이| 중앙값 %.2f px · 차이가 정확히 0 인 변 %.1f %% → 라벨 하나의 흔들림 σ ≈ std / √2 = %.2f px" % (
    len(still_d), still_d.std(), np.median(np.abs(still_d)), 100 * (np.abs(still_d) < 1e-3).mean(), still_d.std() / np.sqrt(2)))
say("  이 쌍들의 옆 프레임 정답 ↔ 이 프레임 정답 IoU: 중앙값 %.3f · ≥ 0.9 %.1f %% · ≥ 0.95 %.1f %%  (안 움직였는데도 라벨끼리 이만큼만 일치)" % (
    np.median(iou_pair(G[:1].repeat(len(still_d), 0) * 0 + np.array([[0, 0, 41.0, 67.0]]), np.array([[0, 0, 41.0, 67.0]]) + still_d)),
    100 * (iou_pair(np.tile([[0, 0, 41.0, 67.0]], (len(still_d), 1)), np.array([[0, 0, 41.0, 67.0]]) + still_d) >= 0.9).mean(),
    100 * (iou_pair(np.tile([[0, 0, 41.0, 67.0]], (len(still_d), 1)), np.array([[0, 0, 41.0, 67.0]]) + still_d) >= 0.95).mean()))
rng = np.random.default_rng(0)
_, d0, g0 = data[SEEDS[0]]


def swap(make):
    """make(정답 박스들, 원래 예측 박스들, 사진 번호) → 새 박스. 짝 = 정답마다 가장 잘 맞는 예측 (점수 ≥ 0.5, IoU ≥ 0.5)."""
    out = []
    for i, ((b, sc), g) in enumerate(zip(d0, g0)):
        b = b.copy()
        if len(g) and len(b):
            M = iou(g, b); M[:, sc < 0.5] = 0
            k = M.argmax(1); hit = M[np.arange(len(g)), k] >= 0.5
            b[k[hit]] = make(g[hit], b[k[hit]], i, np.where(hit)[0])
        out.append((b, sc))
    return ap_table(out, g0)


base_ap = ap_table(d0, g0)
B3_by = {}
off = 0
for i, g in enumerate(g0):
    B3_by[i] = (B3[:, off:off + len(g)], I3[:, off:off + len(g)]); off += len(g)


def ens(g, b, i, w):
    bb, ii = B3_by[i]
    m = bb[:, w].mean(0)
    return np.where((ii[:, w] >= 0.5).all(0)[:, None], m, b)


aps = {"지금 (시드 0)": base_ap,
       "세 시드 평균 박스": swap(ens),
       "완벽 + 라벨 흔들림 (안 움직인 쌍의 차이 / √2)": np.mean([swap(lambda g, b, i, w: g + still_d[rng.integers(0, len(still_d), len(g))] / np.sqrt(2)) for _ in range(3)], 0),
       "완벽 + 라벨 흔들림 (폭 · 높이 차이 / 2, 큰 쪽 어림)": np.mean([swap(lambda g, b, i, w: g + np.stack(
           [dW[rng.integers(0, len(dW), len(g))], dH[rng.integers(0, len(dH), len(g))], dW[rng.integers(0, len(dW), len(g))], dH[rng.integers(0, len(dH), len(g))]], 1) / 2)
           for _ in range(3)], 0),
       "완벽 (정답 그대로)": swap(lambda g, b, i, w: g)}
say("  IoU                                          " + " ".join(f"{t:6.2f}" for t in THS) + "    평균")
for k, a_ in aps.items():
    say(f"  {k:<38s}" + " ".join(f"{100 * v:6.1f}" for v in a_) + f"   {100 * a_.mean():6.2f}")
say("  읽는 법: '완벽' = 찾은 박스의 위치가 정답과 똑같을 때 (못 찾은 것 · 잘못 찾은 것 때문에 100 이 아님). '완벽 + 라벨 흔들림' = 모델이 차의 위치를 완벽히 알아도,")
say("  채점하는 정답 라벨이 옆 프레임 라벨과 어긋나는 만큼 흔들린다고 볼 때의 값 — 위치 개선으로 닿을 수 있는 위쪽 끝의 어림.")
say("  흔들림 어림 둘: 안 움직인 쌍 (σ %.2f px — 중심이 안 움직인 쌍만 골라서 작게 나오는 쪽) · 폭 · 높이 차이 (σ %.2f px — 움직이는 차가 섞여 크게 나오는 쪽). 참값은 그 사이." % (
    still_d.std() / np.sqrt(2), sig))

# ── 7. IoU 0.9 에 못 미친 정답 박스는 어떤 박스인가 ──
low = (I3 < 0.9).mean(0)
say("\n[7] IoU 0.9 에 못 미친 정답 박스는 어떤 박스인가 (시드 평균 %.1f %% = %.0f 개 꼴)" % (100 * low.mean(), low.sum()))
say("  묶음                         정답 수   못 미친 비율   못 미친 것 가운데 몫")
for nm, m in (("낮 · 안쪽 · small 아님", day & ~border & ~small), ("밤 · 안쪽 · small 아님", ~day & ~border & ~small),
              ("화면 끝에 닿음 (small 아님)", border & ~small), ("small (화면 끝에 닿음)", small & border), ("small (안쪽)", small & ~border)):
    say("  %-24s %6d   %8.1f %%   %10.1f %%" % (nm, m.sum(), 100 * low[m].mean(), 100 * low[m].sum() / low.sum()))
say("  small 가운데 화면 끝에 닿은 것 %.1f %% · 화면 끝에 닿은 것 가운데 small %.1f %%" % (100 * border[small].mean(), 100 * small[border].mean()))
if A.out:
    open(A.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
