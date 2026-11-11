"""결과 정리 — 판정 규칙은 README '판정 기준' 과 같고, 결과를 보기 전에 정했다 (2026-11-11).

  python src/report.py select  --runs <runs 폴더> --out <choice.json>   # 실험 2: 검증 mAP 로 최적화 고르기
  python src/report.py summary --runs <runs 폴더> --out <summary 폴더>  # 표 · 판정 · 오차 분해 → tables.md · summary.json · runs.csv

실행 이름 = <팔>_<최적화>_<분할>_s<시드> (예: paper_sgd_full_s1, base_adamw_sub_s0, author_sgd_full_s0).
지표는 % (×100). 로컬에서도 받은 results.zip 을 풀어 같은 명령으로 다시 낼 수 있다 (numpy 만 있으면 됨).
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

PAPER_MAP, PAPER_MAP50, PAPER_PARAMS = 84.24, 98.95, 60.01
W, H = 391, 301
NAME = re.compile(r"^(author|paper|base|inject)_(sgd|adamw)_(full|sub)_s(\d+)$")
COLS = [("all", "map"), ("all", "map_50"), ("all", "map_75"), ("day", "map"), ("night", "map"), ("all", "map_small"), ("all", "map_medium")]


def load_runs(root):
    runs = []
    for n in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        m = NAME.match(n)
        if not m:
            continue
        ev = "eval_val" if m.group(3) == "sub" else "eval_test"
        f = os.path.join(root, n, ev, "metrics.json")
        if os.path.exists(f):
            runs.append({"name": n, "arm": m.group(1), "opt": m.group(2), "split": m.group(3), "seed": int(m.group(4)),
                         "group": n.rsplit("_s", 1)[0], "dir": os.path.join(root, n), "ev": ev, "m": json.load(open(f))})
    return runs


def stat(vals):
    v = np.array(vals, float)
    return {"n": len(v), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def groups(runs):
    g = {}
    for r in runs:
        g.setdefault(r["group"], []).append(r)
    return g


def gstat(rs, part="all", key="map"):
    v = [100 * r["m"][part][key] for r in rs if r["m"][part].get(key) is not None]   # None = 그 부분에 사진 · 박스가 없음
    return stat(v) if v else None


def judge(x, ref):
    """x · ref = gstat 결과. Δ = 평균 차, 폭 = 두 팔의 시드 범위 (최대 − 최소) 중 큰 것."""
    if x is None or ref is None or x["n"] < 2 or ref["n"] < 2:
        return "판정 보류 (시드 2 개 미만)", None, None
    d = x["mean"] - ref["mean"]
    spread = max(x["max"] - x["min"], ref["max"] - ref["min"])
    verdict = "개선" if d > spread else ("약한 개선" if d > 0 else "개선 없음")
    return verdict, d, spread


def select(a):
    rs = {r["opt"]: r for r in load_runs(a.runs) if r["arm"] == "base" and r["split"] == "sub"}
    if set(rs) != {"sgd", "adamw"}:
        print(f"!!! [select] base sub 실행 둘 (sgd · adamw) 이 다 있어야 함 — 있는 것: {sorted(rs)}")
        return 1
    v = {k: 100 * (r["m"]["all"]["map"] or 0.0) for k, r in rs.items()}
    pick = "adamw" if v["adamw"] - v["sgd"] >= 0.3 else "sgd"          # 규칙: 0.3 미만 차이면 저자 최적화 (sgd)
    out = {"opt": pick, "val_map": {k: round(x, 2) for k, x in v.items()}, "rule": "adamw 가 sgd 보다 val mAP 0.3 이상 높으면 adamw, 아니면 sgd"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"[select] val mAP sgd {v['sgd']:.2f} · adamw {v['adamw']:.2f} → {pick}")
    return 0


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def decompose(pred_file, score=0.5, ths=(0.5, 0.75, 0.9)):
    """정답 박스 단위: 점수 ≥ score 예측으로 IoU 문턱마다 찾았는가 (점수순 탐욕 짝짓기) · 가장 잘 맞는 예측의 IoU · 잘못 찾은 수."""
    p = np.load(pred_file)
    stems = p["stems"]
    keep = p["det_score"] >= score
    di, db, ds = p["det_img"][keep], p["det_box"][keep], p["det_score"][keep]
    rows, fp = [], 0
    for i, s in enumerate(stems):
        g = p["gt_box"][p["gt_img"] == i]
        d, sc = db[di == i], ds[di == i]
        d = d[np.argsort(-sc)]
        M = iou(g, d) if len(g) and len(d) else np.zeros((len(g), len(d)))
        best = M.max(1) if len(d) else np.zeros(len(g))
        hit = {}
        for t in ths:
            used = np.zeros(len(g), bool)
            for j in range(len(d)):
                cand = np.where(~used & (M[:, j] >= t))[0] if len(g) else []
                if len(cand):
                    used[cand[np.argmax(M[cand, j])]] = True
                elif t == ths[0]:
                    fp += 1
            hit[t] = used
        area = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        trunc = (g[:, 0] <= 1) | (g[:, 1] <= 1) | (g[:, 2] >= W - 1) | (g[:, 3] >= H - 1)
        for k in range(len(g)):
            rows.append((str(s).startswith("frame_"), area[k] < 32 ** 2, 32 ** 2 <= area[k] < 96 ** 2, bool(trunc[k]), best[k],
                         *[hit[t][k] for t in ths]))
    r = np.array(rows, dtype=float) if rows else np.zeros((0, 5 + len(ths)))
    subsets = {"전체": np.ones(len(r), bool), "낮": r[:, 0] == 1, "밤": r[:, 0] == 0, "small": r[:, 1] == 1, "medium": r[:, 2] == 1,
               "경계에 걸림": r[:, 3] == 1, "경계 안쪽": r[:, 3] == 0}
    out = {}
    for k, msk in subsets.items():
        q = r[msk]
        out[k] = {"n_gt": int(msk.sum()), "best_iou_median": float(np.median(q[:, 4])) if len(q) else None,
                  **{f"recall_{t}": float(q[:, 5 + i].mean()) if len(q) else None for i, t in enumerate(ths)}}
    out["fp_per_image"] = fp / max(len(stems), 1)
    return out


def fmt(s, prec=2):
    if s is None:
        return "—"
    return f"{s['mean']:.{prec}f} ± {s['std']:.{prec}f}" if s["n"] > 1 else f"{s['mean']:.{prec}f}"


def summary(a):
    runs = load_runs(a.runs)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "split"] + [f"{p}_{k}" for p, k in COLS] + ["params_m", "latency_ms"])
        for r in runs:
            w.writerow([r["name"], r["ev"][5:]] + ["" if r["m"][p].get(k) is None else round(100 * r["m"][p][k], 2) for p, k in COLS]
                       + [r["m"]["params"]["detector_m"], r["m"].get("latency_ms_p50", "")])
    G = groups(runs)
    test = {k: v for k, v in G.items() if not k.endswith("_sub")}
    order = ["author_sgd_full", "paper_sgd_full"] + sorted(k for k in test if k.startswith("base_")) + sorted(k for k in test if k.startswith("inject_"))
    label = {"author_sgd_full": "저자 스크립트 그대로", "paper_sgd_full": "논문 설정 재현 (우리 학습 루프)",
             "base_sgd_full": "기본 설정 보강 (SGD)", "base_adamw_full": "기본 설정 보강 (AdamW)",
             "inject_sgd_full": "우리 방법: + DINOv2-L 게이트 주입 (SGD)", "inject_adamw_full": "우리 방법: + DINOv2-L 게이트 주입 (AdamW)"}
    L = ["## 표 1. test 1,950 장 (%, 시드 평균 ± 표준편차)", "",
         "| 설정 | 시드 | mAP | mAP50 | mAP75 | 낮 mAP | 밤 mAP | small mAP | medium mAP | 파라미터 (M) | 추론 (ms) |",
         "|---|---|---|---|---|---|---|---|---|---|---|",
         f"| 논문 보고값 | 1 | {PAPER_MAP:.2f} | {PAPER_MAP50:.2f} | — | — | — | — | — | {PAPER_PARAMS:.2f} | — |"]
    res = {"groups": {}, "judge": {}}
    for k in order:
        if k not in test:
            continue
        rs = test[k]
        st = {f"{p}.{kk}": gstat(rs, p, kk) for p, kk in COLS}
        lat = [r["m"].get("latency_ms_p50") for r in rs if r["m"].get("latency_ms_p50")]
        prm = rs[0]["m"]["params"]
        res["groups"][k] = {"seeds": sorted(r["seed"] for r in rs), **{kk: v for kk, v in st.items()}, "params": prm, "latency_ms": lat[0] if lat else None}
        p_txt = f"{prm['detector_m']:.2f}" + (f" + 고정 {prm['frozen_encoder_m']:.0f}" if prm["frozen_encoder_m"] else "")
        L.append(f"| {label.get(k, k)} | {len(rs)} | " + " | ".join(fmt(st[f'{p}.{kk}']) for p, kk in COLS) + f" | {p_txt} | {lat[0] if lat else '—'} |")

    sel = {k: v for k, v in G.items() if k.endswith("_sub")}
    if sel:
        L += ["", "## 표 2. 최적화 고르기 — 검증 780 장 (train 안), 시드 0", "", "| 설정 | mAP | mAP50 |", "|---|---|---|"]
        for k in sorted(sel):
            r = sel[k][0]["m"]["all"]
            L.append(f"| {k.replace('_sub', '')} | " + " | ".join("—" if r[q] is None else f"{100 * r[q]:.2f}" for q in ("map", "map_50")) + " |")
        ch = os.path.join(a.out, "choice.json")
        if os.path.exists(ch):
            L.append(f"\n고른 최적화: **{json.load(open(ch))['opt']}** (규칙: adamw 가 0.3 이상 높을 때만 adamw)")

    J = res["judge"]
    P = test.get("paper_sgd_full")
    ps = gstat(P) if P else None
    if ps:
        J["재현"] = f"{'재현됨' if abs(ps['mean'] - PAPER_MAP) <= 1.5 else '재현 안 됨'} (평균 {ps['mean']:.2f}, 논문 {PAPER_MAP}, 기준 ±1.5)"
        A = test.get("author_sgd_full")
        if A and A[0]["m"]["all"]["map"] is not None:
            av = 100 * A[0]["m"]["all"]["map"]
            ok = ps["min"] - 0.5 <= av <= ps["max"] + 0.5
            J["A/A"] = f"{'같은 학습' if ok else '다른 학습'} (저자 스크립트 {av:.2f}, 우리 루프 범위 {ps['min']:.2f}–{ps['max']:.2f} ± 0.5)"
    for k in order:
        if not (k.startswith("base_") or k.startswith("inject_")) or k not in test:
            continue
        xs = gstat(test[k])
        if xs is None:
            continue
        refs = [("paper_sgd_full", "논문 재현")] + ([("base_" + k.split("_")[1] + "_full", "기본 설정 보강")] if k.startswith("inject_") else [])
        for rk, rl in refs:
            if rk in test:
                v, d, sp = judge(xs, gstat(test[rk]))
                J[f"{label.get(k, k)} vs {rl}"] = v + ("" if d is None else f" (Δ {d:+.2f}, 시드 폭 {sp:.2f})")
        J[f"{label.get(k, k)} vs 논문 보고값"] = f"평균 {xs['mean']:.2f} vs {PAPER_MAP} ({xs['mean'] - PAPER_MAP:+.2f})"
    for k in [g for g in order if g.startswith("inject_")] + [g for g in order if g.startswith("base_")]:   # 논문을 넘음: inject 먼저, 못 넘으면 base
        xs = gstat(test[k]) if k in test else None
        if xs is None or not P:
            continue
        v, _, _ = judge(xs, gstat(P))
        if xs["mean"] > PAPER_MAP and v == "개선":
            J["논문을 넘음"] = f"예 — {label.get(k, k)} (평균 {xs['mean']:.2f} > {PAPER_MAP}, 논문 재현 대비 개선)"
            break
    else:
        if P and any(g.startswith(("base_", "inject_")) for g in test):
            J["논문을 넘음"] = "아니오 (README 규칙: 시드 평균 > 84.24 이고 논문 재현 대비 '개선')"
    L += ["", "## 판정 (README '판정 기준')", ""] + [f"- {k}: {v}" for k, v in J.items()]

    D = {}
    for k in order:
        if k not in test:
            continue
        per = [decompose(os.path.join(r["dir"], "eval_test", "preds.npz")) for r in test[k] if os.path.exists(os.path.join(r["dir"], "eval_test", "preds.npz"))]
        if not per:
            continue
        D[k] = {s: {m: float(np.mean([p[s][m] for p in per if p[s][m] is not None])) if any(p[s][m] is not None for p in per) else None
                    for m in per[0][s]} for s in per[0] if s != "fp_per_image"}
        D[k]["fp_per_image"] = float(np.mean([p["fp_per_image"] for p in per]))
    if D:
        L += ["", "## 표 3. 오차 분해 — test 정답 박스 단위, 점수 ≥ 0.5 예측, 시드 평균", "",
              "IoU 문턱마다 찾은 비율 (%) · 가장 잘 맞는 예측과의 IoU 중앙값. '경계에 걸림' = 화면 가장자리 1 px 안에 닿은 박스.", "",
              "| 설정 | 부분 | 정답 수 | 찾음 @0.5 | @0.75 | @0.9 | IoU 중앙값 |", "|---|---|---|---|---|---|---|"]
        for k, d in D.items():
            for s in ("전체", "낮", "밤", "small", "medium", "경계에 걸림", "경계 안쪽"):
                q = d[s]
                if not q["n_gt"]:
                    continue
                L.append(f"| {label.get(k, k)} | {s} | {q['n_gt']:.0f} | " + " | ".join(
                    "—" if q[f"recall_{t}"] is None else f"{100 * q[f'recall_{t}']:.1f}" for t in (0.5, 0.75, 0.9))
                    + f" | {q['best_iou_median']:.3f} |")
            L.append(f"| {label.get(k, k)} | 잘못 찾은 수 / 장 | | {d['fp_per_image']:.3f} | | | |")
        res["decompose"] = D
    open(os.path.join(a.out, "tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    for k, g in res["groups"].items():                               # 화면 (이슈 댓글 끝부분) 에 표 1 요약 — 이름과 숫자만
        pc = lambda s: "—" if s is None else (f"{s['mean']:.2f}±{s['std']:.2f}" if s["n"] > 1 else f"{s['mean']:.2f}")
        print(f"[표1] {k} (시드 {len(g['seeds'])}) · mAP {pc(g['all.map'])} · mAP50 {pc(g['all.map_50'])} · mAP75 {pc(g['all.map_75'])} · "
              f"낮 {pc(g['day.map'])} · 밤 {pc(g['night.map'])} · {g['params']['detector_m']} M" + (f" · {g['latency_ms']} ms" if g["latency_ms"] else ""))
    for k, v in J.items():
        print(f"[판정] {k}: {v}")
    print(f"[report] 표 → {os.path.join(a.out, 'tables.md')} (실행 {len(runs)} 개)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["select", "summary"])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit(select(a) if a.what == "select" else summary(a))
