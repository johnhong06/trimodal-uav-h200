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
NAME = re.compile(r"^(author|paper|base|fuse|edge|both)_(sgd|adamw)_(full|sub)_s(\d+)$")
OURS = ("fuse", "edge", "both")                 # 우리 모듈: A 융합 보정 · B 경계 보정 · A + B
ARM_LABEL = {"base": "기본 설정 보강", "fuse": "A 융합 보정", "edge": "B 경계 보정", "both": "A + B"}
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


def label(k):
    arm, opt = k.split("_")[:2]
    if arm in ARM_LABEL:
        return f"{ARM_LABEL[arm]} ({'SGD' if opt == 'sgd' else 'AdamW'})"
    return {"author": "저자 스크립트 그대로", "paper": "논문 설정 재현 (우리 학습 루프)"}[arm]


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
    arm_of = lambda k: k.split("_")[0]
    order = ["author_sgd_full", "paper_sgd_full"] + [k for a_ in ("base",) + OURS for k in sorted(test) if arm_of(k) == a_]
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
        L.append(f"| {label(k)} | {len(rs)} | " + " | ".join(fmt(st[f'{p}.{kk}']) for p, kk in COLS) + f" | {prm['detector_m']:.2f} | {lat[0] if lat else '—'} |")

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
    mean_of = lambda k: (gstat(test[k]) or {"mean": float("-inf")})["mean"] if k in test else float("-inf")
    for k in order:
        arm = arm_of(k)
        if arm not in ("base",) + OURS or k not in test:
            continue
        xs = gstat(test[k])
        if xs is None:
            continue
        same = lambda a_: f"{a_}_{k.split('_')[1]}_full"             # 같은 최적화의 다른 팔
        refs = [(same("base"), "기본 설정 보강")] if arm in OURS else []
        if arm == "both":                                          # 합칠 가치: 둘 중 나은 하나보다도 개선인가
            one = max((same("fuse"), same("edge")), key=mean_of)
            refs.append((one, f"둘 중 나은 하나 ({label(one)})"))
        refs.append(("paper_sgd_full", "논문 재현"))
        for rk, rl in refs:
            if rk in test:
                v, d, sp = judge(xs, gstat(test[rk]))
                J[f"{label(k)} vs {rl}"] = v + ("" if d is None else f" (Δ {d:+.2f}, 시드 폭 {sp:.2f})")
        J[f"{label(k)} vs 논문 보고값"] = f"평균 {xs['mean']:.2f} vs {PAPER_MAP} ({xs['mean'] - PAPER_MAP:+.2f})"
        b = test.get(same("base"))
        for want, (p1, k1, n1), (p2, k2, n2) in ((("fuse", "both"), ("night", "map", "밤"), ("day", "map", "낮")),      # 결과 전에 적어 둔 예상 (README 실험 3)
                                                 (("edge", "both"), ("all", "map_75", "mAP75"), ("all", "map_50", "mAP50"))):
            if arm in want and b:
                d1 = [gstat(z, p1, k1) for z in (test[k], b)]
                d2 = [gstat(z, p2, k2) for z in (test[k], b)]
                if all(d1) and all(d2):
                    e1, e2 = d1[0]["mean"] - d1[1]["mean"], d2[0]["mean"] - d2[1]["mean"]
                    J[f"{label(k)} 예상 ({n1} Δ > {n2} Δ, 기본 설정 보강 대비)"] = f"{'맞음' if e1 > e2 else '아님'} ({n1} {e1:+.2f}, {n2} {e2:+.2f})"
    # 논문을 넘음: A + B → A · B 중 평균이 높은 것 → 기본 설정 보강 순서로 같은 규칙을 적용해 처음 넘는 것
    cands = ([g for g in order if arm_of(g) == "both"] + sorted((g for g in order if arm_of(g) in ("fuse", "edge")), key=mean_of, reverse=True)
             + [g for g in order if arm_of(g) == "base"])
    for k in cands:
        xs = gstat(test[k]) if k in test else None
        if xs is None or not P:
            continue
        v, _, _ = judge(xs, gstat(P))
        if xs["mean"] > PAPER_MAP and v == "개선":
            J["논문을 넘음"] = f"예 — {label(k)} (평균 {xs['mean']:.2f} > {PAPER_MAP}, 논문 재현 대비 개선)"
            break
    else:
        if P and any(arm_of(g) in ("base",) + OURS for g in test):
            J["논문을 넘음"] = "아니오 (README 규칙: 시드 평균 > 84.24 이고 논문 재현 대비 '개선')"
    L += ["", "## 판정 (README '판정 기준')", ""] + [f"- {k}: {v}" for k, v in J.items()]

    NV = {k: test[k] for k in order if k in test and all("all_novote" in r["m"] for r in test[k])}
    if NV:
        L += ["", "## 표 4. B 의 투표를 끄고 잰 값 — 같은 체크포인트, test (%, 시드 평균 ± 표준편차)", "",
              "투표를 꺼도 오른 몫은 경계 맵 보조 손실이 특징을 바꾼 효과, 켰을 때 더 오른 몫이 박스 변 보정의 효과다. g = 학습된 보정 세기.", "",
              "| 설정 | mAP (투표 켬) | mAP (투표 끔) | mAP75 (켬) | mAP75 (끔) | g |", "|---|---|---|---|---|---|"]
        res["novote"] = {}
        for k, rs in NV.items():
            g_ = stat([r["m"]["edge_gate"] for r in rs])
            res["novote"][k] = {"on": gstat(rs), "off": gstat(rs, "all_novote"), "gate": g_}
            L.append(f"| {label(k)} | {fmt(gstat(rs))} | {fmt(gstat(rs, 'all_novote'))} | {fmt(gstat(rs, 'all', 'map_75'))} | "
                     f"{fmt(gstat(rs, 'all_novote', 'map_75'))} | {fmt(g_, 3)} |")

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
                L.append(f"| {label(k)} | {s} | {q['n_gt']:.0f} | " + " | ".join(
                    "—" if q[f"recall_{t}"] is None else f"{100 * q[f'recall_{t}']:.1f}" for t in (0.5, 0.75, 0.9))
                    + f" | {q['best_iou_median']:.3f} |")
            L.append(f"| {label(k)} | 잘못 찾은 수 / 장 | | {d['fp_per_image']:.3f} | | | |")
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
