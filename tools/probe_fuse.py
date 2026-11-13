"""학습된 체크포인트로 융합이 무엇에 기대는지 · A (융합 보정) 가 무엇을 배웠는지 본다 — 학습 없음, CPU 로도 돈다.

  1. γ · β 통계 (fuse · both): stage 마다 γ1 (RGB 줄기) · γ2 (열화상 + 이벤트 줄기) 의 평균 · 퍼짐, ‖β‖ / ‖x‖ — 낮 · 밤, 박스 안 · 밖
  2. BiTE 가 줄기의 크기에 얼마나 민감한가: 한 줄기에 0.5 · 0 을 곱해 넣었을 때 BiTE 출력이 바뀌는 비율 (앞의 사진 몇 장)
  3. 조건을 바꿔 잰 AP (같은 체크포인트):
       clean                     그대로
       norgb · nothermal · noevent   그 채널을 정규화 평균 값으로 채움 (= 정규화 뒤 0, 모달을 지운 입력)
       gamma1 · beta0 · bypass   A 의 γ 를 1 로 · β 를 0 으로 · 둘 다 (= 보정 없음)              ← fuse · both 만
       rgbx<s> · tex<s>          BiTE 에 들어가는 RGB 줄기 · 보조 줄기에 s 를 곱함 (A 가 있으면 보정한 뒤에)
AP 는 tools/error_analysis.py 와 같은 계산 (COCO 방식, 직접 계산 — torchmetrics 값과 일치 확인한 것).

  TRI_CODE=<저자 코드> TRI_DATA=<images · labels 폴더> python tools/probe_fuse.py --arm fuse --ckpt <.pt> --out <json>
      [--every 5] [--conds clean,gamma1,…] [--ref <H200 의 preds.npz>] [--threads 6]
    --every k  test 1,950 장에서 k 장마다 한 장 (저자 분할이 섞은 순서라 고른 표본)
    --ref      같은 체크포인트를 H200 에서 평가한 preds.npz — 같은 사진들로 AP 를 재서 clean 과 견준다
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
from common import is_day  # noqa: E402

THS = np.round(np.arange(0.5, 0.96, 0.05), 2)
DROP = {"norgb": (0, 1, 2), "nothermal": (3,), "noevent": (4,)}
ST = {"gamma": "", "beta": "", "s1": 1.0, "s2": 1.0, "rec": None, "sens": None, "boxes": None, "size": None}


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


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
            mat = iou(b, g) if len(b) and len(g) else np.zeros((len(b), len(g)))
            for j in range(len(b)):
                m = np.where(used, -1.0, mat[j]) if len(g) else np.array([])
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


def score(stems, dets, gts):
    """→ {"all" | "day" | "night": {map, ap50, ap75, ap90, ap95, n}} (%)."""
    out = {}
    for k, pick in (("all", lambda s: True), ("day", is_day), ("night", lambda s: not is_day(s))):
        ii = [i for i, s in enumerate(stems) if pick(s)]
        a = 100 * ap_table([dets[i] for i in ii], [gts[i] for i in ii]) if ii else np.full(len(THS), np.nan)
        out[k] = {"map": round(float(a.mean()), 2), "ap50": round(float(a[0]), 2), "ap75": round(float(a[5]), 2),
                  "ap90": round(float(a[8]), 2), "ap95": round(float(a[9]), 2), "n": len(ii)}
    return out


def box_mask(boxes, hf, wf, size):
    """정답 박스 (원본 좌표) → stage 격자의 박스 안 칸 · 사진이 있는 칸 (덧댄 자리 제외). size = 모델이 키운 사진 크기 (h, w)."""
    hr, wr = size
    stride = 2 ** round(np.log2(max(hr / hf, wr / wf)))            # 덧댄 크기 / 격자 = 4 · 8 · 16 · 32
    ys, xs = (torch.arange(hf) + 0.5) * stride, (torch.arange(wf) + 0.5) * stride
    valid = (ys[:, None] < hr) & (xs[None, :] < wr)
    inside = torch.zeros(hf, wf, dtype=torch.bool)
    for x1, y1, x2, y2 in (boxes * (hr / D_H)).tolist():
        inside |= (ys[:, None] >= y1) & (ys[:, None] <= y2) & (xs[None, :] >= x1) & (xs[None, :] <= x2)
    return inside & valid, valid & ~inside


D_H = 301                                                          # 원본 사진 높이 (src/common.py H)


def wrap(enc, has_fix):
    """BiTE 넷의 forward 를 바꿔 끼운다 — ST 에 따라 γ · β 를 끄거나 줄기에 배수를 곱하고, 통계를 모은다."""
    for i, m in enumerate(enc.BiTEs):
        bite = m.bite if has_fix else m.forward                    # base 는 원래 forward (바꿔 끼우기 전에 잡아 둠)
        fix = m.fix if has_fix else None

        def fwd(x1, x2, i=i, bite=bite, fix=fix):
            if fix is not None:
                g1, b1, g2, b2 = fix.maps(x1, x2)
                if ST["rec"] is not None:
                    inside, outside = box_mask(ST["boxes"], *x1.shape[-2:], ST["size"])
                    r = {"g1": g1[0, 0], "g2": g2[0, 0]}
                    row = {"stage": i}
                    for k, g in r.items():
                        v = g[inside | outside]
                        row |= {k + "_mean": float(v.mean()), k + "_std": float(v.std()), k + "_min": float(v.min()), k + "_max": float(v.max()),
                                k + "_in": float(g[inside].mean()) if inside.any() else None, k + "_out": float(g[outside].mean())}
                    row |= {"b1_ratio": float(b1.norm() / x1.norm()), "b2_ratio": float(b2.norm() / x2.norm())}
                    ST["rec"].append(row)
                if ST["gamma"] == "one":
                    g1 = g2 = 1.0
                if ST["beta"] == "zero":
                    b1 = b2 = 0.0
                x1, x2 = g1 * x1 + b1, g2 * x2 + b2
            if ST["sens"] is not None:                             # 줄기 크기에 대한 BiTE 출력의 민감도
                ref = bite(x1, x2)
                n = ref.norm()
                ST["sens"].append({"stage": i} | {k: float((bite(a * x1, b * x2) - ref).norm() / n)
                                                  for k, (a, b) in {"rgbx0.5": (0.5, 1), "tex0.5": (1, 0.5), "rgbx2": (2, 1), "rgbx0": (0, 1), "tex0": (1, 0)}.items()})
            return bite(ST["s1"] * x1, ST["s2"] * x2)

        m.forward = fwd


def set_cond(c):
    ST.update(gamma="", beta="", s1=1.0, s2=1.0)
    if c in ("gamma1", "bypass"):
        ST["gamma"] = "one"
    if c in ("beta0", "bypass"):
        ST["beta"] = "zero"
    if c.startswith("rgbx"):
        ST["s1"] = float(c[4:])
    if c.startswith("tex"):
        ST["s2"] = float(c[3:])
    return DROP.get(c, ())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=M.ARMS, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conds", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--sens", type=int, default=40, help="민감도를 재는 사진 수 (clean 의 앞에서부터)")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    has_fix = a.arm in ("fuse", "both")
    conds = a.conds.split(",") if a.conds else ["clean", "norgb", "nothermal", "noevent"] + (["gamma1", "beta0", "bypass"] if has_fix else []) + ["rgbx0.5", "tex0.5"]
    bad = [c for c in conds if not (c == "clean" or c in DROP or c.startswith(("rgbx", "tex")) or (has_fix and c in ("gamma1", "beta0", "bypass")))]
    assert not bad, bad

    os.environ.setdefault("TRI_MIT", "-")
    M.load_mit = lambda enc, path: 0                               # 가중치는 체크포인트에서 온다 — 사전학습 파일은 필요 없음
    model = M.build(a.arm)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    model.eval()
    wrap(model.backbone.encoder, has_fix)
    model.transform.register_forward_hook(lambda mod, inp, out: ST.update(size=tuple(out[0].image_sizes[0])))
    mean = torch.tensor(model.transform.image_mean)

    ds = D.make("test")
    idx = list(range(0, len(ds), a.every))[: a.limit or None]
    stems = [ds.stems[i] for i in idx]
    gts = [ds[i][1]["boxes"] for i in idx]
    gnp = [g.numpy() for g in gts]
    out = {"arm": a.arm, "n": len(idx), "n_day": sum(map(is_day, stems)), "every": a.every, "conds": {}}
    print(f"[probe {a.arm}] test {len(ds):,} 장 가운데 {len(idx)} 장 (낮 {out['n_day']}) · 조건 {len(conds)} 개 · CPU {a.threads} 스레드", flush=True)

    if a.ref:
        p = np.load(a.ref, allow_pickle=False)
        pos = {str(s): i for i, s in enumerate(p["stems"])}
        rd = [(p["det_box"][p["det_img"] == pos[s]], p["det_score"][p["det_img"] == pos[s]]) for s in stems]
        out["ref"] = score(stems, rd, gnp)
        print(f"[probe {a.arm}] H200 예측, 같은 사진: mAP {out['ref']['all']['map']:.2f} · 낮 {out['ref']['day']['map']:.2f} · 밤 {out['ref']['night']['map']:.2f}", flush=True)

    for c in conds:
        drop = set_cond(c)
        first = c == "clean"
        rec = [] if (first and has_fix) else None
        sens = []
        dets, t0 = [], time.time()
        with torch.no_grad():
            for n, i in enumerate(idx):
                x = ds[i][0].clone()
                for ch in drop:
                    x[ch] = mean[ch]
                ST.update(rec=rec, boxes=gts[n], sens=sens if (first and n < a.sens) else None)
                p = model([x])[0]
                dets.append((p["boxes"].numpy(), p["scores"].numpy()))
        ST.update(rec=None, sens=None)
        r = score(stems, dets, gnp)
        r["sec_per_image"] = round((time.time() - t0) / len(idx), 2)
        out["conds"][c] = r
        print(f"[probe {a.arm}] {c:10s} mAP {r['all']['map']:6.2f} · AP50 {r['all']['ap50']:6.2f} · AP75 {r['all']['ap75']:6.2f} · AP90 {r['all']['ap90']:6.2f} · "
              f"낮 {r['day']['map']:6.2f} · 밤 {r['night']['map']:6.2f} · {r['sec_per_image']} 초/장", flush=True)
        if first and sens:
            out["bite_sensitivity"] = {}
            for s in range(4):
                rows = [q for q in sens if q["stage"] == s]
                out["bite_sensitivity"][f"stage{s + 1}"] = {k: round(float(np.mean([q[k] for q in rows])), 4) for k in rows[0] if k != "stage"}
                print(f"[probe {a.arm}]   BiTE 출력 변화 (stage {s + 1}, 사진 {len(rows)} 장): " + " · ".join(f"{k} {v:.3f}" for k, v in out["bite_sensitivity"][f"stage{s + 1}"].items()), flush=True)
        if rec:
            out["gamma"] = {}
            for part, pick in (("day", is_day), ("night", lambda s: not is_day(s))):
                for s in range(4):
                    rows = [q for n, q in enumerate(rec) if q["stage"] == s and pick(stems[n // 4])]
                    avg = lambda k: round(float(np.mean([q[k] for q in rows if q[k] is not None])), 4)
                    d = {k: avg(k) for k in rows[0] if k != "stage" and not k.endswith(("_min", "_max"))}
                    d |= {k: round(float(min(q[k] for q in rows)), 4) for k in rows[0] if k.endswith("_min")}
                    d |= {k: round(float(max(q[k] for q in rows)), 4) for k in rows[0] if k.endswith("_max")}
                    out["gamma"][f"{part}_stage{s + 1}"] = d
                    print(f"[probe {a.arm}]   {part:5s} stage {s + 1}: γ1 {d['g1_mean']:.3f} (사진 안 퍼짐 {d['g1_std']:.3f}, {d['g1_min']:.2f}–{d['g1_max']:.2f}, 박스 안 {d['g1_in']:.3f} · 밖 {d['g1_out']:.3f}) · "
                          f"γ2 {d['g2_mean']:.3f} (퍼짐 {d['g2_std']:.3f}, {d['g2_min']:.2f}–{d['g2_max']:.2f}, 안 {d['g2_in']:.3f} · 밖 {d['g2_out']:.3f}) · "
                          f"‖β‖/‖x‖ {d['b1_ratio']:.3f} · {d['b2_ratio']:.3f}", flush=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
