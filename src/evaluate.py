"""평가 — 저자와 같은 지표 (torchmetrics MeanAveragePrecision, COCO 방식 · pycocotools) 를 전체 · 낮 · 밤으로 따로 잰다.

결과 (--out)
  metrics.json  {"all"|"day"|"night": {map, map_50, map_75, map_small, map_medium, map_large, mar_100}, 장 수, 파라미터, (추론 시간)}
  preds.npz     사진 이름 · 예측 박스 · 점수 · 정답 박스 (원본 301×391 좌표) — src/report.py 의 오차 분해용
--time N: 1 장씩 N 장 추론 시간을 잰다 (워밍업 10 장, cuda 동기화, p50 ms). 다른 프로세스가 GPU 를 같이 쓰지 않을 때만 의미가 있다.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchmetrics.detection import MeanAveragePrecision  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
from common import is_day, write_json  # noqa: E402

KEYS = ("map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=M.ARMS, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(os.path.dirname(os.path.normpath(a.out)))
    dev = torch.device(a.device)
    model = M.build(a.arm)
    M.load_state(model, torch.load(a.ckpt, map_location="cpu", weights_only=True))
    model.to(dev).eval()
    ds = D.make(a.split, a.limit)
    mets = {k: MeanAveragePrecision(iou_type="bbox") for k in ("all", "day", "night")}
    stems, dimg, dbox, dsc, gimg, gbox = [], [], [], [], [], []
    loader = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=4, collate_fn=D.collate)
    i0 = 0
    with torch.no_grad():
        for imgs, tgts in loader:
            preds = model([x.to(dev) for x in imgs])
            for j, (p, t) in enumerate(zip(preds, tgts)):
                s = ds.stems[i0 + j]
                p = {k: v.cpu() for k, v in p.items()}
                t = {"boxes": t["boxes"], "labels": t["labels"]}
                for k in ("all", "day" if is_day(s) else "night"):
                    mets[k].update([p], [t])
                stems.append(s)
                dimg += [len(stems) - 1] * len(p["boxes"])
                dbox.append(p["boxes"].numpy())
                dsc.append(p["scores"].numpy())
                gimg += [len(stems) - 1] * len(t["boxes"])
                gbox.append(t["boxes"].numpy())
            i0 += len(imgs)
    out = {"split": a.split, "n_images": len(stems), "n_day": sum(map(is_day, stems)), "params": M.count_params(model)}
    n_sub = {"all": len(stems), "day": out["n_day"], "night": len(stems) - out["n_day"]}
    for k, m in mets.items():                                      # 사진이 없는 부분 · 박스가 없는 크기 구간은 torchmetrics 가 -1 → None
        r = m.compute() if n_sub[k] else {}
        out[k] = {kk: (round(float(r[kk]), 6) if kk in r and float(r[kk]) >= 0 else None) for kk in KEYS}
    np.savez_compressed(os.path.join(a.out, "preds.npz"), stems=np.array(stems), det_img=np.array(dimg, np.int32),
                        det_box=np.concatenate(dbox).astype(np.float32), det_score=np.concatenate(dsc).astype(np.float32),
                        gt_img=np.array(gimg, np.int32), gt_box=np.concatenate(gbox).astype(np.float32))
    if a.time > 0 and dev.type == "cuda":
        ms = []
        for i in range(min(a.time + 10, len(ds))):
            x = [ds[i][0].to(dev)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x)
            torch.cuda.synchronize()
            if i >= 10:
                ms.append((time.perf_counter() - t0) * 1000)
        out["latency_ms_p50"] = round(float(np.median(ms)), 2) if ms else None
        out["latency_n"] = len(ms)
    write_json(os.path.join(a.out, "metrics.json"), out)
    pc = lambda v: "—" if v is None else f"{100 * v:.2f}"
    al, dy, nt = out["all"], out["day"], out["night"]
    print(f"[eval {name}] {a.split} {len(stems):,} 장 · mAP {pc(al['map'])} · mAP50 {pc(al['map_50'])} · mAP75 {pc(al['map_75'])} · "
          f"낮 {pc(dy['map'])} · 밤 {pc(nt['map'])}" + (f" · {out['latency_ms_p50']} ms" if out.get("latency_ms_p50") else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
