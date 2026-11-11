"""학습 — 저자 학습 루프 (trimodaldet/training/trainer.py) 와 같은 순서 · 같은 일정, 팔 (arm) 과 최적화만 바꾼다.

저자 루프 그대로인 것: 손실 = Faster R-CNN 손실 4 개의 합 · zero_grad → backward → step · 500 스텝 선형 warmup 뒤 cosine
(T_max = 전체 스텝 − 500, eta_min 0) · 15 epoch · 배치 16 · AMP 없음. 저자처럼 첫 스텝은 warmup 전 학습률로 간다.
바꾼 것: 시드 고정 (epoch 마다 섞는 순서도 시드로), DataLoader 작업자 4 개, epoch 끝마다 last.pt (이어 하기).

최적화
  sgd    저자 값: SGD lr 0.02 · momentum 0.9 · weight decay 1e-4 (모든 학습 파라미터)
  adamw  AdamW lr 1e-4 · weight decay 0.05 (2 차원 이상 가중치만, 편향 · 정규화는 0) — SegFormer 류 사전학습 백본에 흔한 값
  inject 팔의 주입 모듈 (proj · gate) 은 따로 AdamW (wd 0): proj 1e-4 · gate 1e-3 — depthvlm-silog 해결 실험과 같은 값

결과 (--out): train_log.csv (10 스텝마다) · done.json (끝) · 화면은 epoch 마다 한 줄. 체크포인트 (--ckpt): last.pt · final.pt (고정 DINOv2 제외)
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
from common import seed_all, write_json  # noqa: E402

INJ = ("backbone.proj.", "backbone.gate")


def optimizers(model, opt):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    det = [(n, p) for n, p in named if not n.startswith(INJ)]
    if opt == "sgd":
        o = torch.optim.SGD([p for _, p in det], lr=0.02, momentum=0.9, weight_decay=1e-4)
    else:
        o = torch.optim.AdamW([{"params": [p for _, p in det if p.ndim > 1], "weight_decay": 0.05},
                               {"params": [p for _, p in det if p.ndim <= 1], "weight_decay": 0.0}], lr=1e-4)
    opts = [o]
    proj = [p for n, p in named if n.startswith("backbone.proj.")]
    gate = [p for n, p in named if n == "backbone.gate"]
    if proj:
        opts.append(torch.optim.AdamW([{"params": proj, "lr": 1e-4}, {"params": gate, "lr": 1e-3}], weight_decay=0.0))
    return opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=M.ARMS, required=True)
    ap.add_argument("--opt", choices=["sgd", "adamw"], required=True)
    ap.add_argument("--split", choices=["full", "sub"], default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="학습 장 수 제한 (smoke)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.arm == "paper" and a.opt != "sgd":
        raise SystemExit("!!! paper 팔은 저자 최적화 (sgd) 만")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.ckpt, exist_ok=True)
    name = os.path.basename(os.path.normpath(a.out))
    dev = torch.device(a.device)

    seed_all(a.seed)                                               # 모델 초기화 전에 (저자 Trainer 와 같은 생성 순서)
    ds = D.make(a.split, a.limit)
    model = M.build(a.arm).to(dev)
    opts = optimizers(model, a.opt)
    steps_ep = -(-len(ds) // a.bs)
    total = steps_ep * a.epochs
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(1, total - a.warmup), eta_min=0) for o in opts]
    step, ep0 = 0, 0
    last = os.path.join(a.ckpt, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        M.load_state(model, ck["model"])
        for o, s in zip(opts, ck["opts"]):
            o.load_state_dict(s)
        for sc, s in zip(scheds, ck["scheds"]):
            sc.load_state_dict(s)
        step, ep0 = ck["step"], ck["epoch"]
        torch.set_rng_state(ck["rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print(f"[train {name}] 이어 하기: epoch {ep0} · step {step}", flush=True)
    p = M.count_params(model)
    print(f"[train {name}] {a.arm} · {a.opt} · {a.split} {len(ds):,} 장 · 시드 {a.seed} · {a.epochs} epoch × {steps_ep} 스텝 · "
          f"파라미터 {p['detector_m']} M (고정 인코더 {p['frozen_encoder_m']} M)", flush=True)

    logf = os.path.join(a.out, "train_log.csv")
    new = not (a.resume and os.path.exists(logf) and step > 0)
    fh = open(logf, "w" if new else "a", newline="")
    log = csv.writer(fh)
    if new:
        log.writerow(["epoch", "step", "loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "lr", "sec"])
    t_start = time.time()
    for ep in range(ep0, a.epochs):
        model.train()
        g = torch.Generator()
        g.manual_seed(a.seed * 1000 + ep)                          # 이어 해도 같은 순서
        loader = DataLoader(ds, batch_size=a.bs, shuffle=True, generator=g, num_workers=a.workers,
                            collate_fn=D.collate, pin_memory=dev.type == "cuda")
        t0, run, n = time.time(), 0.0, 0
        for imgs, tgts in loader:
            imgs = [x.to(dev, non_blocking=True) for x in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            ld = model(imgs, tgts)
            loss = sum(ld.values())
            if not torch.isfinite(loss):
                write_json(os.path.join(a.out, "fail.json"), {"epoch": ep, "step": step, "why": "loss 가 유한하지 않음"})
                print(f"!!! [train {name}] epoch {ep + 1} step {step}: loss 가 유한하지 않음 — 멈춤", flush=True)
                return 3
            for o in opts:
                o.zero_grad()
            loss.backward()
            for o in opts:
                o.step()
            step += 1
            if step <= a.warmup:                                   # 저자: lr = 기본 lr × min(1, step / warmup)
                for o in opts:
                    for pg in o.param_groups:
                        pg["lr"] = pg["initial_lr"] * min(1.0, step / a.warmup)
            else:
                for sc in scheds:
                    sc.step()
            run += loss.item()
            n += 1
            if step % 10 == 0:
                log.writerow([ep + 1, step, round(loss.item(), 5)] + [round(ld[k].item(), 5) for k in
                             ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")]
                             + [f"{opts[0].param_groups[0]['lr']:.3g}", round(time.time() - t_start, 1)])
                fh.flush()
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        print(f"[train {name}] epoch {ep + 1}/{a.epochs} · loss {run / max(n, 1):.4f} · {(time.time() - t0) / 60:.1f} 분 · GPU 최대 {mem:.1f} GiB", flush=True)
        torch.save({"model": M.save_state(model), "opts": [o.state_dict() for o in opts], "scheds": [s.state_dict() for s in scheds],
                    "step": step, "epoch": ep + 1, "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, last + ".tmp")
        os.replace(last + ".tmp", last)
    fh.close()
    torch.save(M.save_state(model), os.path.join(a.ckpt, "final.pt"))
    write_json(os.path.join(a.out, "done.json"), {
        "arm": a.arm, "opt": a.opt, "split": a.split, "seed": a.seed, "epochs": a.epochs, "images": len(ds), "steps": step,
        "minutes": round((time.time() - t_start) / 60, 1), "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None,
        "params": M.count_params(model), "gate": [round(float(x), 5) for x in model.backbone.gate] if a.arm == "inject" else None})
    print(f"[train {name}] 끝 — {step} 스텝", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
