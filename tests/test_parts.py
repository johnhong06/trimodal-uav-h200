"""CPU 부품 시험 — bash tests/run_local.sh 가 환경 변수 (TRI_CODE · TRI_MIT) 를 정하고 부른다.

1. 사전학습 MiT-B1 이름 바꾸기: 저자 MiT 의 RGB 줄기 출력이 transformers SegformerModel 과 같은가 (네 stage)
2. 보조 줄기 초기화: 블록 가중치는 RGB 와 같고, 첫 합성곱만 채널 평균
3. 우리 모듈: A (융합 보정) 는 처음에 base 와 출력이 똑같고, B (경계 보정) 는 g = 0 에서 검출이 base 와 똑같다.
   투표는 정답 경계 맵이면 정답 변을 돌려주고, 화면 끝에 닿은 변은 건드리지 않는다. 손실 · 기울기 · 최적화 묶음 · 파라미터 수
4. 판정 규칙 (src/report.py judge)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import torch  # noqa: E402

import model as M  # noqa: E402
from common import author  # noqa: E402
from report import judge  # noqa: E402

torch.set_num_threads(min(6, torch.get_num_threads()))
fails = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


# 1 · 2 ────────────────────────────────────────────────────────────
t = author()
enc = t.models.encoder.get_encoder("mit_b1", in_chans_rgb=3, in_chans_x=2)
n = M.load_mit(enc, os.environ["TRI_MIT"])
enc.eval()
from transformers import SegformerModel  # noqa: E402

hf = SegformerModel.from_pretrained(os.path.dirname(os.environ["TRI_MIT"])).eval()
x = torch.randn(1, 3, 224, 288)
with torch.no_grad():
    ref = hf(pixel_values=x, output_hidden_states=True).hidden_states
    y, outs = x, []
    for i in range(1, 5):
        y, h, w = getattr(enc, f"patch_embed{i}")(y)
        for b in getattr(enc, f"block{i}"):
            y = b(y, h, w)
        y = getattr(enc, f"norm{i}")(y).reshape(1, h, w, -1).permute(0, 3, 1, 2)
        outs.append(y)
for i, (a, b) in enumerate(zip(outs, ref)):
    rel = (a - b).abs().max().item() / b.abs().max().item()
    ok(a.shape == b.shape and rel < 1e-3, f"MiT stage {i + 1} 출력이 HF 와 같음 (상대 최대차 {rel:.2e}, 모양 {tuple(a.shape)})")
rgb_keys = [k for k in enc.state_dict() if k.startswith(("patch_embed", "block", "norm"))]
ok(n == 2 * len(rgb_keys), f"사전학습 텐서 {n} 개 = RGB 줄기 {len(rgb_keys)} 개 × 2 줄기 (빠짐 없음)")
ok(torch.equal(enc.block3[1].attn.kv.weight, enc.extra_block3[1].attn.kv.weight), "보조 줄기 블록 = RGB 줄기 블록")
w_rgb, w_x = enc.patch_embed1.proj.weight, enc.extra_patch_embed1.proj.weight
ok(w_x.shape[1] == 2 and torch.allclose(w_x[:, 0], w_rgb.mean(1)) and torch.allclose(w_x[:, 1], w_rgb.mean(1)), "보조 줄기 첫 합성곱 = RGB 채널 평균 × 2 채널")

# 3 ────────────────────────────────────────────────────────────────
import re  # noqa: E402

import edge as E  # noqa: E402
import rescale as R  # noqa: E402
import train as T  # noqa: E402

small = lambda m: (setattr(m.transform, "min_size", (256,)), setattr(m.transform, "max_size", 512), m)[2]   # 시험은 작은 크기로 (본 크기는 smoke 가 돈다)
npar = lambda m: sum(p.numel() for p in m.parameters())
torch.manual_seed(0)
base = small(M.build("base"))
img = torch.rand(5, 224, 288) * 255
tgt = {"boxes": torch.tensor([[40.0, 30.0, 120.0, 90.0], [0.0, 100.0, 60.0, 160.0]]), "labels": torch.tensor([1, 1])}

# 3a. A — 융합 보정 (fuse)
fuse = small(M.build("fuse"))
r = fuse.load_state_dict({re.sub(r"(BiTEs\.\d+\.)", r"\1bite.", k): v for k, v in base.state_dict().items()}, strict=False)
ok(not r.unexpected_keys and r.missing_keys and all(".fix." in k for k in r.missing_keys), "fuse = base + 보정 모듈 (나머지 이름 · 모양 같음)")
base.eval(); fuse.eval()
x = torch.randn(1, 5, 224, 288)
with torch.no_grad():
    fb, ff = base.backbone(x), fuse.backbone(x)
ok(list(fb) == list(ff) and all(torch.equal(fb[k], ff[k]) for k in fb), "A 처음 (γ = 1 · β = 0): fuse 출력 = base 출력 (다섯 층 모두)")
fixes = [m for m in fuse.modules() if isinstance(m, R.Rescale)]
with torch.no_grad():
    for m in fixes:
        for c in list(m.gamma) + list(m.beta):
            c.weight.normal_(std=0.5)
    ff2 = fuse.backbone(x)
    g1, b1, g2, b2 = fixes[0].maps(torch.randn(1, 64, 56, 72), torch.randn(1, 64, 56, 72))
ok(len(fixes) == 4 and not torch.equal(fb["0"], ff2["0"]), "A 머리를 0 에서 벗어나게 하면 출력이 달라짐 (네 stage)")
ok(g1.shape == (1, 1, 56, 72) and b1.shape == (1, 64, 56, 72) and 0 < g1.min() and g1.max() < 2 and 0 < g2.min() and g2.max() < 2, "A 맵: γ 는 픽셀별 1 채널 (0–2), β 는 C 채널")
fuse.train()
sum(v.mean() for v in fuse.backbone(x).values()).backward()
gr = [p.grad for n, p in fuse.named_parameters() if ".fix." in n]
ok(all(g is not None for g in gr) and all(m.inp[0].weight.grad.abs().sum() > 0 and m.attn[1].q.weight.grad.abs().sum() > 0 for m in fixes), "A 기울기가 보정 모듈 (입력 conv · 어텐션) 까지 감")
d_fuse = npar(fuse) - npar(base)
ours = lambda opts: {id(p) for g in opts[1].param_groups for p in g["params"]}
main = lambda opts: {id(p) for g in opts[0].param_groups for p in g["params"]}
opts = T.optimizers(fuse, "sgd")
ok(0 < d_fuse < 1e6 and len(opts) == 2 and ours(opts) == {id(p) for n, p in fuse.named_parameters() if ".fix." in n} and not (ours(opts) & main(opts))
   and len(main(opts)) + len(ours(opts)) == len(list(fuse.parameters())) and opts[1].defaults["lr"] == 1e-3 and len(T.optimizers(base, "sgd")) == 1,
   f"A 파라미터 +{d_fuse / 1e6:.3f} M (1 M 미만), 보정 모듈만 따로 AdamW 1e-3 · 나머지는 본 최적화 (base 는 최적화 하나)")

# 3b. B — 경계 보정 (edge)
edge = small(M.build("edge"))
r = edge.load_state_dict(base.state_dict(), strict=False)
ok(not r.unexpected_keys and r.missing_keys and all(k.startswith("roi_heads.edge.") for k in r.missing_keys), "edge = base + 경계 맵 (나머지 이름 · 모양 같음)")
hb, he = base.roi_heads, edge.roi_heads
ok((hb.proposal_matcher.high_threshold, hb.proposal_matcher.low_threshold, hb.fg_bg_sampler.batch_size_per_image, hb.fg_bg_sampler.positive_fraction,
    hb.box_coder.weights, hb.score_thresh, hb.nms_thresh, hb.detections_per_img)
   == (he.proposal_matcher.high_threshold, he.proposal_matcher.low_threshold, he.fg_bg_sampler.batch_size_per_image, he.fg_bg_sampler.positive_fraction,
       he.box_coder.weights, he.score_thresh, he.nms_thresh, he.detections_per_img), "B 박스 헤드 설정값 (표본 · NMS · 문턱) 이 base 와 같음")
edge.eval()
with torch.no_grad():
    ob, oe = base([img])[0], edge([img])[0]
    edge.roi_heads.edge.gate.fill_(1.0)
    o1 = edge([img])[0]
    edge.roi_heads.refine = False
    o1off = edge([img])[0]
    edge.roi_heads.refine = True
    edge.roi_heads.edge.gate.fill_(0.0)
ok(len(ob["boxes"]) > 0 and torch.equal(ob["boxes"], oe["boxes"]) and torch.equal(ob["scores"], oe["scores"]), f"B 처음 (g = 0): edge 검출 = base 검출 (박스 {len(ob['boxes'])} 개)")
ok(not torch.equal(o1["boxes"], ob["boxes"]) and torch.equal(o1["scores"], ob["scores"]) and torch.equal(o1off["boxes"], ob["boxes"])
   and bool(((o1["boxes"][:, 2] - o1["boxes"][:, 0]) >= 1).all()) and bool(((o1["boxes"][:, 3] - o1["boxes"][:, 1]) >= 1).all()),
   "B g = 1: 박스만 달라지고 점수는 그대로, 뒤집힌 박스 없음, refine = False 면 base 와 같음")

# 투표: 정답과 똑같은 경계 맵을 주면 어긋난 박스가 정답 변으로 간다
f = he.edge
H, W, shape = 40, 60, (160, 240)
X, Y = E.centers(H, W, "cpu")


def field(boxes):                                                  # 픽셀마다 '자기를 담은 가장 작은 박스' 의 변 좌표 · 신뢰도 (안 +10, 밖 −10)
    side, conf, best = torch.zeros(4, H, W), torch.full((H, W), -10.0), torch.full((H, W), float("inf"))
    for b in boxes:
        ins = (X > b[0]) & (X < b[2]) & (Y > b[1]) & (Y < b[3]) & ((b[2] - b[0]) * (b[3] - b[1]) < best)
        for c in range(4):
            side[c][ins] = b[c]
        conf[ins], best[ins] = 10.0, (b[2] - b[0]) * (b[3] - b[1])
    return torch.cat([side, conf[None]])


gt = torch.tensor([[50.3, 40.7, 150.2, 110.9], [0.0, 60.0, 30.0, 120.0]])
off = torch.tensor([[55.0, 36.0, 144.0, 116.0], [0.0, 64.0, 34.0, 115.0]])
fm = field(gt)
with torch.no_grad():
    f.gate.fill_(0.0); v0 = f.vote(fm, off, shape)
    f.gate.fill_(0.5); vh = f.vote(fm, off, shape)
    f.gate.fill_(1.0); v1 = f.vote(fm, off, shape)
    far = f.vote(fm, torch.tensor([[200.0, 10.0, 230.0, 30.0]]), shape)      # 신뢰도가 0 인 곳의 박스
    lm, lc = f.dense_losses(fm[None], [{"boxes": gt}])
    nest = torch.tensor([[20.0, 20.0, 200.0, 140.0], [60.0, 50.0, 100.0, 90.0]])
    lm2, _ = f.dense_losses(field(nest)[None], [{"boxes": nest}])
    lm3, _ = f.dense_losses(fm[None], [{"boxes": gt + torch.tensor([6.0, 0.0, 6.0, 0.0])}])
    f.gate.fill_(0.0)
ok(torch.equal(v0, off), "투표 g = 0: 박스 그대로")
ok((v1 - gt).abs().max() < 1e-2 and (vh - (off + gt) / 2).abs().max() < 1e-2, f"투표 g = 1 · 신뢰도 1: 정답 변으로 (최대 오차 {(v1 - gt).abs().max():.1e} px), g = 0.5: 절반")
ok(v1[1, 0] == 0.0 and abs(float(v1[1, 2]) - 30.0) < 1e-2, "투표: 화면 끝에 닿은 변은 그대로, 나머지 변만 고침")
ok((far - torch.tensor([[200.0, 10.0, 230.0, 30.0]])).abs().max() < 0.05, f"투표: 신뢰도가 0 인 곳의 박스는 안 움직임 (g = 1 에서 {(far - torch.tensor([[200.0, 10.0, 230.0, 30.0]])).abs().max():.1e} px)")
ok(lm < 1e-4 and lc < 1e-3 and lm2 < 1e-4 and lm3 > 0.01, f"경계 맵 손실: 정답 맵이면 0 (겹치면 작은 박스 기준), 6 px 어긋나면 {lm3:.3f}")

edge.train()
ld = edge([img], [tgt])
sum(ld.values()).backward()
ok(set(ld) == set(T.LOSSES + T.EDGE_LOSSES) and all(torch.isfinite(v) for v in ld.values()), "B 학습 손실 7 개 (Faster R-CNN 4 + 경계 3), 모두 유한")
ok(f.gate.grad is not None and f.gate.grad.abs() > 0 and f.inp.weight.grad.abs().sum() > 0 and f.blocks[1].col.qkv.weight.grad.abs().sum() > 0
   and f.point.q.weight.grad.abs().sum() > 0 and f.point.k.weight.grad.abs().sum() > 0, "B 기울기가 g · 경계 맵 (축 어텐션 · 가리키는 머리) 까지 감")
opts = T.optimizers(edge, "sgd")
nowd = {id(p) for p in opts[1].param_groups[1]["params"]}
ok(len(opts) == 2 and ours(opts) == {id(p) for p in f.parameters()} and not (ours(opts) & main(opts)) and opts[1].defaults["lr"] == 1e-3
   and id(f.gate) in nowd and opts[1].param_groups[1]["weight_decay"] == 0.0 and opts[1].param_groups[0]["weight_decay"] == 0.05,
   "B 최적화: 경계 맵만 따로 AdamW 1e-3 (g · 편향 · 정규화는 weight decay 0), 나머지는 본 최적화")
with torch.no_grad():                                               # 가리키는 머리: 처음엔 줄 전체를 고르게 봐서 줄 가운데를 가리킨다 → 좌표 범위 안
    pm = f(torch.randn(1, 256, 12, 20))
ok(pm.shape == (1, 5, 12, 20) and 0 < pm[:, [0, 2]].min() and pm[:, [0, 2]].max() < 20 * E.STRIDE and 0 < pm[:, [1, 3]].min() and pm[:, [1, 3]].max() < 12 * E.STRIDE,
   "경계 맵 모양 (B, 5, H, W) · 변 좌표가 사진 범위 안")
d_edge = npar(edge) - npar(base)
ok(0 < d_edge < 1e6, f"B 파라미터 +{d_edge / 1e6:.3f} M (1 M 미만)")

# 3c. A + B (both)
both = small(M.build("both"))
ok(set(both.state_dict()) == set(fuse.state_dict()) | {k for k in edge.state_dict() if k.startswith("roi_heads.edge.")}
   and npar(both) - npar(base) == d_fuse + d_edge, "both = fuse + 경계 맵 (모듈 둘 다, 파라미터 합 같음)")
both.train()
ld = both([img], [tgt])
sum(ld.values()).backward()
ok(all(torch.isfinite(v) for v in ld.values()) and len(ld) == 7, "both 학습 한 스텝 (손실 7 개 유한)")

# 4 ────────────────────────────────────────────────────────────────
S = lambda mean, lo, hi, n=3: {"n": n, "mean": mean, "min": lo, "max": hi}
ok(judge(S(85.0, 84.8, 85.3), S(84.0, 83.7, 84.2))[0] == "개선", "판정: Δ 1.0 > 폭 0.5 → 개선")
ok(judge(S(84.3, 83.9, 84.6), S(84.0, 83.7, 84.2))[0] == "약한 개선", "판정: Δ 0.3 ≤ 폭 0.7 → 약한 개선")
ok(judge(S(83.9, 83.5, 84.2), S(84.0, 83.7, 84.2))[0] == "개선 없음", "판정: Δ < 0 → 개선 없음")
ok(judge(S(85.0, 85.0, 85.0, 1), S(84.0, 83.7, 84.2))[0].startswith("판정 보류"), "판정: 시드 1 개 → 보류")

print(f"\n부품 시험: 실패 {len(fails)} 개")
sys.exit(1 if fails else 0)
