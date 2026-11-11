"""CPU 부품 시험 — bash tests/run_local.sh 가 환경 변수 (TRI_CODE · TRI_MIT) 를 정하고 부른다.

1. 사전학습 MiT-B1 이름 바꾸기: 저자 MiT 의 RGB 줄기 출력이 transformers SegformerModel 과 같은가 (네 stage)
2. 보조 줄기 초기화: 블록 가중치는 RGB 와 같고, 첫 합성곱만 채널 평균
3. 게이트 주입: 게이트 0 이면 base 와 출력이 똑같고, 0 이 아니면 달라지며, 기울기는 proj · gate 에만 (DINOv2 는 고정)
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
tiny = {"hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 2, "mlp_ratio": 2, "patch_size": 14, "image_size": 518}
torch.manual_seed(0)
base = M.build("base")
torch.manual_seed(0)
inj = M.build("inject", dino_cfg=tiny)
r = inj.backbone.base.load_state_dict(base.backbone.state_dict())
ok(not r.missing_keys and not r.unexpected_keys, "inject 의 base 부분 = base 백본 (이름 같음)")
base.eval(); inj.eval()
img = torch.randn(1, 5, 224, 288)
with torch.no_grad():
    fb, fi = base.backbone(img), inj.backbone(img)
ok(all(torch.equal(fb[k], fi[k]) for k in fb) and list(fb) == list(fi), "게이트 0: inject 출력 = base 출력 (다섯 층 모두)")
with torch.no_grad():
    inj.backbone.gate.fill_(0.1)
    fi2 = inj.backbone(img)
ok(not torch.equal(fb["0"], fi2["0"]), "게이트 0.1: 출력이 달라짐")
inj.train()
out = inj.backbone(img)
sum(v.float().mean() for v in out.values()).backward()
g = {n: p.grad for n, p in inj.backbone.named_parameters() if p.grad is not None}
ok(any(k.startswith("proj.") for k in g) and "gate" in g, "기울기가 proj · gate 에 감")
ok(not any(k.startswith("dino.") for k in g) and not inj.backbone.dino.training, "DINOv2 는 기울기 없음 · 평가 모드 유지")
sd = M.save_state(inj)
ok(not any(k.startswith(M.FROZEN) for k in sd), "저장 상태에 고정 DINOv2 가중치 없음")

# 4 ────────────────────────────────────────────────────────────────
S = lambda mean, lo, hi, n=3: {"n": n, "mean": mean, "min": lo, "max": hi}
ok(judge(S(85.0, 84.8, 85.3), S(84.0, 83.7, 84.2))[0] == "개선", "판정: Δ 1.0 > 폭 0.5 → 개선")
ok(judge(S(84.3, 83.9, 84.6), S(84.0, 83.7, 84.2))[0] == "약한 개선", "판정: Δ 0.3 ≤ 폭 0.7 → 약한 개선")
ok(judge(S(83.9, 83.5, 84.2), S(84.0, 83.7, 84.2))[0] == "개선 없음", "판정: Δ < 0 → 개선 없음")
ok(judge(S(85.0, 85.0, 85.0, 1), S(84.0, 83.7, 84.2))[0].startswith("판정 보류"), "판정: 시드 1 개 → 보류")

print(f"\n부품 시험: 실패 {len(fails)} 개")
sys.exit(1 if fails else 0)
