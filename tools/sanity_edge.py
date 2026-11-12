"""B (경계 맵, src/edge.py) 가 변의 위치를 배우는지 보는 작은 합성 실험 — 머리 구조와 모듈의 학습률을 고를 때 썼다 (README 실험 3).
실제 데이터의 결과가 아니다. 저자 코드 · 데이터 · GPU 가 필요 없고 CPU 로 몇 분 걸린다.

  가짜 P0    사각형 (차 대신) 이 그려진 흐린 그림 (192×288) 을 고정 무작위 3×3 conv 로 256 채널로 올리고 4 배 줄인 것. 사각형마다 정답 박스
  검출기 박스  정답의 변마다 ±6 px (균등) 를 더한 것 → 변 오차 평균 3.0 px
  학습       EdgeField 만. 본체 = SGD 0.02 · momentum 0.9 또는 AdamW (lr 를 줌), g 는 AdamW 1e-3. 배치 4 · warmup 100 스텝 뒤 일정한 lr
  잰 것      '투표만' = 흔든 박스의 가운데 영역을 신뢰도로 가중 평균한 변의 오차 (px). g · 평균 신뢰도와 무관하게 경계 맵 자체의 정확도를 본다.
            3.0 px 보다 작아야 투표가 박스를 고칠 수 있다.

  python tools/sanity_edge.py <ptr|reg> <sgd|adamw> [lr=1e-3] [스텝=1400]
    ptr  src/edge.py 그대로: 변이 놓인 칸을 어텐션으로 가리켜 좌표를 읽는 머리
    reg  비교용: 변까지 거리를 숫자로 회귀하는 머리 (exp). 몸통 · 손실 · 투표는 같다
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision.ops import roi_align  # noqa: E402

import edge as E  # noqa: E402
from rescale import pos2d  # noqa: E402

head, opt_name = sys.argv[1], sys.argv[2]
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1400
WARM, H, W, S, JIT = 100, 48, 72, E.STRIDE, 6.0
torch.manual_seed(0)
torch.set_num_threads(min(4, torch.get_num_threads()))


class RegField(E.EdgeField):
    """비교용: 변까지 거리를 숫자로 회귀하는 머리 (처음 설계)."""

    def __init__(self, cin=256):
        super().__init__(cin)
        del self.point, self.off
        self.dist = torch.nn.Conv2d(E.D, 4, 3, padding=1)
        torch.nn.init.normal_(self.dist.weight, std=0.01)
        torch.nn.init.constant_(self.dist.bias, math.log(16.0))    # 처음 거리 16 칸 = 64 px

    def forward(self, p0):
        x = self.inp(p0)
        x = self.blocks(x + pos2d(x.shape[-2], x.shape[-1], x.shape[1], x.device))
        d = torch.exp(self.dist(x).clamp(max=6.0)) * S
        X, Y = E.centers(x.shape[-2], x.shape[-1], x.device)
        return torch.cat([torch.stack([X - d[:, 0], Y - d[:, 1], X + d[:, 2], Y + d[:, 3]], 1), self.conf(x)], 1)


lift = torch.nn.Conv2d(3, 256, 3, padding=1)
for p in lift.parameters():
    p.requires_grad_(False)


def sample(bs=4):
    feats, boxes = [], []
    for _ in range(bs):
        img, bx = torch.rand(3, H * S, W * S) * 0.3, []
        for _ in range(int(torch.randint(1, 4, ()))):
            w, h = float(torch.empty(()).uniform_(30, 110)), float(torch.empty(()).uniform_(24, 80))
            x1, y1 = float(torch.empty(()).uniform_(0, W * S - w)), float(torch.empty(()).uniform_(0, H * S - h))
            img[:, int(round(y1)):int(round(y1 + h)), int(round(x1)):int(round(x1 + w))] = torch.rand(3, 1, 1) * 0.5 + 0.5
            bx.append([x1, y1, x1 + w, y1 + h])
        feats.append(F.avg_pool2d(lift(img[None]), S)[0])          # 경계가 칸 사이에 걸린다
        boxes.append(torch.tensor(bx))
    return torch.stack(feats), boxes


f = (E.EdgeField if head == "ptr" else RegField)(256)
body = [p for n, p in f.named_parameters() if n != "gate"]
base_lr = 0.02 if opt_name == "sgd" else lr
o1 = torch.optim.SGD(body, lr=base_lr, momentum=0.9, weight_decay=1e-4) if opt_name == "sgd" else torch.optim.AdamW(body, lr=base_lr, weight_decay=0.05)
o2 = torch.optim.AdamW([f.gate], lr=1e-3, weight_decay=0.0)
shape, acc = (H * S, W * S), [0.0] * 5
print(f"머리 {head} · {opt_name} lr {base_lr:g} · {STEPS} 스텝 · 모듈 파라미터 {sum(p.numel() for p in f.parameters()) / 1e6:.3f} M", flush=True)
for step in range(1, STEPS + 1):
    x, gts = sample()
    fm = f(x)
    l_map, l_conf = f.dense_losses(fm, [{"boxes": g} for g in gts])
    l_box, n, e_box, e_fix, e_vote = fm.sum() * 0, 0, 0.0, 0.0, 0.0
    c = fm[:, 4:5].sigmoid()
    for i, g in enumerate(gts):
        b = g + torch.empty_like(g).uniform_(-JIT, JIT)
        r = f.vote(fm[i], b, shape)
        l_box = l_box + ((r - g).abs() / torch.stack([g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]], 1).repeat(1, 2)).sum()
        with torch.no_grad():                                      # 투표만: g · 평균 신뢰도를 곱하기 전의 가중 평균
            cx, cy, hw, hh = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, (b[:, 2] - b[:, 0]) * E.CENTER / 2, (b[:, 3] - b[:, 1]) * E.CENTER / 2
            z = roi_align(torch.cat([fm[i:i + 1, :4] * c[i:i + 1], c[i:i + 1]], 1), [torch.stack([cx - hw, cy - hh, cx + hw, cy + hh], 1)], E.K, 1.0 / S, 2, True)
            v = z[:, :4].sum((2, 3)) / (z[:, 4].sum((1, 2))[:, None] + E.EPS)
            e_box += float((b - g).abs().sum())
            e_fix += float((r - g).abs().sum())
            e_vote += float((v - g).abs().sum())
        n += g.numel()
    loss = l_map + l_conf + l_box / n
    if not torch.isfinite(loss):
        sys.exit(f"!!! step {step}: loss 가 유한하지 않음")
    for o, top in ((o1, base_lr), (o2, 1e-3)):
        o.zero_grad()
        for pg in o.param_groups:
            pg["lr"] = top * min(1.0, step / WARM)
    loss.backward()
    o1.step()
    o2.step()
    acc = [a + v for a, v in zip(acc, (float(l_map), e_box / n, e_fix / n, e_vote / n, 1))]
    if step % 200 == 0:
        m = [a / acc[4] for a in acc[:4]]
        print(f"step {step:4d} (앞 200 스텝 평균) · 맵 손실 {m[0]:.3f} · g {float(f.gate):+.3f} · 변 오차 px: 흔든 박스 {m[1]:.2f} → 보정 뒤 {m[2]:.2f} · 투표만 {m[3]:.2f}", flush=True)
        acc = [0.0] * 5
