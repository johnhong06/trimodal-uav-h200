"""우리 방법 A — 융합 직전의 픽셀별 scale · shift 보정 (TR2M 의 'A ⊙ D + B' 를 두 줄기 특징에 옮긴 꼴).

저자 융합은 stage 마다 MAGE (서로의 특징을 더해 줌) → BiTE (두 줄기를 하나로) 다. MAGE 는 더하기만 해서 믿기 어려운 줄기 (밤의 RGB) 를
줄이지 못하고, 공간 가중치는 1×1 conv 라 그 픽셀만 본다. 여기서는 BiTE 에 들어가기 직전의 두 줄기 x1 (RGB) · x2 (열화상 + 이벤트) 를 픽셀마다 고친다.

  a, b    두 줄기를 stride 16 격자로 줄이고 1×1 conv 로 d 채널로 (+ 위치 부호)
  c1      교차 어텐션: Q = a, K · V = b (stride 32 격자로 한 번 더 줄임).  c2 는 반대 방향
  γ1, β1  1×1 conv 머리 ([a ; b ; c1]) → γ1 = 2·sigmoid(·) (1 채널, 0–2) · β1 (C 채널). 마지막 conv 는 0 에서 시작 → γ = 1, β = 0
  x1'     γ1 ⊙ x1 + β1 (맵은 stage 해상도로 키움).  x2' 도 같은 식
  BiTE(x1', x2')  — 다음 stage 로 가는 두 줄기는 고치지 않는다 (base 와 같은 길)

학습 첫 순간의 출력은 base 와 똑같다. 추가 인코더는 없다 — 입력은 검출기가 이미 만든 두 줄기 특징뿐.
저자 코드는 고치지 않고 encoder.BiTEs[i] 를 '보정 + 원래 BiTE' 로 감싼다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D, HEADS = 64, 4
POOL, KV_POOL = (4, 2, 1, 1), (2, 2, 2, 1)       # stage 1–4 (stride 4 · 8 · 16 · 32) → Q 격자 stride 16 · 16 · 16 · 32, K·V 격자 stride 32


def pos2d(h, w, d, device):
    """2 차원 사인 위치 부호 (d, h, w) — 앞 절반 채널은 세로, 뒤 절반은 가로 위치."""
    q = d // 4
    f = torch.exp(torch.arange(q, device=device, dtype=torch.float32) * (-math.log(10000.0) / q))
    y = torch.arange(h, device=device, dtype=torch.float32)[:, None] * f
    x = torch.arange(w, device=device, dtype=torch.float32)[:, None] * f
    py = torch.cat([y.sin(), y.cos()], 1)[:, None, :].expand(h, w, 2 * q)
    px = torch.cat([x.sin(), x.cos()], 1)[None, :, :].expand(h, w, 2 * q)
    return torch.cat([py, px], 2).permute(2, 0, 1)


class CrossAttn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, q, kv):                                      # q (B, N, d) · kv (B, M, d)
        B, N, d = q.shape
        h = self.heads
        qq = self.q(q).view(B, N, h, d // h).transpose(1, 2)
        k, v = self.kv(kv).view(B, -1, 2, h, d // h).permute(2, 0, 3, 1, 4)
        return self.o(F.scaled_dot_product_attention(qq, k, v).transpose(1, 2).reshape(B, N, d))


class Rescale(nn.Module):
    """한 stage 의 두 줄기 (B, C, H, W) → 픽셀별 γ · β 로 고친 두 줄기."""

    def __init__(self, dim, pool, kv_pool, d=D, heads=HEADS):
        super().__init__()
        self.pool, self.kv_pool = pool, kv_pool
        self.inp = nn.ModuleList([nn.Conv2d(dim, d, 1) for _ in range(2)])
        self.norm = nn.ModuleList([nn.GroupNorm(1, d) for _ in range(2)])
        self.attn = nn.ModuleList([CrossAttn(d, heads) for _ in range(2)])
        self.mix = nn.ModuleList([nn.Sequential(nn.Conv2d(3 * d, d, 1), nn.GELU()) for _ in range(2)])
        self.gamma = nn.ModuleList([nn.Conv2d(d, 1, 1) for _ in range(2)])
        self.beta = nn.ModuleList([nn.Conv2d(d, dim, 1) for _ in range(2)])
        for m in list(self.gamma) + list(self.beta):               # 0 에서 시작 → γ = 1 · β = 0 (base 와 같은 출력)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def maps(self, x1, x2):
        """→ (γ1, β1, γ2, β2), stage 해상도. γ (B, 1, H, W) · β (B, C, H, W)."""
        size = x1.shape[-2:]
        f = [self.norm[i](self.inp[i](F.avg_pool2d(x, self.pool, ceil_mode=True) if self.pool > 1 else x)) for i, x in enumerate((x1, x2))]
        B, d, h, w = f[0].shape
        pe = pos2d(h, w, d, x1.device)
        tok = lambda z: z.flatten(2).transpose(1, 2)
        red = lambda z: F.avg_pool2d(z, self.kv_pool, ceil_mode=True) if self.kv_pool > 1 else z
        out = []
        for i in range(2):                                         # i = 자기 줄기, 1 − i = 상대 줄기
            c = self.attn[i](tok(f[i] + pe), tok(red(f[1 - i] + pe))).transpose(1, 2).reshape(B, d, h, w)
            m = self.mix[i](torch.cat([f[i], f[1 - i], c], 1))
            up = lambda z: F.interpolate(z, size=size, mode="bilinear", align_corners=False) if z.shape[-2:] != size else z
            out += [2 * torch.sigmoid(up(self.gamma[i](m))), up(self.beta[i](m))]
        return out

    def forward(self, x1, x2):
        g1, b1, g2, b2 = self.maps(x1, x2)
        return g1 * x1 + b1, g2 * x2 + b2


class RescaledBiTE(nn.Module):
    def __init__(self, bite, dim, pool, kv_pool):
        super().__init__()
        self.fix = Rescale(dim, pool, kv_pool)
        self.bite = bite

    def forward(self, x1, x2):
        return self.bite(*self.fix(x1, x2))


def attach(enc):
    """저자 인코더의 BiTE 넷을 '보정 + BiTE' 로 감싼다 (사전학습 가중치를 넣은 뒤에 부른다)."""
    for i, dim in enumerate(enc.embed_dims):
        enc.BiTEs[i] = RescaledBiTE(enc.BiTEs[i], dim, POOL[i], KV_POOL[i])
