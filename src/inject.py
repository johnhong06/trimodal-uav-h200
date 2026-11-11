"""우리 방법: 고정 인코더 특징의 게이트 주입 (depthvlm-silog 해결 실험과 같은 꼴 — 0 에서 여는 층별 게이트 × 투영(LayerNorm(특징))).

  FPN 출력 P_l (l = 0..3, stride 4·8·16·32) ← P_l + g_l · Conv1x1( [LN(DINOv2_RGB 층 k_l) ; LN(DINOv2_thermal 층 k_l)] ) (P_l 크기로 맞춤)
  g_l 은 0 에서 시작 → 학습 첫 순간의 출력은 base 팔과 똑같다. 'pool' 층은 바뀐 P_3 에서 다시 만든다 (저자 backbone 과 같은 식).
  DINOv2 는 고정 (학습 안 함, bf16, 기울기 없음). 층 k_l = 깊이의 1/4 · 2/4 · 3/4 · 끝 (ViT-L 이면 6 · 12 · 18 · 24).
  RGB 는 검출기 정규화 그대로 (ImageNet 값) 넣고, thermal 은 0–1 로 되돌려 3 채널로 복사한 뒤 ImageNet 값으로 정규화한다.
  밤에는 RGB 가 거의 검어서 (밝기 중앙값 18.9/255) thermal 쪽을 같이 넣는다. event 는 넣지 않는다 (밤에는 거의 비어 있음).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import te_stats
from model import IMNET_MEAN, IMNET_STD


class InjectBackbone(nn.Module):
    def __init__(self, base, dino_dir=None, dino_cfg=None):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model
        self.base = base
        self.out_channels = base.out_channels
        if dino_cfg is not None:                                   # 시험용 작은 무작위 모델
            self.dino = Dinov2Model(Dinov2Config(**dino_cfg))
        else:
            self.dino = Dinov2Model.from_pretrained(dino_dir, attn_implementation="sdpa")
        self.dino.requires_grad_(False)
        self.dino.eval()
        d = self.dino.config.num_hidden_layers
        self.layers = [d // 4, d // 2, 3 * d // 4, d]
        self.patch = self.dino.config.patch_size
        hid = self.dino.config.hidden_size
        self.proj = nn.ModuleList([nn.Conv2d(2 * hid, self.out_channels, 1) for _ in self.layers])
        self.gate = nn.Parameter(torch.zeros(len(self.layers)))
        s = te_stats()
        self.register_buffer("t_mean", torch.tensor(s["thermal_mean"]), persistent=False)
        self.register_buffer("t_std", torch.tensor(s["thermal_std"]), persistent=False)
        self.register_buffer("im_mean", torch.tensor(IMNET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("im_std", torch.tensor(IMNET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()                                           # 고정 인코더는 늘 평가 모드
        return self

    @torch.no_grad()
    def dino_feats(self, x):
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        x = F.interpolate(x, size=(h * self.patch, w * self.patch), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            hs = self.dino(pixel_values=x, output_hidden_states=True).hidden_states
        out = []
        for k in self.layers:
            t = hs[k][:, 1:, :].float()                            # CLS 를 뺀 패치 토큰 (dinov2-large 는 register 없음)
            t = F.layer_norm(t, t.shape[-1:])
            out.append(t.transpose(1, 2).reshape(t.shape[0], -1, h, w))
        return out

    def forward(self, images):
        feats = self.base(images)
        rgb = images[:, :3]
        th = (images[:, 3:4] * self.t_std + self.t_mean).expand(-1, 3, -1, -1)
        th = (th - self.im_mean) / self.im_std
        fr, ft = self.dino_feats(rgb), self.dino_feats(th)
        for i, k in enumerate(("0", "1", "2", "3")):
            f = self.proj[i](torch.cat([fr[i], ft[i]], 1))
            f = F.interpolate(f, size=feats[k].shape[-2:], mode="bilinear", align_corners=False)
            feats[k] = feats[k] + self.gate[i] * f
        feats["pool"] = F.max_pool2d(feats["3"], 1, 2, 0)
        return feats
