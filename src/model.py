"""모델: 저자 모델 (MiT-B1 두 줄기 + MAGE·BiTE 네 stage + FPN + Faster R-CNN) 을 저자 Trainer.build_model 과 같은 순서로 만든다.

팔 (arm)
  paper   저자 그대로: 처음부터 학습, 입력 0–255 를 0–1 용 평균·표준편차로 정규화 (저자 config.py 값)
  base    + ImageNet 사전학습 MiT-B1 (nvidia/mit-b1) 을 두 줄기에 · 입력을 /255 한 것과 같은 정규화 (RGB = ImageNet, thermal·event = 학습 분할 통계)
  inject  base + 고정 DINOv2-L 특징을 0 에서 시작하는 게이트로 FPN 에 더함 (src/inject.py)
모든 팔에서 Faster R-CNN 설정 (앵커 · RoIAlign · 이미지 크기 기본값 800/1333) 은 저자와 같다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from torchvision.models.detection import FasterRCNN  # noqa: E402
from torchvision.models.detection.rpn import AnchorGenerator  # noqa: E402
from torchvision.ops import MultiScaleRoIAlign  # noqa: E402

from common import author, need, te_stats  # noqa: E402

ARMS = ("paper", "base", "inject")
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def norm_values(arm):
    """Faster R-CNN 의 image_mean · image_std (입력은 저자 Dataset 그대로 0–255)."""
    if arm == "paper":
        c = author().config.Config()
        return list(c.image_mean), list(c.image_std)
    s = te_stats()
    mean = [m * 255 for m in IMNET_MEAN] + [s["thermal_mean"] * 255, s["event_mean"] * 255]
    std = [v * 255 for v in IMNET_STD] + [s["thermal_std"] * 255, s["event_std"] * 255]
    return mean, std


def map_mit(sd):
    """nvidia/mit-b1 (transformers SegformerForImageClassification) 의 이름 → 저자 MiT (SegFormer 원본) 이름, RGB 줄기.
    key · value 두 Linear 는 저자 Attention 의 kv 하나로 이어 붙인다 (앞 절반 = key, 뒤 절반 = value)."""
    pre = "segformer.encoder."
    rename = [("layer_norm_1.", "norm1."), ("layer_norm_2.", "norm2."), ("attention.self.query.", "attn.q."),
              ("attention.self.sr.", "attn.sr."), ("attention.self.layer_norm.", "attn.norm."),
              ("attention.output.dense.", "attn.proj."), ("mlp.dense1.", "mlp.fc1."),
              ("mlp.dwconv.dwconv.", "mlp.dwconv.dwconv."), ("mlp.dense2.", "mlp.fc2.")]
    out, kv = {}, {}
    for k, v in sd.items():
        if not k.startswith(pre):
            continue                                              # classifier.* 등
        p = k[len(pre):].split(".")
        if p[0] == "patch_embeddings":
            sub = {"proj": "proj", "layer_norm": "norm"}[p[2]]
            out[f"patch_embed{int(p[1]) + 1}.{sub}.{p[3]}"] = v
        elif p[0] == "layer_norm":
            out[f"norm{int(p[1]) + 1}.{p[2]}"] = v
        elif p[0] == "block":
            i, j, rest = int(p[1]) + 1, int(p[2]), ".".join(p[3:])
            for a, b in rename:
                if rest.startswith(a):
                    out[f"block{i}.{j}.{b}{rest[len(a):]}"] = v
                    break
            else:
                if not rest.startswith(("attention.self.key.", "attention.self.value.")):
                    raise KeyError(k)
                kv.setdefault((i, j, p[-1]), {})[p[-2]] = v
        else:
            raise KeyError(k)
    for (i, j, wb), d in kv.items():
        out[f"block{i}.{j}.attn.kv.{wb}"] = torch.cat([d["key"], d["value"]], 0)
    return out


def load_mit(enc, path):
    """사전학습 가중치를 두 줄기에 넣는다. 보조 줄기 (thermal + event, 2 채널) 는 같은 가중치,
    첫 합성곱만 RGB 3 채널 가중치의 평균을 2 채널로 복사. 융합 모듈 (MAGEs · BiTEs) 은 저자 초기화 그대로."""
    rgb = map_mit(torch.load(path, map_location="cpu", weights_only=True))
    full = dict(rgb)
    nx = enc.extra_patch_embed1.proj.in_channels
    for k, v in rgb.items():
        if k == "patch_embed1.proj.weight":
            v = v.mean(1, keepdim=True).repeat(1, nx, 1, 1)
        full["extra_" + k] = v.clone()
    r = enc.load_state_dict(full, strict=False)
    assert not r.unexpected_keys, r.unexpected_keys[:5]
    assert all(k.startswith(("MAGEs.", "BiTEs.")) for k in r.missing_keys), [k for k in r.missing_keys if not k.startswith(("MAGEs.", "BiTEs."))][:5]
    return len(full)


def build(arm, dino_dir=None, dino_cfg=None):
    """arm → FasterRCNN. dino_cfg (시험용) 를 주면 그 설정으로 무작위 초기화한 DINOv2 를 쓴다."""
    assert arm in ARMS, arm
    t = author()
    c = t.config.Config()
    enc = t.models.encoder.get_encoder(c.backbone_type, in_chans_rgb=c.in_chans_rgb, in_chans_x=c.in_chans_x)
    if arm != "paper":
        load_mit(enc, need("TRI_MIT"))
    backbone = t.models.backbone.InterModalBackbone(enc, fpn_out_channels=c.fpn_out_channels)
    if arm == "inject":
        from inject import InjectBackbone
        backbone = InjectBackbone(backbone, dino_dir or (None if dino_cfg else need("TRI_DINO")), dino_cfg)
    mean, std = norm_values(arm)
    return FasterRCNN(backbone, num_classes=2,                         # 저자: 라벨 최대 id 0 + 1, 배경 + 1
                      rpn_anchor_generator=AnchorGenerator(sizes=c.anchor_sizes, aspect_ratios=c.anchor_aspect_ratios),
                      box_roi_pool=MultiScaleRoIAlign(featmap_names=c.roi_featmap_names, output_size=c.roi_output_size,
                                                      sampling_ratio=c.roi_sampling_ratio),
                      image_mean=mean, image_std=std)


FROZEN = "backbone.dino."


def save_state(model):
    """고정 DINOv2 가중치는 빼고 저장 (HF 에서 다시 받으므로)."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith(FROZEN)}


def load_state(model, sd):
    r = model.load_state_dict(sd, strict=False)
    assert not r.unexpected_keys, r.unexpected_keys[:5]
    assert all(k.startswith(FROZEN) for k in r.missing_keys), [k for k in r.missing_keys if not k.startswith(FROZEN)][:5]


def count_params(model):
    tot = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(FROZEN))
    frozen = sum(p.numel() for n, p in model.named_parameters() if n.startswith(FROZEN))
    return {"detector_m": round(tot / 1e6, 2), "frozen_encoder_m": round(frozen / 1e6, 2)}
