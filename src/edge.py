"""우리 방법 B — 픽셀별 경계 맵으로 박스의 네 변을 고친다 (거친 예측을 픽셀별 맵으로 보정한다는 점이 TR2M 과 같다).

저자 검출기는 후보 박스마다 RoIAlign 7×7 → FC 2 층 → 보정값 4 개를 한 번에 낸다. 차 한 대 (학습 크기에서 한 변 약 138 px) 에
격자 한 칸이 약 20 px 라서, IoU 0.9 에 필요한 몇 px 단위의 위치는 FC 가 짐작해야 한다. 여기서는 가장 고해상도 FPN 층에서 변의 위치를 따로 읽는다.

  경계 맵  M = EdgeField(P0).  P0 = FPN stride 4 층. 픽셀마다 [자기를 담은 차의 왼 · 위 · 오른 · 아래 변의 좌표 (px) 4 장 ; 신뢰도 1 장]
           몸통 = 1×1 conv (256 → 64) + 위치 부호 → [가로줄 어텐션 → 세로줄 어텐션 → conv MLP] × 2.
           머리 = 변마다 어텐션이 같은 줄에서 '변이 놓인 칸' 을 가리키고 그 칸의 좌표를 읽는다 (soft-argmax) + 3×3 conv 의 작은 보정값.
           박스의 변은 축에 나란하니 왼 · 오른 변은 같은 가로줄, 위 · 아래 변은 같은 세로줄만 보면 된다 (5 만 토큰을 줄 단위로 나눠 싸다).
           거리를 숫자로 회귀하는 머리보다 이쪽이 빨리 배운다 (2026-11-12 합성 실험, NOTES D-15).
  투표     박스 가운데 절반 영역의 픽셀이 말한 변의 좌표를 신뢰도로 가중 평균한다 (RoIAlign 7×7 한 번).
  보정     변 ← 변 + g · c̄ · (투표 − 변).  g 는 0 에서 시작하는 스칼라 → 학습 첫 순간의 박스는 base 와 같다.
           c̄ = 그 영역의 평균 신뢰도 (0–1) — 경계 맵이 확신하지 못하는 곳의 박스는 움직이지 않는다. 화면 끝에 닿은 변은 그대로 둔다.
  손실     loss_edge_map   픽셀이 말한 변 vs 정답 변: 오차 / 박스 크기의 L1 (정답 박스 안의 픽셀, 겹치면 작은 박스)
           loss_edge_conf  신뢰도가 그 픽셀 박스의 IoU 를 맞히게 (quality focal loss, 박스 밖 픽셀의 정답은 0)
           loss_edge_box   보정한 박스 vs 정답: 변 오차 / 박스 크기의 L1 (학습 때 양성 후보의 예측 박스에 투표를 적용).
                           예측 박스 쪽 기울기는 끊는다 — 박스 헤드는 base 와 같은 손실로만 학습한다.
추론에서는 NMS 뒤 최종 박스 (사진마다 100 개까지) 에 투표를 적용한다. refine = False 면 투표를 끈다 (같은 체크포인트로 '투표 끔' 평가).
추가 인코더는 없다 — 입력은 검출기가 이미 만든 P0 뿐. torchvision RoIHeads 를 물려받아 박스 부분의 forward 만 다시 쓴다 (0.22.1 기준).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops
from torchvision.ops import roi_align

from rescale import pos2d

D, HEADS, DEPTH, STRIDE, DK = 64, 4, 2, 4, 32      # DK = 가리키는 머리의 q · k 폭
K, CENTER, EPS = 7, 0.5, 1e-4                    # 투표: 박스 가운데 CENTER 영역을 K×K 로 읽는다


class Axial(nn.Module):
    """한 축 안에서만 보는 자기 어텐션. rows = True 면 같은 가로줄, False 면 같은 세로줄."""

    def __init__(self, d, heads, rows):
        super().__init__()
        self.heads, self.rows = heads, rows
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):                                          # (B, d, H, W)
        t = x.permute(0, 2, 3, 1) if self.rows else x.permute(0, 3, 2, 1)
        B, n1, n2, d = t.shape
        h = self.heads
        q, k, v = self.qkv(t.reshape(B * n1, n2, d)).view(B * n1, n2, 3, h, d // h).permute(2, 0, 3, 1, 4)
        o = self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, n1, n2, d))
        return o.permute(0, 3, 1, 2) if self.rows else o.permute(0, 3, 2, 1)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.GroupNorm(1, d), nn.GroupNorm(1, d), nn.GroupNorm(1, d)
        self.row, self.col = Axial(d, heads, True), Axial(d, heads, False)
        self.mlp = nn.Sequential(nn.Conv2d(d, 2 * d, 1), nn.Conv2d(2 * d, 2 * d, 3, padding=1, groups=2 * d), nn.GELU(), nn.Conv2d(2 * d, d, 1))

    def forward(self, x):
        x = x + self.row(self.n1(x))
        x = x + self.col(self.n2(x))
        return x + self.mlp(self.n3(x))


def centers(h, w, device):
    """P0 픽셀 (i, j) 의 가운데 = 학습 크기 좌표 ((j + 0.5)·STRIDE, (i + 0.5)·STRIDE). → X, Y (h, w)"""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * STRIDE
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * STRIDE
    return xs[None, :].expand(h, w), ys[:, None].expand(h, w)


class Pointer(nn.Module):
    """변마다 같은 줄에서 '변이 놓인 칸' 을 어텐션으로 가리키고 그 칸의 좌표를 읽는다 (soft-argmax).
    왼 · 오른 변 (0 · 2) 은 가로줄 안에서 x 좌표, 위 · 아래 변 (1 · 3) 은 세로줄 안에서 y 좌표."""

    def __init__(self, d, dk=DK):
        super().__init__()
        self.dk = dk
        self.q, self.k = nn.Conv2d(d, 4 * dk, 1), nn.Conv2d(d, 4 * dk, 1)

    def forward(self, x):                                          # (B, d, H, W) → (B, 4, H, W) [x1, y1, x2, y2] (학습 크기 px)
        B, _, H, W = x.shape
        q, k = self.q(x).view(B, 4, self.dk, H, W), self.k(x).view(B, 4, self.dk, H, W)
        X, Y = centers(H, W, x.device)
        out = []
        for s in range(4):
            if s % 2 == 0:                                         # 가로줄 안에서: 어텐션 (B, H, W, W)
                a = torch.softmax(q[:, s].permute(0, 2, 3, 1) @ k[:, s].permute(0, 2, 1, 3) / self.dk ** 0.5, -1)
                out.append(a @ X[0])
            else:                                                  # 세로줄 안에서: 어텐션 (B, W, H, H)
                a = torch.softmax(q[:, s].permute(0, 3, 2, 1) @ k[:, s].permute(0, 3, 1, 2) / self.dk ** 0.5, -1)
                out.append((a @ Y[:, 0]).transpose(1, 2))
        return torch.stack(out, 1)


class EdgeField(nn.Module):
    def __init__(self, cin=256, d=D, heads=HEADS, depth=DEPTH):
        super().__init__()
        self.inp = nn.Conv2d(cin, d, 1)
        self.blocks = nn.Sequential(*[Block(d, heads) for _ in range(depth)])
        self.point = Pointer(d)
        self.off, self.conf = nn.Conv2d(d, 4, 3, padding=1), nn.Conv2d(d, 1, 3, padding=1)
        nn.init.zeros_(self.off.weight)                            # 칸 안의 작은 보정값 (px) — 0 에서 시작
        nn.init.zeros_(self.off.bias)
        nn.init.normal_(self.conf.weight, std=0.01)
        nn.init.constant_(self.conf.bias, -4.6)                    # 처음 신뢰도 0.01
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, p0):
        """P0 (B, 256, H, W) → (B, 5, H, W): 픽셀이 말하는 변의 좌표 [x1, y1, x2, y2] (학습 크기 px) · 신뢰도 logit."""
        x = self.inp(p0)
        pe = pos2d(x.shape[-2], x.shape[-1], x.shape[1], x.device)
        x = self.blocks(x + pe)
        return torch.cat([self.point(x + pe) + self.off(x), self.conf(x)], 1)

    def vote(self, fmap, boxes, shape):
        """사진 한 장: fmap (5, H, W) · boxes (N, 4, 학습 크기 px) · shape (h, w) → 보정한 박스 (N, 4)."""
        if boxes.numel() == 0:
            return boxes
        c = fmap[None, 4:5].sigmoid()
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        hw, hh = (boxes[:, 2] - boxes[:, 0]) * CENTER / 2, (boxes[:, 3] - boxes[:, 1]) * CENTER / 2
        rois = torch.stack([cx - hw, cy - hh, cx + hw, cy + hh], 1)
        z = roi_align(torch.cat([fmap[None, :4] * c, c], 1), [rois], output_size=K, spatial_scale=1.0 / STRIDE, sampling_ratio=2, aligned=True)
        v = z[:, :4].sum((2, 3)) / (z[:, 4].sum((1, 2))[:, None] + EPS)                      # 신뢰도 가중 평균
        out = boxes + self.gate * z[:, 4].mean((1, 2))[:, None] * (v - boxes)                # 평균 신뢰도만큼만 움직인다
        h, w = shape
        at_border = torch.stack([boxes[:, 0] <= 1, boxes[:, 1] <= 1, boxes[:, 2] >= w - 1, boxes[:, 3] >= h - 1], 1)
        out = box_ops.clip_boxes_to_image(torch.where(at_border, boxes, out), shape)
        bad = ((out[:, 2] - out[:, 0]) < 1) | ((out[:, 3] - out[:, 1]) < 1)                  # 뒤집힌 박스는 원래대로
        return torch.where(bad[:, None], boxes, out)

    def dense_losses(self, fmap, targets):
        """경계 맵 손실 둘 (loss_edge_map, loss_edge_conf). 양성 = 정답 박스 안의 픽셀 (겹치면 작은 박스)."""
        B, _, H, W = fmap.shape
        X, Y = centers(H, W, fmap.device)
        l_map, l_conf, n_pos = fmap.sum() * 0, fmap.sum() * 0, 0
        for i, t in enumerate(targets):
            g = t["boxes"].to(fmap.dtype)
            q = torch.zeros(H, W, device=fmap.device, dtype=fmap.dtype)                      # 신뢰도 정답 (IoU)
            if len(g):
                inside = (X > g[:, 0, None, None]) & (X < g[:, 2, None, None]) & (Y > g[:, 1, None, None]) & (Y < g[:, 3, None, None])
                area = ((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))[:, None, None].expand(-1, H, W).masked_fill(~inside, float("inf"))
                amin, idx = area.min(0)
                pos = torch.isfinite(amin)
                if pos.any():
                    pb, gb = fmap[i, :4].permute(1, 2, 0)[pos], g[idx[pos]]
                    size = torch.stack([gb[:, 2] - gb[:, 0], gb[:, 3] - gb[:, 1]], 1).clamp(min=1).repeat(1, 2)
                    l_map = l_map + ((pb - gb).abs() / size).sum() / 4
                    with torch.no_grad():                          # 신뢰도 정답 = 픽셀 박스의 IoU (변이 뒤집혀 있으면 바로 세워서)
                        pb = torch.cat([torch.min(pb[:, :2], pb[:, 2:]), torch.max(pb[:, :2], pb[:, 2:])], 1)
                        wh = (torch.min(pb[:, 2:], gb[:, 2:]) - torch.max(pb[:, :2], gb[:, :2])).clamp(min=0)
                        inter = wh[:, 0] * wh[:, 1]
                        union = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]) + (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]) - inter
                        q[pos] = (inter / union.clamp(min=1e-6)).clamp(0, 1)
                    n_pos += int(pos.sum())
            z = fmap[i, 4]
            l_conf = l_conf + (F.binary_cross_entropy_with_logits(z, q, reduction="none") * (z.sigmoid() - q).abs().pow(2)).sum()
        n = max(n_pos, 1)
        return l_map / n, l_conf / n


class EdgeRoIHeads(RoIHeads):
    """torchvision RoIHeads + 경계 맵. 박스 헤드의 학습 (표본 뽑기 · 손실) 과 후처리 (NMS) 는 그대로다."""

    def __init__(self, old, cin):
        super().__init__(old.box_roi_pool, old.box_head, old.box_predictor,
                         old.proposal_matcher.high_threshold, old.proposal_matcher.low_threshold,
                         old.fg_bg_sampler.batch_size_per_image, old.fg_bg_sampler.positive_fraction,
                         old.box_coder.weights, old.score_thresh, old.nms_thresh, old.detections_per_img)
        self.edge = EdgeField(cin)
        self.refine = True

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            fmap = self.edge(features["0"])
            l_map, l_conf = self.edge.dense_losses(fmap, targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "loss_edge_map": l_map, "loss_edge_conf": l_conf,
                      "loss_edge_box": self.box_loss(fmap, proposals, box_regression, labels, matched_idxs, targets, image_shapes)}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            if self.refine:
                fmap = self.edge(features["0"])
                boxes = [self.edge.vote(fmap[i], b, image_shapes[i]) for i, b in enumerate(boxes)]
            result = [{"boxes": b, "labels": lb, "scores": s} for b, lb, s in zip(boxes, labels, scores)]
        return result, losses

    def box_loss(self, fmap, proposals, box_regression, labels, matched_idxs, targets, image_shapes):
        pred = self.box_coder.decode(box_regression.detach(), proposals).split([len(p) for p in proposals])
        tot, n = fmap.sum() * 0, 0
        for i, (pb, lb, mi) in enumerate(zip(pred, labels, matched_idxs)):
            pos = lb > 0
            if not pos.any():
                continue
            b = box_ops.clip_boxes_to_image(pb[pos, lb[pos]], image_shapes[i])
            g = targets[i]["boxes"].to(b.dtype)[mi[pos]]
            wh = torch.stack([g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]], 1).clamp(min=1).repeat(1, 2)
            tot = tot + ((self.edge.vote(fmap[i], b, image_shapes[i]) - g).abs() / wh).sum()
            n += 4 * int(pos.sum())
        return tot / max(n, 1)


def attach(model):
    """FasterRCNN 의 roi_heads 를 경계 맵이 달린 것으로 바꾼다 (박스 헤드 모듈 · 설정값은 그대로 옮김)."""
    model.roi_heads = EdgeRoIHeads(model.roi_heads, model.backbone.out_channels)
