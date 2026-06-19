"""2D losses: CenterNet focal loss on heatmap + L1 on offset & size."""
import torch
import torch.nn.functional as F


def focal_loss(pred, target, alpha=2.0, beta=4.0, eps=1e-6):
    """pred: (B,C,H,W) logits (pre-sigmoid). target: (B,C,H,W) gaussian heatmaps in [0,1]."""
    pred = pred.sigmoid().clamp(eps, 1 - eps)
    pos = target.eq(1).float()
    neg = target.lt(1).float()
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg
    num_pos = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def l1_loss(pred, target, mask):
    """mask: (B,1,H,W) selects cells that own a box. Smooth-L1 optional."""
    return F.l1_loss(pred * mask, target * mask, reduction="sum") / mask.sum().clamp(min=1)


def compute_2d_loss(pred, target, w_heat=1.0, w_off=1.0, w_size=0.1):
    """pred: {heat, off, size}; target: {heat, off, size}."""
    # build a mask from the heatmap peaks (where a GT center exists)
    mask = (target["heat"].max(dim=1, keepdim=True)[0] > 0.99).float()
    loss = w_heat * focal_loss(pred["heat"], target["heat"])
    loss = loss + w_off * l1_loss(pred["off"], target["off"], mask)
    loss = loss + w_size * l1_loss(pred["size"], target["size"], mask)
    return {"loss": loss, "heat": focal_loss(pred["heat"], target["heat"]).item(),
            "off": l1_loss(pred["off"], target["off"], mask).item(),
            "size": l1_loss(pred["size"], target["size"], mask).item()}