"""3D losses: CenterPoint-style. Focal on heatmap + L1 on offset, height, size, yaw(sin/cos)."""
import torch
import torch.nn.functional as F

from train.losses_2d import focal_loss, l1_loss


def compute_3d_loss(pred, target, w_heat=1.0, w_off=1.0, w_height=1.0, w_size=0.1, w_yaw=0.5):
    mask = (target["heat"].max(dim=1, keepdim=True)[0] > 0.99).float()
    loss = w_heat * focal_loss(pred["heat"], target["heat"])
    loss = loss + w_off * l1_loss(pred["off"], target["off"], mask)
    loss = loss + w_height * l1_loss(pred["height"], target["height"], mask)
    loss = loss + w_size * l1_loss(pred["size"], target["size"], mask)
    loss = loss + w_yaw * l1_loss(pred["yaw"], target["yaw"], mask)
    return {"loss": loss, "heat": focal_loss(pred["heat"], target["heat"]).item()}