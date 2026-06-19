"""Run a model over a loader -> per-frame detections + GT, for the metric suite."""
import torch

from common.geometry.boxes2d import nms2d
from common.geometry.boxes3d import rotated_nms3d


@torch.no_grad()
def run_inference(model, loader, cfg, device, nms=True):
    space = model.output_space()
    model.eval()
    dets, gts = {}, {}
    for batch in loader:
        batch["image"] = batch["image"].to(device)
        pred = model(batch)
        outs = model.decode(pred, cfg)
        for i, img_id in enumerate(batch["image_id"]):
            d = outs[i]
            boxes = d["boxes"].detach().cpu()
            scores = d["scores"].detach().cpu()
            labels = d["labels"].detach().cpu()
            if nms and boxes.shape[0] > 0:
                if space == "2d":
                    keep = nms2d(boxes.numpy(), scores.numpy(), 0.45)
                else:
                    keep = rotated_nms3d(boxes.numpy(), scores.numpy(), 0.1)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            dets[img_id] = {"boxes": boxes, "scores": scores, "labels": labels}
            gts[img_id] = batch["boxes2d"][i] if space == "2d" else batch["boxes3d"][i]
    return dets, gts, space