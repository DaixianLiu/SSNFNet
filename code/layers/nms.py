import torch
from torchvision.ops import boxes as box_ops

def nms(boxes, scores, iou_threshold, sigma=0.5, score_threshold=0.9):
    """
    Args:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold
        sigma (float): parameter for Gaussian penalty function
        score_threshold (float): boxes with scores < score_threshold are discarded before computation

    Returns:
        Tensor: indices to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = torch.clamp(x1[order[1:]], min=x1[i].item())
        yy1 = torch.clamp(y1[order[1:]], min=y1[i].item())
        xx2 = torch.clamp(x2[order[1:]], max=x2[i].item())
        yy2 = torch.clamp(y2[order[1:]], max=y2[i].item())

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        weight = torch.exp(-(iou**2) / sigma)
        scores[order[1:]] *= weight
        mask = scores[order[1:]] > score_threshold
        order = order[1:][mask]

    return torch.tensor(keep, device=boxes.device, dtype=torch.long)

def batched_nms(boxes, scores, idxs, iou_threshold):
    assert boxes.shape[-1] == 4
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep

# Note: this function (nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
def nms_rotated(boxes, scores, iou_threshold):
    from fsdet import _C
    return _C.nms_rotated(boxes, scores, iou_threshold)

def batched_nms_rotated(boxes, scores, idxs, iou_threshold):
    # This function remains the same as it's not clear how to apply Soft-NMS to rotated boxes
    assert boxes.shape[-1] == 5
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = (
        torch.max(boxes[:, 0], boxes[:, 1]) + torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).max()
    min_coordinate = (
        torch.min(boxes[:, 0], boxes[:, 1]) - torch.min(boxes[:, 2], boxes[:, 3]) / 2
    ).min()
    offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, :2] += offsets[:, None]
    keep = nms_rotated(boxes_for_nms, scores, iou_threshold)
    return keep