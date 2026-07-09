import torch
import torch.nn.functional as F

from modelinhos.tasks.standard import StandardDetection


def convert_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            boxes[..., :2] - boxes[..., 2:] / 2,
            boxes[..., :2] + boxes[..., 2:] / 2,
        ),
        dim=-1,
    )


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    # [batch, n_obj, n_anchors, 2] ~
    max_xy = torch.min(box_a[..., 2:], box_b[..., 2:])
    min_xy = torch.max(box_a[..., :2], box_b[..., :2])

    inter = torch.clamp(max_xy - min_xy, min=0)
    # [batch, n_obj, n_anchors] ~
    return inter[..., 0] * inter[..., 1]


def iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    # [batch, n_obj, n_anchors] ~
    inter = intersect(box_a, box_b)

    # [batch, n_obj] ~
    area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    # [batch, n_anchors] ~
    area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])
    union = area_a + area_b - inter

    # [batch, n_obj, n_anchors] ~
    return inter / union


def match_boxes(
    boxes: torch.Tensor,  # [n_obj, 4] normalised xyxy
    priors: torch.Tensor,  # [n_anchors, 4] normalised cxcywh
    overlap_threshold: float,
) -> torch.Tensor:  # returns a tensor of shape [n_anchors, n_obj] ~
    n_anchors = priors.shape[0]
    n_obj = boxes.shape[0]

    # Images without annotations produce empty (and, before collate has
    # seen a non-empty sample, mis-shaped (0, 1)) box tensors -- there is
    # nothing to match either way, so bail out before the IoU broadcast.
    if n_obj == 0:
        return torch.zeros(
            (n_anchors, 0), dtype=torch.bool, device=boxes.device
        )

    # Compute IoU overlaps: [n_obj, n_anchors]
    overlaps = iou(boxes[:, None], convert_to_xyxy(priors))

    # For each ground truth box, find the best matching prior (anchor)
    best_prior_overlap, best_prior_idx = overlaps.max(dim=1, keepdim=True)
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2

    # For each prior (anchor), find the best matching ground truth box
    best_truth_overlap, best_truth_idx = overlaps.max(dim=0, keepdim=True)
    best_truth_overlap = best_truth_overlap.squeeze(0)
    best_truth_idx = best_truth_idx.squeeze(0)
    best_prior_idx = best_prior_idx.squeeze(1)

    # Even if no good matches, return a zero-filled match matrix
    matching_table = torch.zeros(
        (n_anchors, n_obj), dtype=torch.bool, device=boxes.device
    )
    if valid_gt_idx.sum() == 0:
        return matching_table

    # Force match: assign each valid GT to its best matching prior. Only
    # valid GTs may claim an anchor -- invalid ones (best IoU < 0.2, e.g.
    # padding boxes, whose IoU is 0 everywhere and whose argmax therefore
    # lands on anchor 0) must not redirect anchors already force-matched
    # to real GTs.
    best_truth_overlap.index_fill_(0, best_prior_idx[valid_gt_idx], 2)
    best_truth_idx[best_prior_idx[valid_gt_idx]] = torch.where(valid_gt_idx)[0]

    # Mark anchors as positive if IoU exceeds threshold
    valid_anchors = best_truth_overlap >= overlap_threshold
    matching_table[valid_anchors, best_truth_idx[valid_anchors]] = 1
    return matching_table


def atss_boxes(
    boxes: torch.Tensor,  # [n_obj, 4] normalised xyxy
    priors: torch.Tensor,  # [n_anchors, 4] normalised cxcywh
    topk: int,
    level_sizes: list[int] | None = None,
) -> torch.Tensor:  # returns a tensor of shape [n_anchors, n_obj] ~
    """Adaptive Training Sample Selection (arxiv.org/abs/1912.02424).

    For every GT box: take the topk anchors whose centres are closest to
    the GT centre on each pyramid level, use mean(IoU) + std(IoU) of
    those candidates as a per-GT threshold, and keep the candidates above
    it whose centre lies inside the GT box. An anchor claimed by several
    GTs goes to the one it overlaps most. Padding boxes ([-1] * 4 after
    collate) are degenerate, so the centre-inside test discards them.

    level_sizes is the number of anchors per feature map, in the order
    they were generated (see level_sizes() in models/anchors.py); None
    treats the whole anchor set as a single level.
    """
    n_anchors = priors.shape[0]
    n_obj = boxes.shape[0]

    # Same guard as match_boxes: nothing to match on annotation-less
    # (possibly mis-shaped (0, 1)) box tensors.
    if n_obj == 0:
        return torch.zeros(
            (n_anchors, 0), dtype=torch.bool, device=boxes.device
        )

    sizes = list(level_sizes) if level_sizes is not None else [n_anchors]
    if sum(sizes) != n_anchors:
        raise ValueError(
            f"level_sizes {sizes} must sum to the anchor count {n_anchors}"
        )

    # [n_anchors, n_obj] ~
    overlaps = iou(boxes[:, None], convert_to_xyxy(priors)).t()
    gt_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    distances = torch.cdist(priors[:, :2], gt_centers)

    # Per pyramid level, the topk anchors closest to each GT centre:
    # [n_candidates, n_obj] flat anchor indices
    candidates_ = []
    start = 0
    for size in sizes:
        _, idx = distances[start : start + size].topk(
            min(topk, size), dim=0, largest=False
        )
        candidates_.append(idx + start)
        start += size
    candidates = torch.cat(candidates_, dim=0)

    # The adaptive part: each GT gets its own IoU threshold from the
    # statistics of its candidates (std is NaN for a single candidate)
    cious = overlaps.gather(0, candidates)
    threshold = cious.mean(dim=0) + cious.std(dim=0).nan_to_num(0.0)
    positive = cious >= threshold.unsqueeze(0)

    cx = priors[candidates, 0]
    cy = priors[candidates, 1]
    positive &= (
        (cx > boxes[:, 0])
        & (cy > boxes[:, 1])
        & (cx < boxes[:, 2])
        & (cy < boxes[:, 3])
    )

    matching_table = torch.zeros(
        (n_anchors, n_obj), dtype=torch.bool, device=boxes.device
    )
    cand_idx, obj_idx = torch.where(positive)
    matching_table[candidates[cand_idx, obj_idx], obj_idx] = True

    # Resolve anchors claimed by several GTs in favour of the highest IoU
    claimed = matching_table.any(dim=1)
    best_obj = overlaps.masked_fill(~matching_table, -1).argmax(dim=1)
    resolved = torch.zeros_like(matching_table)
    resolved[claimed, best_obj[claimed]] = True
    return resolved


def mine_negatives(
    label: torch.Tensor,
    pred: torch.Tensor,
    negpos_ratio: int,
    positive: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_anchors, _ = positive.shape
    pos_batch, pos_anchor, pos_obj = torch.where(positive)
    labels = torch.zeros_like(pred[:, :, 0], dtype=label.dtype)
    # TODO: Check why?
    labels = labels.long()
    labels[pos_batch, pos_anchor] = (
        label[pos_batch, pos_obj].squeeze(-1).long()
    )
    loss = F.cross_entropy(
        pred.view(-1, pred.shape[-1]),
        labels.view(-1),
        reduction="none",
    ).view(batch_size, num_anchors)
    loss[pos_batch, pos_anchor] = 0
    _, loss_sorted_idx = loss.sort(dim=1, descending=True)
    _, rank = loss_sorted_idx.sort(dim=1)
    num_pos = positive.sum(dim=(1, 2), dtype=label.dtype).unsqueeze(1)
    num_neg = torch.clamp(negpos_ratio * num_pos, max=num_anchors - 1)
    return rank < num_neg.expand_as(rank)


def iterative_mathing(
    classes: torch.Tensor,  # [batch_size, n_anchors, n_classes]
    boxes: torch.Tensor,  # [batch_size, n_obj, 4]
    priors: torch.Tensor,  # [n_anchors, 4]
    confidences: torch.Tensor,  # [batch_size, n_anchors, n_classes]
    negpos_ratio: int,
    overalp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    positives = torch.stack([match_boxes(b, priors, overalp) for b in boxes])
    negatives = mine_negatives(classes, confidences, negpos_ratio, positives)
    return positives, negatives


def match(
    y_pred: StandardDetection[torch.Tensor],
    y_true: StandardDetection[torch.Tensor],
    anchors: torch.Tensor,
    negpos_ratio: int,
    overalp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return iterative_mathing(
        y_true.labels,
        y_true.bboxes,
        anchors,
        confidences=y_pred.labels,
        negpos_ratio=negpos_ratio,
        overalp=overalp,
    )


def match_all_negatives(
    y_pred: StandardDetection[torch.Tensor],
    y_true: StandardDetection[torch.Tensor],
    anchors: torch.Tensor,
    overalp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matching without hard-negative mining: every unmatched anchor is a
    negative. This is the focal-loss convention (arxiv.org/abs/1708.02002)
    -- the loss itself downweights easy negatives, so mining on top of it
    would re-bias the very sample focal loss was designed to keep whole."""
    positives = torch.stack(
        [match_boxes(b, anchors, overalp) for b in y_true.bboxes]
    )
    return positives, ~positives.any(dim=2)


def atss_match(
    y_pred: StandardDetection[torch.Tensor],
    y_true: StandardDetection[torch.Tensor],
    anchors: torch.Tensor,
    negpos_ratio: int,
    topk: int = 9,
    level_sizes: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for match(): once the knobs are partial'd
    away it has the same (y_pred, y_true, anchors) tail, e.g.

        DetectionLoss(match=partial(atss_match, negpos_ratio=7))

    Unlike match() there is no IoU threshold to tune -- ATSS derives one
    per GT from its candidate statistics."""
    positives = torch.stack(
        [atss_boxes(b, anchors, topk, level_sizes) for b in y_true.bboxes]
    )
    negatives = mine_negatives(
        y_true.labels,
        y_pred.labels,
        negpos_ratio,
        positives,
    )
    return positives, negatives
