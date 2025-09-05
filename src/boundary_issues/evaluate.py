import numpy as np
from skimage.segmentation import relabel_sequential
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from funlib.evaluate import rand_voi


def evaluate(gt_labels: np.ndarray, pred_labels: np.ndarray, th: float = 0.5):
    """
    Evaluate segmentation using multiple metrics:
    - Precision
    - Recall
    - Accuracy
    - Mean IoU
    - VOI (split and merge)
    - F1 score (Dice)
    """

    # Relabel to ensure contiguous labeling
    pred_labels_rel, _, _ = relabel_sequential(pred_labels)
    gt_labels_rel, _, _ = relabel_sequential(gt_labels)

    overlay = np.array([pred_labels_rel.flatten(), gt_labels_rel.flatten()])

    # Get overlap counts
    overlay_labels, overlay_labels_counts = np.unique(
        overlay, return_counts=True, axis=1
    )
    overlay_labels = np.transpose(overlay_labels)

    # Count label sizes
    gt_labels_list, gt_counts = np.unique(gt_labels_rel, return_counts=True)
    pred_labels_list, pred_counts = np.unique(pred_labels_rel, return_counts=True)

    gt_count_dict = {l: c for l, c in zip(gt_labels_list, gt_counts)}
    pred_count_dict = {l: c for l, c in zip(pred_labels_list, pred_counts)}

    num_gt = int(np.max(gt_labels_rel))
    num_pred = int(np.max(pred_labels_rel))
    num_matches = min(num_gt, num_pred)

    # Create IoU table
    iou_mat = np.zeros((num_gt + 1, num_pred + 1), dtype=np.float32)
    for (u, v), c in zip(overlay_labels, overlay_labels_counts):
        iou = c / (gt_count_dict[v] + pred_count_dict[u] - c)
        iou_mat[int(v), int(u)] = iou

    # Remove background
    iou_mat = iou_mat[1:, 1:]

    # Matching via Hungarian algorithm
    if num_matches > 0 and np.max(iou_mat) > th:
        costs = -(iou_mat > th).astype(float) - iou_mat / (2 * num_matches)
        gt_ind, pred_ind = linear_sum_assignment(costs)
        match_ok = iou_mat[gt_ind, pred_ind] > th
        tp = np.count_nonzero(match_ok)
    else:
        tp = 0
    fp = num_pred - tp
    fn = num_gt - tp

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = tp / (tp + fp + fn)
    f1_score = 2 * precision * recall / max(1e-5, precision + recall)

    # Mean IoU
    ious = iou_mat[gt_ind, pred_ind] if num_matches > 0 else []
    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0



    report = rand_voi(gt_labels_rel.astype(np.uint64), pred_labels_rel.astype(np.uint64))

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "mean_iou": mean_iou,
        **report
    }