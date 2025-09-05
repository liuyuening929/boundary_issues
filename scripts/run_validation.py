# run_validation.py

import os
import zarr
import sys
from boundary_issues.evaluate import evaluate
from pprint import pprint


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_validation_YL.py <gt_path> <mask_path> <pred_path>")
        sys.exit(1)

    gt_path = sys.argv[1]       # Ground truth label zarr dataset path
    mask_path = sys.argv[2]     # Binary mask zarr path
    pred_path = sys.argv[3]     # Predicted segmentation zarr path

    # gt_path = "/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/crop_3_2.zarr/labels"
    # mask_path = "/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/crop_3_2.zarr/mask"
    # pred_path = os.path.expanduser("~/predictions/crop_3_2.zarr/filled_and_dilated.zarr")

    # Load arrays
    gt_np = zarr.open(gt_path, mode='r')[:]
    mask_np = zarr.open(mask_path, mode='r')[:]
    pred_np = zarr.open(pred_path, mode='r')[:]

    # Apply mask and crop prediction to match GT shape
    adapted_pred = pred_np[15:31,:,:] * mask_np

    # Evaluate
    metrics = evaluate(gt_np, adapted_pred)

    # Print all metrics
    pprint(metrics)