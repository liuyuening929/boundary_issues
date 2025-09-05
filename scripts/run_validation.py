# %%

import os
import zarr
from boundary_issues.evaluate import evaluate as evaluate_bi
import sys
import numpy as np

from funlib import evaluate as evaluate_fl

def run_val_no_mask(prediction_arr, gt_arr):
    """
    Fill interior gaps in segmented volumes.

    Args:
        prediction_arr (str): Path to the pred zarr array.
        gt_arr (str): Path to the gt zarr array.

    Returns:
        precision, recall, accuracy: Evaluation metrics.
    """
     # set up arrays
    pred_zarr = zarr.open(pred_path, mode='r')
    pred_np_array = pred_zarr[:]
    
    gt_zarr = zarr.open(gt_path, mode='r')
    gt_np_array = gt_zarr[:]
    
    precision, recall, accuracy = evaluate_bi(gt_np_array, pred_np_array)

    #####
    # Also check voi
    pred_np_array = pred_np_array.astype(np.uint64)
    gt_np_array = gt_np_array.astype(np.uint64)
    m = evaluate_fl.rand_voi(pred_np_array, gt_np_array)
    voi_merge = m['voi_merge']
    voi_split = m['voi_split']

    #####

    print(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}, VOI merge: {voi_merge}, VOI split: {voi_split}")


def run_val_maskcrop(prediction_arr, gt_arr, mask_arr, offset = (15,31)):
    """
    Fill interior gaps in segmented volumes.

    Args:
        prediction_arr (str): Path to the pred zarr array.
        gt_arr (str): Path to the gt zarr array.
        mask_arr (str): Path to the mask zarr array.

    Returns:
        precision, recall, accuracy: Evaluation metrics.
    """
    # set up arrays
    pred_zarr = zarr.open(pred_path, mode='r')
    pred_np_array = pred_zarr[:]
    
    gt_zarr = zarr.open(gt_path, mode='r')
    gt_np_array = gt_zarr[:]

    mask_zarr = zarr.open(mask_path, mode='r')
    mask_np_array = mask_zarr[:]

    # OPTIONAL STEP: crop and mask gt array to match prediction array shape
    adapted_pred = pred_np_array[offset[0]:offset[1], :, :]
    adapted_pred = adapted_pred * mask_np_array
    ##########
    
    precision, recall, accuracy = evaluate_bi(gt_np_array, adapted_pred)

    #####
    # Also check voi
    adapted_pred = adapted_pred.astype(np.uint64)
    gt_np_array = gt_np_array.astype(np.uint64)

    m = evaluate_fl.rand_voi(adapted_pred, gt_np_array)
    voi_merge = m['voi_merge']
    voi_split = m['voi_split']

    #####

    print(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}, VOI merge: {voi_merge}, VOI split: {voi_split}")



if __name__ == "__main__":

    # # get paths from command line arguments
    # pred_path = sys.argv[1] # path to predicted zarr store
    # gt_path = sys.argv[2] # path to GT label array
    # mask_path = sys.argv[3] # path to GT mask path

    gt_path = "/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/crop_3_2.zarr/labels"
    mask_path = "/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/crop_3_2.zarr/mask"
    pred_path = os.path.expanduser("~/predictions/crop_3_2.zarr/_filled_lblock")

    # run_val_no_mask(pred_path, gt_path)
    run_val_maskcrop(pred_path, gt_path, mask_path, offset=(15,31))


# %%
