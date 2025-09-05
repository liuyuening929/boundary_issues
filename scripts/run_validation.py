# %%

import os
import zarr
from boundary_issues.evaluate import evaluate
import sys

from funlib import evaluate

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

    
    precision, recall, accuracy = evaluate(gt_np_array, pred_np_array)
    print(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")


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
    adapted_pred = pred_np_array[offset[0]:offset[1], :, :]*mask_np_array
    ##########
    
    precision, recall, accuracy = evaluate(gt_np_array, adapted_pred)

    #####
    # Also check voi
    m = evaluate.rand_voi(adapted_pred, gt_np_array)
    voi_merge = m['voi_merge']
    voi_split = m['voi_split']
    voi_total = m['voi_total']

    #####

    print(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}, VOI merge: {voi_merge}, VOI split: {voi_split}, VOI total: {voi_total}")



if __name__ == "__main__":

    # get paths from command line arguments
    pred_path = sys.argv[1] # path to predicted zarr store
    gt_path = sys.argv[2] # path to GT label array
    mask_path = sys.argv[3] # path to GT mask path

    # gt_path = "/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/crop_3_2.zarr/labels"
    # mask_path = "/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/crop_3_2.zarr/mask"
    # pred_path = os.path.expanduser("~/predictions/crop_3_2.zarr/_filled_lblock")

    # run_val_no_mask(pred_path, gt_path)
    run_val_maskcrop(pred_path, gt_path, mask_path, crop=(15,31))


# %%
