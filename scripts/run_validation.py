
import os
import zarr
from boundary_issues.evaluate import evaluate
import sys

if __name__ == "__main__":

    # get paths from command line arguments
    gt_path = sys.argv[1] # path to GT label array
    mask_path = sys.argv[2] # path to predicted zarr store
    pred_path = sys.argv[3] # path to predicted zarr store
    
    # gt_path = "/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/crop_3_1.zarr/labels"
    # mask_path = "/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/crop_3_1.zarr/mask"
    # pred_path = os.path.expanduser("~/predictions/crop_3_1")

    # set up arrays
    gt_zarr = zarr.open(gt_path, mode='r')
    gt_np_array = gt_zarr[:]

    mask_zarr = zarr.open(mask_path, mode='r')
    mask_np_array = mask_zarr[:]

    pred_zarr = zarr.open(pred_path, mode='r')
    pred_np_array = pred_zarr[:]

    # OPTIONAL STEP: crop and mask gt array to match prediction array shape
    adapted_pred = pred_np_array[15:31, :, :]*mask_np_array
    ##########

    precision, recall, accuracy = evaluate(gt_np_array, adapted_pred)
