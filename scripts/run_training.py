from boundary_issues.train import train
from boundary_issues.loss import WeightedLoss
from config_unet import model
import torch

if __name__ == "__main__":
    train(
        model=model,
        loss=WeightedLoss(),
        optimizer= torch.optim.Adam(model.parameters()),
        input_size=(24, 256, 256),
        output_size= (8, 210, 210),
        raw_path="/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/crop_1_1.zarr/raw",
        labels_path="/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/crop_1_1.zarr/labels",
        mask_path="/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/crop_1_1.zarr/mask",
        iterations=5001,
        snapshots_every=500,
        save_every=500,
    )

## run this script from your folder in the boundary_issues from the mnt