from boundary_issues.train import train
from boundary_issues.loss import WeightedLoss
from config_unet import model
import torch

if __name__ == "__main__":
    train(
        model=model,
        loss=WeightedLoss(),
        optimizer= torch.optim.Adam(model.parameters()),
        input_size=(5, 256, 256),
        output_size= (5, 210, 210),
        raw_path="/home/S-YL/boundary_issues/scripts/Data_3D/0426_cross_fov0.zarr/raw",
        labels_path="/home/S-YL/boundary_issues/scripts/Data_3D/0426_cross_fov0.zarr/labels",
        mask_path="/home/S-YL/boundary_issues/scripts/Data_3D/0426_cross_fov0.zarr/labels_mask",
        iterations=10000,
        snapshots_every=500,
        save_every=500,
    )

## run this script from your folder in the boundary_issues from the mnt