from boundary_issues.train_multisource import train
from boundary_issues.loss import WeightedLoss
from config_unet import model
import torch
import glob

if __name__ == "__main__":
    train(
        model=model,
        loss=WeightedLoss(),
        optimizer= torch.optim.Adam(model.parameters()),
        input_size=(5, 480, 480),
        output_size= (5, 434, 434),
        zarr_roots = glob.glob("/mnt/efs/aimbl_2025/student_data/S-YL/Data_3D/*.zarr"),
        iterations=10000,
        snapshots_every=100,
        save_every=100,
    )

## run this script from your folder in the boundary_issues from the mnt