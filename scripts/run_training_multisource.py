from boundary_issues.train import train
from boundary_issues.loss import WeightedLoss
from config_unet import model
import torch
import glob

if __name__ == "__main__":
    train(
        model=model,
        loss=WeightedLoss(),
        optimizer= torch.optim.Adam(model.parameters()),
        input_size=(32, 512, 512),
        output_size= (16, 466, 466),
        zarr_roots = glob.glob("/mnt/efs/aimbl_2025/student_data/S-MC/Chlebowski_Mady_AIMBL2025/compiled_neuronal_em*.zarr"),
        iterations=2501,
        snapshots_every=250,
        save_every=250,
    )

## run this script from your folder in the boundary_issues from the mnt