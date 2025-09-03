from config_unet import model
import torch
from boundary_issues.predict import predict
import sys

if __name__ == "__main__":

    input_shape=(24, 256, 256)
    output_shape= (8, 210, 210)
    neighborhood = [(1, 0 ,0), (0, 1, 0), (0, 0, 1),(2, 0, 0),(0, 5, 0),(0, 0, 5)]

    raw_path = "/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/big_1.zarr/raw"
    output_zarr = "~/test.zarr" # "/mnt/efs/aimbl_2025/student_data/S-EK/EK_transfer/GT_movie1/big_1.zarr/"
    output_dataset = "prediction"
    checkpoint_path = "/mnt/efs/aimbl_2025/boundary_issues/erika/train_1/model_checkpoint_3500"

    #raw_path = sys.argv[1] # path to input raw dataset
    #output_zarr = sys.argv[2] # path to output zarr container
    #output_dataset = sys.argv[3] # name of output dataset inside the zarr container
    #checkpoint_path = sys.argv[4] # path to model checkpoint

    predict(
        model=model.eval(),
        raw_path=raw_path,
        input_shape=input_shape,
        output_shape=output_shape,
        output_zarr=output_zarr,
        output_dataset=output_dataset,
        checkpoint_path=checkpoint_path,
        neighborhood=neighborhood
    )

