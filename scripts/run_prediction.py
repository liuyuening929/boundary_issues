from config_unet import model
import torch
from boundary_issues.predict import predict
import sys

if __name__ == "__main__":

    input_size=(16, 256, 256)
    output_size= (16, 210, 210)
    neighborhood = [(1, 0 ,0), (0, 1, 0), (0, 0, 1),(2, 0, 0),(0, 5, 0),(0, 0, 5)]

    raw_path = sys.argv[1] # path to input raw dataset
    output_zarr = sys.argv[2] # path to output zarr container
    output_dataset = sys.argv[3] # name of output dataset inside the zarr container
    checkpoint_path = sys.argv[4] # path to model checkpoint

    predict(
        model=model,
        raw_path=raw_path,
        input_size=input_size,
        output_size=output_size,
        output_zarr=output_zarr,
        output_dataset=output_dataset,
        checkpoint_path=checkpoint_path,
        neighborhood=neighborhood
    )

