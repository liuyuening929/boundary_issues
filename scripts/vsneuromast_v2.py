# %% Imports and paths
from pathlib import Path

from iohub import open_ome_zarr
from viscy.data.hcs import HCSDataModule

# Viscy classes for the trainer and model
from viscy.translation.engine import VSUNet
from viscy.translation.predict_writer import HCSPredictionWriter
from viscy.trainer import VisCyTrainer
from viscy.transforms import NormalizeSampled
from plot import plot_vs_n_fluor
# %%
if __name__ == "__main__":
    # TODO: modify the path to the downloaded dataset
    input_data_path = "/mnt/efs/aimbl_2025/student_data/S-MC/Chlebowski_Mady_AIMBL2025/250627_4dpf_zarrreconstruction/reconstruction_250627_93a_4dpf_nm4_em1_001_WellA01_ChannelX_Seq0000.zarr"
    # input_data_path = "/mnt/efs/aimbl_2025/S-MC/neuromast_demo/20230803_fish2_60x_1_cropped_zyx_resampled_clipped_2.zarr"
    # TODO: modify the path to the downloaded checkpoint
    model_ckpt_path = "/mnt/efs/aimbl_2025/TA-EH/epoch=1-step=1012.ckpt"

    # TODO: modify the path
    # Zarr store to save the predictions
    output_path = "/mnt/efs/aimbl_2025/S-MC/neuromast_demo/250627_4dpf_em1_vsneuromastV2.zarr"

    # TODO: Choose an FOV
    # FOV of interest
    fov = "A/1/0"
    # fov = "0/3/0"

    input_data_path = Path(input_data_path) / fov
    # %%
    # Create the VSNeuromast model

    # Reduce the batch size if encountering out-of-memory errors
    BATCH_SIZE = 1
    # NOTE: Set the number of workers to 0 for Windows and macOS
    # since multiprocessing only works with a
    # `if __name__ == '__main__':` guard.
    # On Linux, set it to the number of CPU cores to maximize performance.
    NUM_WORKERS = 8
    phase_channel_name = "Phase3D"

    #%%
    import matplotlib.pyplot as plt
    with open_ome_zarr(input_data_path, mode="r") as dataset:
        print(dataset.data.shape)
    plt.imshow(dataset.data[:][0, 0, 60])
    plt.show()
    # %%
    # Setup the data module.
    data_module = HCSDataModule(
        data_path=input_data_path,
        source_channel=[phase_channel_name],
        target_channel=["Nuclei","Membrane"],
        z_window_size=21,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalizations=[
          NormalizeSampled(
             [phase_channel_name],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
        ],
        persistent_workers=False,

    )
    # %%
    # Setup the model.
    # Dictionary that specifies key parameters of the model.
    config_VSNeuromast = {
        "in_channels": 1,
        "out_channels": 2,
        "in_stack_depth": 21,
        "stem_kernel_size": (7, 4, 4),
        "encoder_blocks": [3, 3, 9, 3],
        "dims": [96, 192, 384, 768],
        "encoder_drop_path_rate": 0.0,
        "decoder_conv_blocks": 2,
        "pretraining": False,
        "head_conv": True,
        "head_conv_expansion_ratio": 4,
        "head_conv_pool": False,
    }

    model_VSNeuromast = VSUNet.load_from_checkpoint(
        model_ckpt_path, architecture="fcmae", model_config=config_VSNeuromast
    )
    model_VSNeuromast.eval()

    # %%
    # Setup the Trainer
    trainer = VisCyTrainer(
        accelerator="gpu",
        callbacks=[HCSPredictionWriter(output_path)],
        logger = False,
    )

    # Start the predictions
    trainer.predict(
        model=model_VSNeuromast,
        datamodule=data_module,
        return_predictions=False,
    )

    # %%
    # Open the output_zarr store and inspect the output
    # Show the individual channels and the fused in a 1x3 plot
    output_path = Path(output_path) / fov
    output_path.exists()
    # %%
    # Open the predicted data
    vs_store = open_ome_zarr(output_path, mode="r")
    T, C, Z, Y, X = vs_store.data.shape
    print(f"Data shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    crop_yx=800
    Y_slice = slice(Y//2-crop_yx//2,Y//2+crop_yx//2)
    X_slice = slice(X//2-crop_yx//2,X//2+crop_yx//2)
    # Get a z-slice
    z_slice = Z // 2  # NOTE: using the middle slice of the stack. Change as needed.
    vs_nucleus = vs_store[0][0, 0, z_slice,Y_slice,X_slice]  # (t,c,z,y,x)
    vs_membrane = vs_store[0][0, 1, z_slice,Y_slice,X_slice]  # (t,c,z,y,x)

    # Open the experimental fluorescence
    fluor_store = open_ome_zarr(input_data_path, mode="r")
    # Get the 2D images
    # NOTE: Channel indeces hardcoded for this dataset
    phase_img   = fluor_store[0][0, 0, z_slice,Y_slice,X_slice]  # (t,c,z,y,x)
    # fluor_nucleus = fluor_store[0][0, 1, z_slice]  # (t,c,z,y,x)
    # fluor_membrane = fluor_store[0][0, 2, z_slice]  # (t,c,z,y,x)

    # Plot
    plot_vs_n_fluor(vs_nucleus, vs_membrane, phase_img,phase_img)

    vs_store.close()
    fluor_store.close()
# %%
