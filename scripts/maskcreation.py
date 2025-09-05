## mask creation for training
import zarr
import pathlib
import numpy as np
print(zarr.__version__)

print("Processing files")
group_path=pathlib.Path(f"/mnt/efs/aimbl_2025/student_data/S-MC/Chlebowski_Mady_AIMBL2025/em4_haircell_label.zarr")
mask_path= group_path / "mask"
labels_path= group_path / "label"

group = zarr.group(store=group_path, overwrite=False)


label = zarr.open(labels_path)
print(label)
print(label.shape)

mask = zarr.creation.ones_like(label, store=mask_path)
for i in label.attrs.keys():
    mask.attrs[i] = label.attrs[i]