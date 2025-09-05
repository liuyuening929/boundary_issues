# from funlib.learn.torch.models.unet import UNet
from boundary_issues.model import UNet
import torch
import numpy as np

model = torch.nn.Sequential(UNet(
    in_channels=1,
    num_fmaps=16,
    fmap_inc_factor=3,
    downsample_factors=[
        [1, 2, 2],  
        [1, 2, 2]
    ],
    kernel_size_down=[[(1, 3, 3), (1, 3, 3)],[(1, 3, 3), (1, 3, 3)],[(3, 3, 3), (3, 3, 3)]],
    kernel_size_up=[[(3, 3, 3), (3, 3, 3),(3, 3, 3)],[(3, 3, 3), (3, 3, 3), (3, 3, 3)]],
    padding=("valid", "valid", "valid"),
    voxel_size=(300, 108, 108),
    fov=(1, 1, 1),  
    num_fmaps_out=None,
    constant_upsample=True
), torch.nn.Conv3d(in_channels = 16, out_channels= 6, kernel_size=(1,1,1)),torch.nn.Sigmoid())


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model= model.to(device)
#
# Create a dummy input array with the shape (1, 1, 1000, 170, 170)
# dummy_input = np.ones((2, 1, 16, 196,196 ), dtype=np.float32)
# dummy_input = torch.from_numpy(dummy_input).to(device)
# dummy_label = np.ones((2, 3, 16, 150,150 ), dtype=np.float32)
# dummy_label = torch.from_numpy(dummy_label).to(device)

# print(dummy_input.shape)
# print(dummy_label.shape)gi

# print(next(model.parameters()).device)

# loss_function: torch.nn.Module = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# for i in range(10):

#     dummy_input, dummy_label = dummy_input.to(device), dummy_label.to(device)
#     optimizer.zero_grad()

#     prediction = model(dummy_input)

#     loss = loss_function(prediction, dummy_label)

#     loss.backward()
#     optimizer.step()


# # Test the model with the dummy input
# # Assuming `model` and `dummy_input_tensor` are already defined
# # for i in range(10):
# #     output = model(dummy_input)
# #     print(f"Iteration {i + 1}: Output shape = {output.shape}")
