import gunpowder as gp
from funlib.persistence import open_ds #may need to change to ome later
import torch
import numpy as np

import sys
sys.path.append("src")   # add src/ to Python path
from boundary_issues.loss import WeightedLoss

def train(model: torch.nn.Module,
          loss: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          input_size,
          output_size,
          zarr_roots,
          iterations = 5000,
          batch_size = 10,
          neighborhood = [(1, 0 ,0), (0, 1, 0), (0, 0, 1),(2, 0, 0),(0, 5, 0),(0, 0, 5)],
          snapshots_every = 1000,
          save_every = 1000,
          deform_pt_space = (30, 30),
          jitter_sigma = (2, 2),
          percent_covered = 0.8,
          ):

    """ train the model

    Args:
        model: The model to train
        loss: The loss function to use
        optimizer: The optimizer to use
        zarr_roots: Path to zarr directories or list of paths
        input_size: Input size for the network (voxels)
        output_size: Output size for the network (voxels)
        iterations: Number of training iterations
        batch_size: Batch size for training
        neighborhood: Neighborhood configuration for affinity calculation
        snapshot_every: Frequency of snapshots
        save_every: Frequency of model saves
        deform_pt_space: Point spacing for deformation augmentation (voxels)
        jitter_sigma: Standard deviation for Gaussian jitter (voxels)
        percent_covered: Minimum percent of the input that must be covered by the mask

    Returns:
        runs train function
    """

    # declaring arrays to use in the pipeline
    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    mask = gp.ArrayKey('MASK')
    gt_affs = gp.ArrayKey('GT_AFFS')
    gt_affs_mask = gp.ArrayKey('GT_AFFS_MASK')
    prediction = gp.ArrayKey('PREDICT')
    aff_scale= gp.ArrayKey("SCALE")

    ####################
    # SETTING UP NODES #
    ####################
    #This is where we set up the mutiple inputs for the pipeline
    merged_src=[]
    for zarr in zarr_roots:
        zarr_raw = open_ds(f"{zarr}/raw") ##change name if needed
        zarr_labels = open_ds(f"{zarr}/labels") ##change name if needed
        zarr_mask = open_ds(f"{zarr}/labels_mask") ##change name if needed
        source_raw = gp.ArraySource( key= raw, array= zarr_raw, interpolatable= True)
        source_labels = gp.ArraySource( key= labels, array= zarr_labels, interpolatable= False)
        source_mask = gp.ArraySource( key= mask, array= zarr_mask, interpolatable= False)
        padding_amount = max(t[0] for t in neighborhood)*zarr_raw.voxel_size[0]

        merged_sources = (source_raw , source_labels , source_mask) + gp.MergeProvider()+ gp.Pad(labels, size= (padding_amount,0,0)) + gp.Pad(mask, size= (padding_amount,0,0)) + gp.RandomLocation()

        merged_src.append(merged_sources)
    merged_src = tuple(merged_src)
    print("merged_src done!")
    # setting up normalization node
    normalization = gp.Normalize(array= raw)

    # setting up deformation node
    deform_pt_space_wc = gp.Coordinate(deform_pt_space) * gp.Coordinate(zarr_raw.voxel_size[1:])
    jitter_sigma_space_wc = gp.Coordinate(jitter_sigma) * gp.Coordinate(zarr_raw.voxel_size[1:])
    deform = gp.DeformAugment(deform_pt_space_wc, jitter_sigma_space_wc, spatial_dims=2)

    # setting up affinities node
    add_affs = gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        unlabelled=mask,
        affinities_mask=gt_affs_mask
    )
    
    # setting up intensity node
    intensity_augment = gp.IntensityAugment(array=raw, scale_min=0.9, scale_max=1.1, shift_min=-0.1, shift_max=0.1, z_section_wise=False, clip=True) # Please set z_section_wise according to your data set

    # setting up noise node
    # noise = gp.NoiseAugment(array=raw, mode='Gaussian', clip=True)

    # setting up snapshot node
    snapshot = gp.Snapshot(every = snapshots_every,
        dataset_names={raw:"raw", labels:"labels", mask:"mask", gt_affs: "affs", gt_affs_mask: "affS_mask",prediction:"prediction", aff_scale:"scale"},
        dataset_dtypes={gt_affs: 'float32'})

    train = gp.torch.Train(
        model = model,
        loss = loss,
        optimizer = optimizer,
        inputs = {0:raw},
        loss_inputs = {0: prediction, 1: gt_affs, 2: aff_scale},
        outputs = {0: prediction},
        save_every = save_every,
        log_dir = "training_logs"
    )

    # setting up balanced labels node
    balanced_labels=gp.BalanceLabels(gt_affs,scales=aff_scale, mask= gt_affs_mask)
        

    ######################
    # SETTING UP PIPELINE #
    ######################
    # print("beofre pipeline")
    pipeline = merged_src
    # print("pipeline merge_src done!")

    pipeline += gp.RandomProvider()
    # print("pipeline random provider  done!")

    pipeline += normalization 
    pipeline += gp.SimpleAugment(transpose_only=[1,2], mirror_only=[1,2]) 
    pipeline += deform 
    pipeline += intensity_augment
    # print("pipeline intensity augment  added!")
    # pipeline += noise
    pipeline += gp.PreCache(cache_size=32, num_workers=8)
    pipeline += add_affs
    # print("pipeline add affinity done!")

    pipeline += balanced_labels 
    pipeline += gp.Unsqueeze([raw],0)
    pipeline += gp.Stack(batch_size) 
    # print("pipeline before add train!")
    pipeline += gp.PrintProfilingStats(every=1)
    pipeline += train 
    # print("pipeline after add train!")

    # pipeline += gp.Squeeze([raw,labels,mask, gt_affs,gt_affs_mask,prediction,aff_scale],0)
    pipeline += snapshot
    

    # formulating a request for "raw"
    input_size = gp.Coordinate(input_size) * zarr_raw.voxel_size
    output_size = gp.Coordinate(output_size) * zarr_raw.voxel_size
    print(f"Input size: {input_size}, Output size: {output_size}") 
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(mask, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(prediction, output_size)
    request.add(aff_scale,output_size)

    ################
    # TRAINING!!!! #
    ###############
    
    print("Training for", iterations, "iterations")

    # build the pipeline...
    with gp.build(pipeline):
        for i in range(iterations):
            # ...and request a batch
            print(".", end="")
            print(f"Requesting batch at iteration {i}")
            batch = pipeline.request_batch(request)
            print(f"Batch received at iteration {i}")

            if i%50==0:
                print(f"Iteration {i}: {batch.loss}")


