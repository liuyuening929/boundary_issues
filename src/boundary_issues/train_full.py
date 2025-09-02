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
          raw_path: str,
          labels_path: str,
        #   mask_path: str,
          input_size,
          output_size,
          iterations = 5000,
          batch_size = 1,
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
        raw_path: Path to the raw data
        labels_path: Path to the labels
        mask_path: Path to the mask
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
    # mask = gp.ArrayKey('MASK')
    gt_affs = gp.ArrayKey('GT_AFFS')
    # gt_affs_mask = gp.ArrayKey('GT_AFFS_MASK')
    prediction = gp.ArrayKey('PREDICT')
    aff_scale= gp.ArrayKey("SCALE")

    # loading files from paths
    raw_array = open_ds(raw_path)
    labels_array = open_ds(labels_path)
    # mask_array = open_ds(mask_path)

    # creating "pipeline" consisting only of a data source
    source_raw = gp.ArraySource( key= raw, array= raw_array, interpolatable= True)
    source_labels = gp.ArraySource( key= labels, array= labels_array, interpolatable= False)
    # source_mask = gp.ArraySource( key= mask, array= mask_array, interpolatable= False)

    ####################
    # SETTING UP NODES #
    ####################

    # setting up random location nodes
    random_location = gp.RandomLocation()#min_masked=percent_covered,mask=mask)

    # setting up pad nodes
    padding_amount = max(t[0] for t in neighborhood)*raw_array.voxel_size[0]
    pad_label=gp.Pad(labels, size= (padding_amount,0,0))
    # pad_mask=gp.Pad(mask, size= (padding_amount,0,0))

    # setting up normalization node
    normalization = gp.Normalize(array= raw)

    # setting up deformation node
    deform_pt_space_wc = gp.Coordinate(deform_pt_space) * gp.Coordinate(raw_array.voxel_size[1:])
    jitter_sigma_space_wc = gp.Coordinate(jitter_sigma) * gp.Coordinate(raw_array.voxel_size[1:])
    deform = gp.DeformAugment(deform_pt_space_wc, jitter_sigma_space_wc, spatial_dims=2)

    # setting up affinities node
    add_affs = gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        # unlabelled=mask,
        # affinities_mask=gt_affs_mask
    )
    
    # setting up intensity node
    intensity_augment = gp.IntensityAugment(array=raw, scale_min=0.9, scale_max=1.1, shift_min=-0.1, shift_max=0.1, z_section_wise=False, clip=True) # Please set z_section_wise according to your data set

    # setting up noise node
    noise = gp.NoiseAugment(array=raw, mode='Gaussian', clip=True)

    # setting up snapshot node
    snapshot = gp.Snapshot(every = snapshots_every,
        dataset_names={
            raw:"raw", 
            labels:"labels", 
            gt_affs: "affs", 
            # mask:"mask", 
            # gt_affs_mask: "affS_mask",
            prediction:"prediction", 
            aff_scale:"scale"
        },
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
    balanced_labels=gp.BalanceLabels(gt_affs,scales=aff_scale)# mask= gt_affs_mask)
        

    ######################
    # SETTING UP PIPELINE #
    ######################

    pipeline = (
        source_raw,
        source_labels,
        #source_mask
    ) + gp.MergeProvider() 
    pipeline += pad_label
   # pipeline += pad_mask
    pipeline += random_location 
    pipeline += normalization 
    pipeline += gp.SimpleAugment(transpose_only=[1,2]) 
    pipeline += deform 
    pipeline += intensity_augment
    pipeline += noise
    pipeline += add_affs
    pipeline += balanced_labels 
    # pipeline += gp.Unsqueeze([raw],0)
    pipeline += gp.Stack(batch_size) 
    pipeline += gp.PreCache()
    pipeline += train 
    # pipeline += gp.Squeeze([raw,labels,mask, gt_affs,gt_affs_mask,prediction,aff_scale],0)
    pipeline += gp.Squeeze([raw,labels,gt_affs,prediction,aff_scale],0)

    pipeline += snapshot
    

    # formulating a request for "raw"
    input_size = gp.Coordinate(input_size) * raw_array.voxel_size
    output_size = gp.Coordinate(output_size) * raw_array.voxel_size

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(labels, output_size)
    # request.add(mask, output_size)
    request.add(gt_affs, output_size)
    # request.add(gt_affs_mask, output_size)
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
            # print(".", end="")
            batch = pipeline.request_batch(request)
            print(f"Iteration {i}: {batch.loss}")


