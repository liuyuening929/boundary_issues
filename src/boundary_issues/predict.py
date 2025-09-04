import gunpowder as gp
from funlib.persistence import open_ds, prepare_ds #may need to change to ome later
import torch
import numpy as np
import os

import sys
sys.path.append("src")   # add src/ to Python path
from boundary_issues.loss import WeightedLoss


def predict(model: torch.nn.Module,
          raw_path: str,
          input_shape,
          output_shape,
          output_zarr:str,
          output_dataset: str,
          checkpoint_path: str,
          neighborhood = [(1, 0 ,0), (0, 1, 0), (0, 0, 1),(2, 0, 0),(0, 5, 0),(0, 0, 5)],
          ):

    """ predict the model

    Args:
        model: The model to train
        raw_path: Path to the raw data
        input_shape: Input size for the network (voxels)
        output_shape: Output size for the network (voxels)
        checkpoint_path: Path to the model checkpoint
        output_path: Path to save the output
        neighborhood: Neighborhood for the affinities
        
    Returns:
        runs predict function
    """

    # declaring arrays to use in the pipeline
    raw = gp.ArrayKey('RAW')
    prediction = gp.ArrayKey('PREDICT')

    # loading files from paths
    raw_array = open_ds(raw_path)
    pred_array = prepare_ds(
        os.path.join(output_zarr, output_dataset), # Path(output_zarr) / Path(output_dataset)
        (len(neighborhood), *raw_array.shape), 
        offset=raw_array.offset, 
        voxel_size=raw_array.voxel_size,
        axis_names=['c^', *raw_array.axis_names],
        units=raw_array.units,
        dtype=np.float32,
        chunk_shape=(len(neighborhood), *output_shape)
    )

    # creating "pipeline" consisting only of a data source
    source_raw = gp.ArraySource(key= raw, array= raw_array, interpolatable= True)

    ####################
    # SETTING UP NODES #
    ####################

    # Scan node##### Check the parameters here too #####
    scan_request = gp.BatchRequest()
    scan_request.add(raw, gp.Coordinate(input_shape) * raw_array.voxel_size)
    scan_request.add(prediction, gp.Coordinate(output_shape) * raw_array.voxel_size)
    scan_node= gp.Scan(
        scan_request,
        num_workers=1,
    )

    # setting up pad nodes for x and y ### check parameters here too #####
    padding = (gp.Coordinate(input_shape) * raw_array.voxel_size - gp.Coordinate(output_shape) * raw_array.voxel_size) // 2
    pad = gp.Pad(raw, padding)

    # setting up normalization node
    normalization = gp.Normalize(array= raw)

    # ZARR WRITER ########
    zarr_writer = gp.ZarrWrite(
        store = output_zarr,
        dataset_names={prediction: output_dataset},
        dataset_dtypes={prediction: pred_array.dtype},
    )

    predict = gp.torch.Predict(
        model = model,
        inputs = {0:raw},
        outputs = {0: prediction},
        checkpoint = checkpoint_path,
    )


    ######################
    # SETTING UP PIPELINE #
    ######################

    pipeline = source_raw
    pipeline += pad
    pipeline += normalization 
    pipeline += gp.Unsqueeze([raw],0)
    pipeline += gp.Stack(1) 
    pipeline += predict 
    pipeline += gp.Squeeze([prediction],0)
    pipeline += zarr_writer
    pipeline += scan_node    

    # formulating a request for whole array
    request = gp.BatchRequest()

    ################
    # PREDICTION #
    ###############

    # build the pipeline
    with gp.build(pipeline):
        pipeline.request_batch(request)


