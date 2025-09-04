import click
import numpy as np
import daisy
from funlib.persistence import open_ds, prepare_ds
from functools import partial
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fill_holes_blockwise(in_ds, out_ds, block):
    import numpy as np
    from funlib.persistence import Array
    import fastmorph
    import cc3d

    # read block
    logger.info(f"Processing raw mask for block: {block}")
    in_data = in_ds.to_ndarray(block.read_roi)

    # process
    out_data = in_data.copy()

    # # section-wise, unique label-wise
    # for u, mask in cc3d.each(out_data, binary=True):
    #     if u == 0:
    #         continue
    #     for z in range(mask.shape[0]):
    #         section = mask[z]

    #         closed = fastmorph.spherical_dilate(section, parallel=2, radius=2)
    #         closed = fill_voids.fill(closed)
    #         closed = fastmorph.spherical_erode(closed, parallel=2, radius=4)#[1:-1, 1:-1]
    #         out_data[z][closed] = u

    # # full block
    #out_data = fastmorph.fill_holes(out_data, remove_enclosed=True, fix_borders=True, morphological_closing=True)

    # # full block (which is full section), but stack so that its 3d
    # Stack the block 3 times along the z-axis, fill holes in 3D, then extract the middle slice
    out_data = np.repeat(out_data, 3, axis=0)
    out_data = fastmorph.fill_holes(out_data, remove_enclosed=True, fix_borders=True, morphological_closing=True)
    out_data = out_data[1]  # keep middle slice, preserve 3D shape

    for u, mask in cc3d.each(out_data, binary=True):
        if u == 0:
            continue
        closed = fastmorph.spherical_close(mask, parallel=2, radius=2)
        out_data[closed] = u

    out_data = np.expand_dims(out_data, axis=0)  # add back the z-axis

    # # full block, unique label-wise
    # for u, mask in cc3d.each(out_data, binary=True):
    #     if u == 0:
    #         continue
    #     filled = fastmorph.fill_holes(mask, fix_borders=True, remove_enclosed=True, morphological_closing=True)
    #     out_data[filled] = u

    # write block
    try:
        out_array = Array(out_data, block.read_roi.offset, out_ds.voxel_size)
        out_ds[block.write_roi] = out_array.to_ndarray(block.write_roi)
    except Exception as e:
        logger.error(f"Failed to write to {block.write_roi}: {str(e)}")
        raise

    return 0


@click.command()
@click.option(
    "--in_array",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="The path of the input zarr array",
)
@click.option(
    "--out_array",
    "-o",
    type=click.Path(),
    help="The path of the output mask zarr array",
)
@click.option(
    "--num_workers",
    "-w",
    type=int,
    default=40,
    help="Number of workers for parallel processing",
)
def fill_holes(in_array, out_array, num_workers):
    """
    Fill interior gaps in segmented volumes.

    Args:
        in_array (str): Path to the input zarr array.
        out_array (str): Path to the output zarr array.
        num_workers (int): Number of parallel workers.

    Returns:
        str: Path to the output array.
    """

    # open
    in_ds = open_ds(in_array)

    # prepare
    dims = in_ds.roi.dims
    block_size = in_ds.chunk_shape * in_ds.voxel_size # this does it block by block
    #block_size = daisy.Coordinate((1, *in_ds.shape[1:])) * in_ds.voxel_size #this does it sectiuon by section
    #block_size = daisy.Coordinate([in_ds.voxel_size[0], *in_ds.roi.shape[-2:]])
    context = daisy.Coordinate((0,0,0)) * in_ds.voxel_size
    write_block_roi = daisy.Roi((0,) * dims, block_size)
    read_block_roi = write_block_roi.grow(context, context)

    if out_array is None:
        in_f, in_ds_name = in_array.split(".zarr/")
        out_ds =  in_ds_name + "_filled"
        out_array = f"{in_f}.zarr/{out_ds}"

    print(f"Writing mask to {out_array}")
    out_ds = prepare_ds(
        out_array,
        shape=in_ds.roi.shape / in_ds.voxel_size,
        offset=in_ds.roi.offset,
        voxel_size=in_ds.voxel_size,
        axis_names=in_ds.axis_names,
        units=in_ds.units,
        dtype=in_ds.dtype,
        chunk_shape=daisy.Coordinate((1, *in_ds.shape[1:]))#in_ds.chunk_shape,
    )

    # run
    task = daisy.Task(
        f"FillHolesTask",
        out_ds.roi.grow(context, context),
        read_block_roi,
        write_block_roi,
        process_function=partial(
            fill_holes_blockwise, in_ds, out_ds
        ),
        read_write_conflict=True,
        num_workers=num_workers,
        max_retries=0,
        fit="shrink",
    )

    ret = daisy.run_blockwise([task])

    if ret:
        logger.info("Ran all blocks successfully!")
    else:
        logger.error("Did not run all blocks successfully...")

    return out_array


if __name__ == "__main__":
    fill_holes()
