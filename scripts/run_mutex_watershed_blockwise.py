from funlib.geometry import Coordinate
from funlib.persistence import open_ds
from pathlib import Path
from volara.blockwise import ExtractFrags, AffAgglom, GraphMWS, Relabel
from volara.datasets import Affs, Labels
from volara.dbs import SQLite
from volara.lut import LUT
import sys

if __name__ == "__main__":

    file = Path(sys.argv[1]) # path to our zarr store. blah/blah.zarr
    affs_dataset = sys.argv[2] # name of affinities within zarr store. for example 'affinities'
    output_seg = sys.argv[3] # name of output seg dataset within zarr store

    num_workers=8

    affs_ds = open_ds(file / affs_dataset)

    block_shape = Coordinate(affs_ds.chunk_shape[1:]) // 2
    context = Coordinate(affs_ds.chunk_shape[1:]) // 6
    print(block_shape, context)

    bias = [-0.2, -0.8]

    affs = Affs(
        store=file / affs_dataset,
        neighborhood=[
            Coordinate(1, 0, 0),
            Coordinate(0, 1, 0),
            Coordinate(0, 0, 1),
            Coordinate(2, 0, 0),
            Coordinate(0, 5, 0),
            Coordinate(0, 0, 5),
        ],
    )

    db = SQLite(
        path=file / "db.sqlite",
        edge_attrs={
            "adj_weight": "float",
            "lr_weight": "float",
        },
    )

    fragments = Labels(store=file / "fragments")

    extract_frags = ExtractFrags(
        db=db,
        affs_data=affs,
        frags_data=fragments,
        block_size=block_shape,
        context=context,
        noise_eps=0.01,
        remove_debris=20,
        bias=[bias[0]] * 3 + [bias[1]] * 3,
        num_workers=num_workers,
    )

    aff_agglom = AffAgglom(
        db=db,
        affs_data=affs,
        frags_data=fragments,
        block_size=block_shape,
        context=context,
        scores={"adj_weight": affs.neighborhood[0:3], "lr_weight": affs.neighborhood[3:]},
        num_workers=num_workers,
    )

    lut = LUT(path=file / "lut.npz")
    roi = open_ds(file / affs_dataset).roi

    global_mws = GraphMWS(
        db=db,
        lut=lut,
        weights={"adj_weight": (1.0, bias[0]), "lr_weight": (1.0, bias[1])},
        roi=[roi.get_begin(), roi.get_shape()],
    )

    relabel = Relabel(
        frags_data=fragments,
        seg_data=Labels(store=file / output_seg),
        lut=lut,
        block_size=block_shape,
        num_workers=num_workers*2,
    )

    pipeline = extract_frags + aff_agglom + global_mws + relabel

    pipeline.run_blockwise()