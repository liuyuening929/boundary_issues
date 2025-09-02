import sys
import click
import pandas as pd


def split_objects(input_mod_txt, output_mod_txt):
    # Read input txt file into DataFrame
    df = pd.read_csv(input_mod_txt, delim_whitespace=True, header=None)
    # Expect columns: object_id, contour_id, x, y, z
    df.columns = ["object_id", "contour_id", "x", "y", "z"]

    # Find all unique contour_ids for object_id==1
    contours = df[df["object_id"] == 1]["contour_id"].unique()
    # Map each contour to a new object_id (starting from 1)
    contour_map = {cid: i+1 for i, cid in enumerate(contours)}

    # Update object_id for each row
    df["object_id"] = df.apply(lambda row: contour_map[row["contour_id"]] if row["object_id"] == 1 else row["object_id"], axis=1)
    # Also set contour_id = object_id for all rows
    df["contour_id"] = df["object_id"]

    # Write to output file, space-separated, no header, no index
    df.to_csv(output_mod_txt, sep=" ", header=False, index=False, float_format="%.2f")

@click.command()
@click.argument("input_mod_txt", type=click.Path(exists=True))
@click.argument("output_mod_txt", type=click.Path())
def split(input_mod_txt, output_mod_txt):
    split_objects(input_mod_txt, output_mod_txt)
