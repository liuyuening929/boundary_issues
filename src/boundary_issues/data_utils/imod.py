import click
import pandas as pd

def split_objects(input_mod_txt, output_mod_txt):
    # Read input txt file into DataFrame
    df = pd.read_csv(input_mod_txt, sep=r'\s+', header=None)
    # Expect columns: object_id, contour_id, x, y, z
    df.columns = ["object_id", "contour_id", "x", "y", "z"]
    df["object_id"] = df["contour_id"].copy()
    df["contour_id"] = 1
    df.to_csv(output_mod_txt, sep=" ", header=False, index=False, float_format="%.2f")

@click.command()
@click.argument("input_mod_txt", type=click.Path(exists=True))
@click.argument("output_mod_txt", type=click.Path())
def split(input_mod_txt, output_mod_txt):
    split_objects(input_mod_txt, output_mod_txt)
