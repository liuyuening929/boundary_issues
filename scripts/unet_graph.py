from config_unet import model
import click
import torchview

@click.command()
@click.argument('input_size')
def main(input_size):
    # Parse input_size as a tuple of integers, e.g. "1,3,256,256"
    input_size_tuple = tuple(int(x) for x in input_size.split(','))
    graph = torchview.draw_graph(model, input_size=input_size_tuple, device="meta", save_graph=True)
    

if __name__ == "__main__":
    main()