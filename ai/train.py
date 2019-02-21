from argparse import ArgumentParser
from importlib import import_module

# Map command line arguments to neural net class properties
map = {
    "boundaries": {
        "filename": "boundariesnn", "classname": "BoundariesNN"
    },
    "interior": {
        "filename": "interiornn", "classname": "InteriorNN"
    }
}

# Parse command line arguments
parser = ArgumentParser(description="Trains the neural nets")
help_str = f'Which neural net to train ({"|".join(k for k in map)})'
parser.add_argument("neural_net", type=str, help=help_str)
parser.add_argument("--stencil", default=-1,
                    help="What stencil size to use for training InteriorNN")
args = parser.parse_args()

if args.stencil == -1 and args.neural_net == "interior":
    raise ValueError("If you want to train InteriorNN you must provide a stencil size")

# Instantiate instance of given neural net class
classname = map[args.neural_net]["classname"]
NeuralNet = getattr(import_module(map[args.neural_net]["filename"]), classname)


# Train neural net
if classname == "BoundariesNN":
    print(f"Training BoundariesNN")
    NeuralNet.train()
else:
    print(f"Training {classname} with {args.stencil}x{args.stencil} stencil")
    NeuralNet.train(int(args.stencil))
