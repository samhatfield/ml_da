from argparse import ArgumentParser
from importlib import import_module

# Map command line arguments to neural net class properties
map = {
    "boundaries": {
        "filename": "boundariesnn", "classname": "BoundariesNN"
    },
    "three_by_three": {
        "filename": "threebythreenn", "classname": "ThreeByThreeNN"
    }
}

# Parse command line arguments
parser = ArgumentParser(description="Trains the neural nets")
parser.add_argument("neural_net", type=str, help="Which neural net to train")
args = parser.parse_args()

# Instantiate instance of given neural net class
classname = map[args.neural_net]["classname"]
NeuralNet = getattr(import_module(map[args.neural_net]["filename"]), classname)
print(f"Training {classname}")

# Train neural net
NeuralNet.train()