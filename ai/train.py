from argparse import ArgumentParser
from importlib import import_module
from iris import load_cubes

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
help_str = f'Which neural net to train ({"|".join(k for k in map)})'
parser.add_argument("neural_net", type=str, help=help_str)
args = parser.parse_args()

# Instantiate instance of given neural net class
classname = map[args.neural_net]["classname"]
NeuralNet = getattr(import_module(map[args.neural_net]["filename"]), classname)
print(f"Training {classname}")

# Load training data
q, u, v = load_cubes("training_data.nc", ["pv", "u", "v"])

# Transpose data so it's lon, lat, lev, time
q.transpose()
u.transpose()
v.transpose()

# Train neural net
NeuralNet.train(q.data, u.data, v.data)