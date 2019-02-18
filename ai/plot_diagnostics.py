from argparse import ArgumentParser
from importlib import import_module
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', rc={'font.sans-serif': "Helvetica"})
sns.set_palette(sns.color_palette('Set1'))

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
parser.add_argument("--stencil", default=-1, help="What stencil size to use for training InteriorNN")
args = parser.parse_args()

if args.stencil == -1 and args.neural_net == "interior":
    raise ValueError("If you want to plot InteriorNN diagnostics you must provide a stencil size")

if args.neural_net == "BoundariesNN":
    diagnostics = np.loadtxt(f"boundariesnn_history.txt")
else:
    diagnostics = np.loadtxt(f"interior_{args.stencil}_history.txt")

plt.plot(diagnostics[:,0], label="val\_loss")
plt.plot(diagnostics[:,1], label="loss")
plt.legend()
plt.show()
