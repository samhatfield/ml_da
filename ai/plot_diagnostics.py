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

if args.neural_net == "boundaries":
    diagnostics = np.loadtxt(f"models/boundariesnn_history.txt")
else:
    diagnostics = np.loadtxt(f"models/interior_{args.stencil}_history.txt")

ax1 = plt.subplot(211)
ax1.plot(diagnostics[:,0], label="validation data")
ax1.plot(diagnostics[:,1], label="training data")
ax1.set_ylabel("Mean average error")
ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(diagnostics[:,2], label="validation data")
ax2.plot(diagnostics[:,3], label="training data")
ax2.set_ylabel("Correlation")
ax2.set_xlabel("Epoch")
plt.legend()
if args.neural_net == "boundaries":
    plt.savefig("plots/boundaries_diag.pdf")
else:
    plt.savefig(f"plots/interior_{args.stencil}_diag.pdf")
plt.show()
