from argparse import ArgumentParser
from importlib import import_module
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', rc={'font.sans-serif': "Helvetica"})
sns.set_palette(sns.color_palette('Set1'))

# Map command line arguments to file names
map = {
    "boundaries": "boundariesnn",
    "three_by_three": "threebythreenn"
}

# Parse command line arguments
parser = ArgumentParser(description="Plots the training diagnostics")
help_str = f'Which neural net to plot ({"|".join(k for k in map)})'
parser.add_argument("neural_net", type=str, help=help_str)
args = parser.parse_args()

diagnostics = np.loadtxt(f"{map[args.neural_net]}_history.txt")

plt.plot(diagnostics[:,0], label="val\_loss")
plt.plot(diagnostics[:,1], label="loss")
plt.legend()
plt.show()