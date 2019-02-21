from argparse import ArgumentParser
from interiornn import InteriorNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set_style('whitegrid', {'font.sans-serif': "Helvetica"})
sns.set_palette(sns.color_palette('Set1'))

# Parse command line arguments
parser = ArgumentParser(description="Trains the neural nets")
help_str = f'Which neural net to train (boundaries|interior)'
parser.add_argument("neural_net", type=str, help=help_str)
parser.add_argument("--stencil", default=-1, type=int,
                    help="What stencil size to use for training InteriorNN")
parser.add_argument("--num_points", default=10000, type=int,
                    help="Number of points to plot (default: 10000)")
args = parser.parse_args()

if args.stencil == -1 and args.neural_net == "interior":
    raise ValueError("If you want to plot InteriorNN diagnostics you must provide a stencil size")

# Load training history file
if args.neural_net == "boundaries":
    raise NotImplementedError
else:
    # Load validation data
    validation_data = np.load(f"training_data/interior_{args.stencil}_validation_data.npz")
    val_in, val_out = validation_data["val_in"], validation_data["val_out"]

    # Build neural net
    net = InteriorNN(args.stencil)

    # Subset data
    val_in = val_in[:args.num_points,:]
    val_out = val_out[:args.num_points,:]

    # Infer tendencies from validation data
    infer_out = net.interior_model.predict(val_in, batch_size=1)

# Plot scatter plots
fig, axes = plt.subplots(1,2, figsize=(10,5))
plt.suptitle(f"{args.stencil}x{args.stencil} stencil")
axes[0].scatter(val_out[:,0], infer_out[:,0], s=2.0, alpha=0.3)
axes[0].set_title(f"$\psi_0$ r={pearsonr(val_out[:,0], infer_out[:,0])[0]:.2f}")
axes[1].scatter(val_out[:,1], infer_out[:,1], s=2.0, alpha=0.3)
axes[1].set_title(f"$\psi_1$ r={pearsonr(val_out[:,1], infer_out[:,1])[0]:.2f}")


for i in range(2):
    axes[i].set_xlabel("Actual")
    axes[i].set_ylabel("Inferred")

# Make axis limits equal
x_min_0, x_max_0 = axes[0].get_xlim()
y_min_0, y_max_0 = axes[0].get_ylim()
x_min_1, x_max_1 = axes[1].get_xlim()
y_min_1, y_max_1 = axes[1].get_ylim()

ax_min = min(x_min_0, y_min_0, y_min_1)

axes[0].set_xlim([min(x_min_0, y_min_0), max(x_max_0, y_max_0)])
axes[0].set_ylim([min(x_min_0, y_min_0), max(x_max_0, y_max_0)])
axes[1].set_xlim([min(x_min_1, y_min_1), max(x_max_1, y_max_1)])
axes[1].set_ylim([min(x_min_1, y_min_1), max(x_max_1, y_max_1)])

# Save to file
if args.neural_net == "boundaries":
    raise NotImplementedError
else:
    plt.savefig(f"plots/interior_{args.stencil}_corr.pdf", bbox_inches="tight")

plt.show()
