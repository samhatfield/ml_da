import numpy as np

def output(ensemble, truth):
    ens_mean = np.mean(ensemble, axis=0)
    with open('output.txt', 'a') as output_file:
        output_file.write(f"{ens_mean[0]}\t{truth[0]}\n")