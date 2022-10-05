
from hub import DataHub, blue, red
from paths import *

import numpy as np
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')


def map_num_hist(dh=DataHub()):

    map_num_pc = dh.col('map_num_pc')

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=.12, bottom=.12, right=.92, top=.92)
    ax = plt.subplot(111)

    bin_size = .25
    bins = np.arange(1, 4.5, bin_size)
    counts, bin_edges = np.histogram(np.log10(map_num_pc), bins)
    ax.bar(bin_edges[:-1] + .5 * bin_size, counts, width=bin_size, color=blue, alpha=.8)
    ax.set_xticks([2, 3, 4])
    ax.set_xticklabels(['10$^2$', '10$^3$', '10$^4$'])
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('Median spatial resolution of the binned maps (pc)', fontsize=20)
    ax.set_ylabel('N', fontsize=20)

    if server_mode:
        plt.savefig(savefig_path + '/map_num_pc.pdf')

map_num_hist()







