
from hub import DataHub, read_table, open_full_chain, blue, red
from paths import *
from config import n_sample, n_walker

import numpy as np
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')


def winj_hist(dh=DataHub()):

    v50, v16, v84, v10 = dh.inj_width(long_return=True)
    uplim = (v10 < v50 / 5.) | (v16 < v50 / 3.)

    full_chain = open_full_chain()

    winj_arr, winj_good = np.zeros(1), np.zeros(1)
    for i, line in enumerate(full_chain.readlines()):
        first_space = line.index(' ')
        array = np.array(line[first_space+1:-1].split(' '), dtype=np.float64)
        array = array.reshape([n_sample * n_walker, 3])
        winj_arr = np.append(winj_arr, np.where(array[:, 1] > 0., array[:, 1], 1e-3) * 1e3)
        if not uplim[i]:
            winj_good = np.append(winj_good, np.where(array[:, 1] > 0., array[:, 1], 1e-3) * 1e3)
    winj_arr = winj_arr[1:]
    winj_good = winj_good[1:]

    bins = np.arange(0., 3., .2)#np.arange(0, 200, 5)
    diff = np.diff(bins)

    xx = np.arange(0., 1., .01)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))
    fig.text(0.5, 0.04, '$w_{\mathrm{inj}}$ / pc', ha='center', fontsize=20)
    fig.text(0.02, 0.5, '$dn/d$log($w_{\mathrm{inj}}$ / pc)',
             va='center', rotation='vertical', fontsize=20)

    ax = axes[0]
    counts, bin_edges = np.histogram(np.log10(winj_arr), bins, density=True)
    ax.bar(bin_edges[:-1] + .5 * diff[0], counts, width=diff, color=blue, alpha=.8)
    ax.plot(xx, xx, color='gray', ls='--')
    x_ticks = [1, 2, 4, 10, 20, 40, 100, 200, 1000]
    ax.set_xticks(np.log10(x_ticks))
    ax.set_xticklabels(x_ticks)
    ax.set_ylim(0., .9)
    ax.annotate('Posteriors', xy=(2.3, .7), xytext=(2.3, .7), fontsize=15)
    ax.tick_params(axis='both', labelbottom=False, labelsize=20)

    ax = axes[1]
    counts, bin_edges = np.histogram(np.log10(np.where(v50 > 0., v50, 1.)), bins,
                                     density=True)
    ax.bar(bin_edges[:-1] + .5 * diff[0], counts, width=diff, color=red, alpha=.8)
    ax.plot(xx, xx, color='gray', ls='--')
    x_ticks = [1, 2, 4, 10, 20, 40, 100, 200, 1000]
    ax.set_xticks(np.log10(x_ticks))
    ax.set_xticklabels(x_ticks)
    ax.set_ylim(0., .9)
    ax.annotate('Medians', xy=(2.3, .7), xytext=(2.3, .7), fontsize=15)
    ax.tick_params(axis='both', labelsize=20)

    if server_mode:
        plt.savefig(savefig_path + '/property/winj_hist.pdf')

winj_hist()







