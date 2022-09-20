
from hub import DataHub, red, blue
from paths import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import binned_statistic
if server_mode:
    plt.switch_backend('agg')


def group(x, y, mask=None):
    x = np.where(x > 14,  16, x)
    x = np.where((x < 10) & (x > 0), 8.5, x)
    x = np.where((x > 9) & (x < 12), 10.5, x)

    if mask is not None:
        df = pd.DataFrame({'y': y[mask], 'x': x[mask]})
    else:
        df = pd.DataFrame({'y': y, 'x': x})
    # df.groupby(by='x')
    return df
    # count = group.count()
    # v50, v16, v84 = group.quantile(q=.50), group.quantile(q=.16), group.quantile(q=.84)
    # return count, v50, v16, v84


def plot_morph(dh=DataHub()):
    x, y = np.nan_to_num(dh.col('Ttype')), dh.corr_len()[0]

    plt.figure(figsize=(10, 8))
    ax = plt.axes([.1, .1, .85, .85])

    mass_group = group(x, y, mask=(dh.col('Mass')<10.5))
    mass_group.boxplot(
        by='x', column='y', ax=ax, grid=False, showfliers=False, patch_artist=True,
        widths=.25,
            boxprops=dict(facecolor=blue, color=blue, alpha=.5),
            capprops=dict(color=blue),
            whiskerprops=dict(color=blue),
            medianprops=dict(color=blue))
    ax.scatter(-100, -100, marker='s', s=5e2, color=blue, alpha=.5,
        label='Low stellar Mass subset\nlog($M_*/\mathrm{M_{\odot}}$) < 10.5')
    print mass_group

    for c, pos_x, pos_y in zip(count.y, y50.index, y50.y):
        ax.annotate(c, xy=(pos_x+.2, pos_y), fontsize=15, color='gray')

    mass_group = group(x, y, mask=(dh.col('Mass')>10.5))
    mass_group.boxplot(
        by='x', column='y', ax=ax, grid=False, showfliers=False, patch_artist=True,
        widths=.50,
            boxprops=dict(facecolor=red, color=red, alpha=.5),
            capprops=dict(color=red),
            whiskerprops=dict(color=red),
            medianprops=dict(color=red))
    ax.scatter(-100, -100, marker='s', s=2e3, color=red, alpha=.5,
        label='High stellar Mass subset\nlog($M_*/\mathrm{M_{\odot}}$) > 10.5')
    print mass_group
    # ax.scatter(y50.index - .025, y50.y, color=blue, edgecolor='k', s=count.y*10,
    #            label='Low stellar Mass subset\nlog($M_*/\mathrm{M_{\odot}}$) < 10.5')
    # ax.vlines(y50.index - .025, y16.y, y84.y, color='gray')
    # for c, pos_x, pos_y in zip(count.y, y50.index, y50.y):
    #     ax.annotate(c, xy=(pos_x+.2, pos_y), fontsize=15, color='gray')

    ticks = range(2, 8)
    labels = ['S0 - S0a', 'Sa - Sab', 'Sb', 'Sbc', 'Sc', 'Scd - I']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', labelsize=20)
    plt.suptitle('')
    ax.set_title('')
    ax.set_xlim(1.5, 7.5)
    ax.set_ylim(-.1, 3.6)
    ax.set_xlabel('Hubble type', fontsize=20)
    ax.set_ylabel('Correlation length (kpc)', fontsize=20)
    ax.legend(loc='upper left', fontsize=20)

    if server_mode:
        plt.savefig(savefig_path + '/property/morph.pdf')


plot_morph()

plt.show()


