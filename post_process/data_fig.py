
from hub import DataHub, read_table, blue
from paths import *

import numpy as np
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')


def basic_info(dh=DataHub()):

    mother_sample = read_table(name='AMUSING_galaxies.csv')

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=.15, bottom=.12, right=.96, top=.96, wspace=.2)

    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.scatter(10 ** mother_sample.log_Mass, 10 ** mother_sample.lSFR, s=50,
               color='w', edgecolor='k', label='AMUSING++ mother sample')
    ax.scatter(10 ** dh.col('Mass'), 10 ** dh.col('lSFR'), s=50,
               color=blue, edgecolor='k', label='Selected galaxies in this work')

    ax.set_xlim(2e7, 1.5e12)
    ax.set_ylim(1e-7, 6e2)
    ax.set_xlabel('Stellar mass (M$_{\odot}$)', fontsize=20)
    ax.set_ylabel('SFR (M$_{\odot}$ yr$^{-1}$)', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(loc='upper left', fontsize=15)

    if server_mode:
        plt.savefig(savefig_path + 'sample.pdf')

basic_info()



