
from hub import DataHub
import config
import constant
from paths import *

import numpy as np
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')
from scipy.stats import pearsonr


def beam_test(dh=DataHub()):

    kpc_per_arcsec = dh.col('dist') * constant.arcsec

    s50, s16, s84 = dh.sigma_beam()
    X = (s50 / kpc_per_arcsec  / np.sqrt((dh.col('b2a') ** 2 - config.q0 ** 2) / (1 - config.q0 ** 2)))
    X_lower = (s16 / kpc_per_arcsec / np.sqrt((dh.col('b2a') ** 2 - config.q0 ** 2) / (1 - config.q0 ** 2)))
    X_upper = (s84 / kpc_per_arcsec / np.sqrt((dh.col('b2a') ** 2 - config.q0 ** 2) / (1 - config.q0 ** 2)))


    l50, l16, l84 = dh.corr_len()
    Y = l50 / kpc_per_arcsec
    Y_upper = l16 / kpc_per_arcsec
    Y_lower = l84 / kpc_per_arcsec

    plt.subplots(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xx = np.arange(.1, 10., .01)
    ax.plot(xx, xx, color='gray', linestyle='--')
    ax.set_xlim(2e-3, 10)
    ax.set_ylim(.2, 25)
    ax.set_xlabel('$\lambda_{\mathrm{beam}}$ (arcsec)', fontsize=20)
    ax.set_ylabel('$\lambda_{\mathrm{corr}}$ (arcsec)', fontsize=20)
    x_ticks = [1e-2, 1e-1, 1, 1e1]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['10$^{-2}$', '10$^{-1}$', '10$^0$', '10$^1$'])
    y_ticks = [.2, .4, .6, 1, 2, 3, 10]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.tick_params(axis='both', labelsize=20)

    ax.scatter(X, Y, color='w', edgecolor='k', s=80)
    ax.vlines(X, Y_lower, Y_upper, color='gray', linewidth=2)
    ax.hlines(Y, X_lower, X_upper, color='gray', linewidth=1)

    pr_value = pearsonr(np.log10(X[(X>0) & (Y>0)]), np.log10(Y[(X>0) & (Y>0)]))[0]
    ax.annotate("Pearson correlation is %.2f" %(pr_value),
                xy=(3e-3, 18), fontsize=15)

    if server_mode:
        plt.savefig(savefig_path + '/property/beam_size.pdf')



opt = dict(color='gray', alpha=.5, connectionstyle='arc3, rad=0',
           arrowstyle='simple, head_width=.25, head_length=.5, tail_width=.03')

def _corner(quantity, d1, d2, num):
    ax = plt.subplot(220+num)
    ax.set(aspect=1.0/ax.get_data_ratio())

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=15)
    
    if quantity == 'lcorr':
        ax.set_xlim(3e-2, 5e1)
        ax.set_ylim(3e-2, 5e1)
        if num % 2 == 1:
            ax.set_ylabel('$l_{\mathrm{corr}}$ [%s] (kpc)' %(d2), fontsize=20)
        if num > 2:
            ax.set_xlabel('$l_{\mathrm{corr}}$ [%s] (kpc)' %(d1), fontsize=20)
        x0, x1, x2, l10_d1 = DataHub(diag=d1).corr_len(long_return=True)
        y0, y1, y2, l10_d2 = DataHub(diag=d2).corr_len(long_return=True)
        uplims_d1, uplims_d2 = l10_d1 < x0 / 5., l10_d2 < y0 / 5.

    elif quantity == 'winj':
        ax.set_xlim(1.5, 6e3)
        ax.set_ylim(1.5, 6e3)
        if num % 2 == 1:
            ax.set_ylabel('$w_{\mathrm{inj}}$ [%s] (pc)' %(d2), fontsize=20)
        if num > 2:
            ax.set_xlabel('$w_{\mathrm{inj}}$ [%s] (pc)' %(d1), fontsize=20)
        x0, x1, x2, w10_d1 = DataHub(diag=d1).inj_width(long_return=True)
        y0, y1, y2, w10_d2 = DataHub(diag=d2).inj_width(long_return=True)
        uplims_d1, uplims_d2 = w10_d1 < x0 / 5., w10_d2 < y0 / 5.
    else:
        raise ValueError("Physical quantity must be 'lcorr' or 'winj'!")

    ind = (x1 > 0) & (y1 > 0)

    good = ind & ~uplims_d1 & ~uplims_d2
    ax.scatter(x0[good], y0[good], color='gray', edgecolor='k', s=40)
    ax.vlines(x0[good], y1[good], y2[good], color='gray', zorder=0)
    ax.hlines(y0[good], x1[good], x2[good], color='gray', zorder=0)

    pr_value = pearsonr(np.log10(x0[good]), np.log10(y0[good]))[0]
    if quantity == 'lcorr':
        ax.annotate("Pearson correlation is %.2f" %(pr_value),
                    xy=(5e-2, 3e1), fontsize=15)
    else:
        ax.annotate("Pearson correlation is %.2f" %(pr_value),
                    xy=(4, 3e3), fontsize=15)

    for x, y in zip(x2[ind & uplims_d1], y0[ind & uplims_d1]):
        ax.scatter(x, y, color='gray', s=20)
        ax.annotate("", xy=(.5*x, y), xytext=(x, y), arrowprops=opt)
    for x, y in zip(x0[ind & uplims_d2], y2[ind & uplims_d2]):
        ax.scatter(x, y, color='gray', s=20)
        ax.annotate("", xy=(x, .5*y), xytext=(x, y), arrowprops=opt)
    for x, y in zip(x2[ind & uplims_d1 & uplims_d2], y2[ind & uplims_d1 & uplims_d2]):
        ax.scatter(x, y, color='gray', s=20)
        ax.annotate("", xy=(.5*x, y), xytext=(x, y), arrowprops=opt)
        ax.annotate("", xy=(x, .5*y), xytext=(x, y), arrowprops=opt)

    xx = np.arange(3e-2, 6e3)
    ax.plot(xx, xx, color='gray', ls='--')


def diag_test(quantity='lcorr'):
    plt.subplots(figsize=(10, 10))
    _corner(quantity, 'PPN2', 'PPO3N2', 1)
    _corner(quantity, 'PPN2', 'D16', 3)
    _corner(quantity, 'PPO3N2', 'D16', 4)

    if server_mode:
        plt.savefig(savefig_path + '/property/corner_lcorr.pdf')


beam_test()
diag_test()

