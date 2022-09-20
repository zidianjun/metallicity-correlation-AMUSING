
from paths import *
from utils import read_file, get_one_value

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')


def fill_factor():

    table = read_file(name='filling_factor.csv', path=output_path + '/thres/')
    mother_sample = read_file()

    b2a_list, fill_f_A, fill_f_p, l1_norm = [], [], [], []
    for name in mother_sample.name:
        b2a = get_one_value(name, 'b2a', obj_catalog=mother_sample)
        b2a_list.append(b2a)
        if b2a > 0.4:
            ind = (table.name == name)
            fill_f_A.append(float(table.fill_f_A[ind]))
            fill_f_p.append(float(table.fill_f_p[ind]))
            l1_norm.append(float(table.l1_norm[ind]))
            if (float(table.fill_f_A[ind]) > 0.04 and float(table.fill_f_p[ind]) < 0.4) and \
               float(table.l1_norm[ind]) >= 2.5:
               print name, float(table.l1_norm[ind])
    print("b/a ratio criteria remove %d galaxies." %(len(table) - len(fill_f_A)))

    shaded = (np.array(fill_f_A) <= 0.04) & (np.array(fill_f_p) <= 0.4)
    unshaded = (np.array(l1_norm) >= 2.5) & (
               (np.array(fill_f_A) > 0.04) | (np.array(fill_f_p) > 0.4))
    print("Filling factor criteria remove %d galaxies,\nand $L^1$ norm " %(np.sum(shaded)) +
          "criterion removes %d galaxies.\nThus, %d galaxies are left." %(np.sum(unshaded),
           len(fill_f_A) - np.sum(shaded) - np.sum(unshaded)))
    name_list = mother_sample.name[(np.array(b2a_list) > 0.4)][~shaded & ~unshaded]
    pd.DataFrame({'name': name_list}).to_csv(obj_path + 'high_fill_list.csv', index=False)


    plt.figure(figsize=(10, 8))
    ax = plt.axes([.12, .12, .83, .83])
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig = ax.scatter(fill_f_A, fill_f_p, c=l1_norm, edgecolor='k', s=75,
                     cmap=plt.cm.RdYlBu_r, vmax=2.5)
    cbar = plt.colorbar(fig)
    cbar.set_label(r'$L^1$ norm $\times$ 100', size=20)
    cbar.ax.tick_params(labelsize=20)
    xx = np.arange(8e-5, 4e-2, 1e-5)
    ax.fill_between(xx, xx*0 + 8e-4, xx*0 + 4e-1, color='gray', zorder=0, alpha=.2)
    ax.set_xlim(8e-5, 1)
    ax.set_ylim(8e-4, 1)
    ax.set_xlabel(r'H$\alpha$ area filling factor', fontsize=20)
    ax.set_ylabel(r'H$\alpha$ pixel number filling factor', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)

    if server_mode:
        plt.savefig(savefig_path + '/selection.pdf')

fill_factor()

