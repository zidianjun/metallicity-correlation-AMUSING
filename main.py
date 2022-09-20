
from galmet import Galaxy, GalaxyFigure
from paths import *
from utils import read_file
import config

import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')

def analyze(name, suffix=''):

    for diag in config.diag_list:
        samples = Galaxy(name, diag=diag).samples.reshape(-1)

        f = open(output_path + '/output' + suffix + '/total_chain_' + name + '.txt', 'a+')
        for i in range(len(samples)):
            if i == 0:
                f.write("%.3f" %(samples[i]))
            else:
                f.write(" %.3f" %(samples[i]))
        f.write("\n")
        f.close()


def filling_factor_criteria(name):

    g = Galaxy(name, fit_once=True)
    total_number, fill_f_A, fill_f_p = g.total_number, g.fill_f_A, g.fill_f_p
    l1_norm = g.l1_norm
    f = open(thres_path + '/filling_factor.csv', 'a+')
    f.write('%s,%d,%.4f,%.4f,%.1f\n' %(name, total_number,
            fill_f_A, fill_f_p, l1_norm))
    f.close()


def l1_norm_illus(name1='ASASSN-18oa', name2='NGC0613'):
    plt.figure(figsize=(30, 11))
    plt.subplots_adjust(left=.06, bottom=.12, right=.98, top=.96, wspace=.5, hspace=.3)

    gf = GalaxyFigure(Galaxy(name1, fit_once=True), savefig_path=savefig_path)
    gf.map(ax=plt.subplot2grid((2, 4), (0, 0)), dtype='met')
    gf.map(ax=plt.subplot2grid((2, 4), (0, 1)), dtype='met_fluc')
    gf.met_fluc_corr(ax=plt.subplot2grid((2, 4), (0, 2), colspan=2), stage=1)

    gf = GalaxyFigure(Galaxy(name2, fit_once=True), savefig_path=savefig_path)
    gf.map(ax=plt.subplot2grid((2, 4), (1, 0)), dtype='met')
    gf.map(ax=plt.subplot2grid((2, 4), (1, 1)), dtype='met_fluc')
    gf.met_fluc_corr(ax=plt.subplot2grid((2, 4), (1, 2), colspan=2), stage=1)

    if server_mode:
        plt.savefig(savefig_path + 'l1_norm_illus.pdf')
    else:
        plt.show()

if __name__ == '__main__':
    
    l1_norm_illus()

    f = open(thres_path + '/filling_factor.csv', 'a+')
    f.write('name,total_number,fill_f_A,fill_f_p,l1_norm\n')
    f.close()
    name_list = read_file().name
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    pool.map(filling_factor_criteria, name_list)
    
    name_list = read_file(name='high_fill_list.csv').name
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    pool.map(analyze, name_list)






    

    