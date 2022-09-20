
import sys 
sys.path.append("..")
from paths import *

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')

def read_table(name='AMUSING_galaxy.csv', path=obj_path):
    return pd.read_csv(path + name)

def open_full_chain(diag='D16', path=output_path, suffix=''):
    return open(path + '/total_chain_' + diag + suffix + '.txt', 'r')


red = '#df1d27'
blue = '#367db7'
green = '#4aad4a'


class DataHub(object):
    def __init__(self, diag='D16', proj_path=proj_path):

        self.diag, self.proj_path = diag, proj_path
        self.obj_path = self.proj_path + '/data/'
        self.output_path = self.proj_path + '/output/'
        self.name_list = read_table(name='high_fill_list.csv', path=self.obj_path)
        self.PSF = read_table(name='PSF.csv', path=self.obj_path + '/PSF/')
        self.Ttype = read_table(name='Ttype.csv', path=self.obj_path)
        self.basic = read_table(name='AMUSING_galaxy.csv', path=self.obj_path)
        self.SII_ratio = read_table(name='SII_ratio.csv', path=self.obj_path)
        self.fill_fac = read_table(name='filling_factor.csv',
                                   path=self.output_path+'/thres/')
        self.max_sep = read_table(name='max_sep.csv', path=self.output_path)

    def col(self, col_name):
        arr = []
        for name in self.name_list.name:
            if 'FWHM' in col_name:
                database = self.PSF
            elif col_name == 'Ttype':
                database = self.Ttype
            elif col_name == 'SII_ratio':
                database = self.SII_ratio
            elif col_name in ['total_number', 'fill_f_A', 'fill_f_p']:
                database = self.fill_fac
            elif col_name == 'max_sep':
                database = self.max_sep
            else:
                database = self.basic
            dtype_func = str if col_name == 'name' else float
            arr.append(dtype_func(database[database.name == name].iloc[0,
                database.columns.get_loc(col_name)]))
        return np.array(arr)

    def sigma_beam(self):
        if 'CALIFA' in self.proj_path:
            raise ValueError("CALIFA has no presumed beam size!")
        suffix = '_' + self.diag if self.diag in ['PPN2', 'PPO3N2'] else ''
        beam = read_table(name='sigma_beam_' +
                          self.diag + '.csv', path=self.output_path)
        return (np.array(beam['s_50']),
                np.array(beam['s_16']),
                np.array(beam['s_84']))

    def corr_len(self, long_return=False, suffix=''):
        lcorr = read_table(name='correlation_length_' +
                           self.diag + suffix +'.csv', path=self.output_path)
        if long_return:
            return (np.array(lcorr['l_50']),
                    np.array(lcorr['l_16']),
                    np.array(lcorr['l_84']),
                    np.array(lcorr['l_10']))
        else:
            return (np.array(lcorr['l_50']),
                    np.array(lcorr['l_16']),
                    np.array(lcorr['l_84']))

    def inj_width(self, long_return=False, suffix=''):
        if 'CALIFA' in self.proj_path:
            raise ValueError("CALIFA has no injection width!")
        suffix = '_' + self.diag if self.diag in ['PPN2', 'PPO3N2'] else ''
        winj = read_table(name='injection_width_' + 
                          self.diag + suffix + '.csv', path=self.output_path)
        if long_return:
            return (np.array(winj['w_50']),
                    np.array(winj['w_16']),
                    np.array(winj['w_84']),
                    np.array(winj['w_10']))
        else:
            return (np.array(winj['w_50']),
                    np.array(winj['w_16']),
                    np.array(winj['w_84']))

    def rand_draw(self, dtype, diag='D16', suffix=''):
        array = []
        f = open_full_chain(diag=diag, suffix=suffix)
        i = 1 if dtype == 'winj' else 2

        for line in f.readlines():
            s = line[(1+line.index(' ')):]
            chain = np.array([float(n) for n in s.split()])
            data_cube = np.reshape(chain, [500, 100, 3])
            
            rs = np.random.randint(500)
            rw = np.random.randint(100)
            L = np.mean(data_cube[rs, rw, i])
            
            if dtype == 'winj':
                array.append(L)
            else:
                array.append(np.sqrt(L))

        return np.array(array)

    def lcorr_vs_lcorr(self, suffix):
        l1_50, l1_16, l1_84 = self.corr_len()
        l2_50, l2_16, l2_84 = self.corr_len(suffix=suffix)

        mask = (l1_50 > 0) & (l2_50 > 0)
        pr_value = pearsonr(np.log10(l1_50[mask]), np.log10(l2_50[mask]))[0]

        plt.figure(figsize=(8, 8))
        ax = plt.axes([.15, .15, .8, .8])
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.scatter(l1_50, l2_50, color='k')
        ax.vlines(l1_50, l2_16, l2_84, color='gray')
        ax.hlines(l2_50, l1_16, l1_84, color='gray')

        if suffix == '_q0':
            ax.set_ylabel('$l_{\mathrm{corr}}$ [$q_0$ = 0.00] (kpc)', fontsize=20)
            ax.set_xlabel('$l_{\mathrm{corr}}$ [$q_0$ = 0.13] (kpc)', fontsize=20)
        elif suffix == '_q0_35':
            ax.set_ylabel('$l_{\mathrm{corr}}$ [$q_0$ = 0.35] (kpc)', fontsize=20)
            ax.set_xlabel('$l_{\mathrm{corr}}$ [$q_0$ = 0.13] (kpc)', fontsize=20)
        else:
            ax.set_ylabel('$l_{\mathrm{corr}}$ [S/N = 2] (kpc)', fontsize=20)
            ax.set_xlabel('$l_{\mathrm{corr}}$ [S/N = 3] (kpc)', fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlim(2e-2, 2e1)
        ax.set_ylim(2e-2, 2e1)
        xx = np.arange(2e-2, 2e1, .01)
        ax.plot(xx, xx, color='gray', ls='--')
        ax.annotate("Pearson correlation is %.2f" %(pr_value),
                    xy=(3e-2, 1e1), fontsize=15)

        plt.savefig(savefig_path + 'lcorr' + suffix + '.pdf')



# DataHub().lcorr_vs_lcorr('_SN_2')
# DataHub().lcorr_vs_lcorr('_q0_35')


