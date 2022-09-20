
from constant import arcsec_per_pix as app
from paths import *
from galmet import Galaxy
from utils import read_eline, display_map, plot_cross, find_center
from utils import read_file, get_one_value

import multiprocessing
import numpy as np
from os.path import isfile
import pandas as pd
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')


def _plot_box(ax, coords, color, lw=3):

    x1, x2, y1, y2 = coords
    ax.vlines(x1, y1, y2, color=color, lw=lw)
    ax.vlines(x2, y1, y2, color=color, lw=lw)
    ax.hlines(y1, x1, x2, color=color, lw=lw)
    ax.hlines(y2, x1, x2, color=color, lw=lw)

def _draw_line(x1, y1, x2, y2, ax, color, lw=3):
    xx = np.arange(x1, x2, 1e-2)
    ax.plot(xx, ((y2 - y1) * xx + (x2*y1 - x1*y2)) / (x2 - x1), color=color, lw=lw)

def _plot_boxes(ax, coords, color, lw=3):

    x1, y1, x2, y2, x3, y3, x4, y4 = coords
    _draw_line(x1, y1, x2, y2, ax, color, lw=lw)
    _draw_line(x2, y2, x3, y3, ax, color, lw=lw)
    _draw_line(x4, y4, x3, y3, ax, color, lw=lw)
    _draw_line(x1, y1, x4, y4, ax, color, lw=lw)

def ams_name(alias, size):
    if alias == 'IC5179':
        l = ['IC+5179', 'target_5_centre']
    elif alias == 'NGC2466':
        l = ['SN2016iye', 'ASASSN14dd_1']
    else:  # alias == 'NGC3318':
        l = ['SN2000cl', 'SN2017ahn']
    return l[size == 'large']


def plot_image():

    eline, fontsize = 'Halpha', 20
    alias_list = ['IC5179', 'NGC2466', 'NGC3318']


    fig = plt.figure(figsize=(18, 12))
    plt.subplots_adjust(left=.08, bottom=.08, right=.96, top=.8, hspace=0.)
    
    
    for i, alias in enumerate(alias_list):
        for j, (size, color) in enumerate(zip(['small', 'large'], ['#df1d27', '#367db7'])):
            
            ax = plt.subplot(231 + i + j * 3)
            if j == 1:
                _plot_box(ax, extent, '#df1d27', lw=.1)

            name = ams_name(alias, size=size)
            signal, noise = read_eline(name, 'Halpha')
            cx, cy = find_center(name)
            im, extent = display_map(signal, noise, ax, cmap=plt.cm.binary, log=True,
                                    vrange=(-2, 1), center=(cx, cy), long_return=True)
            beam = get_one_value(name, 'FWHM', obj_catalog=read_file(name='/PSF/PSF.csv'))

            if j == 1 and i == 0:
                coords = np.array([0, 160, 165, 416, 421, 251, 256, -5]) * app
                coords[0::2] -= cx * app
                coords[1::2] -= cy * app
                _plot_boxes(ax, coords, color)
            else:
                _plot_box(ax, extent, color)

            if j == 0:
                plot_cross(ax, name, (-65, -55))
                ax.annotate('5 kpc', xy=(-75, -75), fontsize=fontsize)
                ax.annotate('%.2f"' %(beam), xy=(45, -75), fontsize=fontsize)
                ax.tick_params(axis='both', labelbottom=False, labelsize=fontsize)
                if i == 2:
                    ax.vlines(1e3, 1e4, 2e4, color='#df1d27', lw=3, label='Observation 1')
                    ax.legend(loc='upper right', fontsize=fontsize)
            else:
                ax.annotate(alias, xy=(-75, 65), fontsize=fontsize)
                ax.annotate('%.2f"' %(beam), xy=(45, 65), fontsize=fontsize)
                ax.set_xlabel('x (arcsec)', fontsize=fontsize)
                if i == 2:
                    ax.vlines(1e3, 1e4, 2e4, color='#367db7', lw=3, label='Observation 2')
                    ax.legend(loc='lower right', fontsize=fontsize)

            if i == 0:
                ax.set_ylabel('y (arcsec)', fontsize=fontsize)

            ax.tick_params(axis='both', labelsize=fontsize)
            ax.set_xlim(-80, 80)
            ax.set_ylim(-80, 80)


    cbar_ax = fig.add_axes([0.088, 0.92, 0.863, 0.03])
    cbar = plt.colorbar(im, ticks=[-2, -1, 0, 1.], cax=cbar_ax, orientation="horizontal")
    cbar.ax.set_xticklabels(['10$^{-2}$', '10$^{-1}$', '10$^{0}$', '10$^{1}$'])
    cbar.set_label('Line flux ($10^{-16}$ erg)', size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    if server_mode:
        plt.savefig(savefig_path + 'overlapping_image.pdf')

def write_lcorr(par):
    name, size, truncate, flag = par
    f = open(output_path + '/overlap.csv', 'a+')
    f.write('%s,%s,%d,%.3f\n' %(name, size, flag, np.sqrt(
             Galaxy(ams_name(name, size), truncate=truncate, fit_once=True).par_kt)))
    f.close()

def plot_overlap(opt=dict(color='gray', alpha=.5, connectionstyle='arc3, rad=0',
                 arrowstyle='simple, head_width=5, head_length=20, tail_width=.06')):
    
    alias_list = ['IC5179', 'NGC2466', 'NGC3318']

    if isfile(output_path + '/overlap.csv'):
        print("Overlap.csv is ready!\n")
    else:
        truncate_dic = {'IC+5179': (0, 10000, 0, 10000),
                        'SN2016iye': (0, 10000, 0, 10000),
                        'SN2000cl': (115, 10000, 0, 10000),
                        'target_5_centre': (50, 368, 41, 359),
                        'ASASSN14dd_1': (64, 397, 139, 460),
                        'SN2017ahn': (0, 206, 11, 329)}

        par_list = []
        for alias in alias_list:
            for size in ['small', 'large']:
                par_list.append((alias, size, (0, 10000, 0, 10000), False))
                par_list.append((alias, size, truncate_dic[ams_name(alias, size)], True))

        f = open(output_path + '/overlap.csv', 'a+')
        f.write('name,size,if_truncate,lcorr\n')
        f.close()
        pool = multiprocessing.Pool(processes=len(par_list))
        pool.map(write_lcorr, par_list)


    plt.figure(figsize=(10, 10))
    ax = plt.axes([.15, .15, .8, .8])

    data = pd.read_csv(output_path + '/overlap.csv')
    small_size = np.array(data['size'] == 'small', dtype=bool)
    large_size = np.array(data['size'] == 'large', dtype=bool)
    is_truncate = np.array(data['if_truncate'], dtype=bool)
    lines0, lines1, labels0, labels1 = [], [], [], []
    for alias, marker in zip(alias_list, ['^', 's', 'p']):
        ind = data.name == alias
        ax.scatter(data[ind & small_size & ~is_truncate].sort_values(by='name').lcorr,
                   data[ind & large_size & ~is_truncate].sort_values(by='name').lcorr,
                   color='w', edgecolor='k', s=300, marker=marker, zorder=3)
        ax.scatter(data[ind & small_size & is_truncate].sort_values(by='name').lcorr,
                   data[ind & large_size & is_truncate].sort_values(by='name').lcorr,
                   color='gray', edgecolor='k', s=300, marker=marker, zorder=4)
        lines0.append(ax.scatter(np.nan, np.nan, color='w', edgecolor='k',
                      s=200, marker=marker))
        labels0.append(alias)
    for label, color in zip(['original FoV', 'overlapping FoV'], ['w', 'gray']):
        lines1.append(ax.scatter(np.nan, np.nan,
                                 color=color, edgecolor='k', s=300, marker='o'))
        labels1.append(label)


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(.4, 5.)
    ax.set_ylim(.4, 5.)
    xx = np.arange(.3, 5., .01)
    ax.plot(xx, xx, color='gray', ls='--')
    ax.fill_between(xx, xx / 1.25, xx * 1.25, color='gray', alpha=.2, zorder=2)
    ax.fill_between(xx, xx / 1.50, xx * 1.50, color='gray', alpha=.1, zorder=1)
    ax.set_xlabel('$l_{\mathrm{corr}}$ [Observation 1] (kpc)', fontsize=20)
    ax.set_ylabel('$l_{\mathrm{corr}}$ [Observation 2] (kpc)', fontsize=20)
    ticks = [.4, .6, .8, 1, 2, 4]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.tick_params(axis='both', labelsize=20)

    legend0 = ax.legend(lines0, labels0, loc='upper left', fontsize=20)
    plt.gca().add_artist(legend0)
    legend1 = ax.legend(lines1, labels1, loc='lower right', fontsize=20)
    plt.gca().add_artist(legend1)

    if server_mode:
        plt.savefig(savefig_path + 'overlapping_lcorr.pdf')

plot_image()
plot_overlap()






