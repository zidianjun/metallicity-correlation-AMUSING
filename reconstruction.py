
import config
from constant import arcsec, arcsec_per_pix
from paths import *
from utils import display_map, multi_order_bin, reconstruct_maps
from utils import get_beam, get_morph, tpcf
from mcmc import fit_once

from astropy.io import fits
from cv2 import GaussianBlur
from os.path import isfile
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')


def recon_plot(name, eline='SII6731', vrange=(-3, 0), fontsize=20):

    ticks = range(vrange[0], vrange[1] + 1)
    ticklabels = []
    for tick in ticks:
        ticklabels.append('10$^{%d}$' %(tick))

    eline_data = fits.open(fits_path + 'flux_elines.' +  name + '.cube.fits')[0].data
    signal = eline_data[config.eline_dict[eline]]
    signal = np.where(signal > 0., signal, 0.)
    noise = eline_data[config.eline_dict[eline] + config.diff]
    noise = np.where(noise > 0., noise, np.mean(noise))
    
    rec_s, rec_n, maps = multi_order_bin(signal, noise, config.min_SN)

    fig = plt.figure(figsize=(24, 9))
    plt.subplots_adjust(left=.04, bottom=.08, right=.98, top=.8, wspace=.15)

    ax = plt.subplot(131)
    signal = eline_data[config.eline_dict[eline]]
    signal = np.where(signal > 0., signal, 0.)
    im = display_map(signal, noise, ax, min_SN=config.min_SN, vrange=vrange)
    cbar_ax = fig.add_axes([0.04, 0.92, 0.28, 0.05])
    cbar = plt.colorbar(im, orientation="horizontal", cax=cbar_ax, ticks=ticks)
    cbar.ax.set_xticklabels(ticklabels)
    cbar.set_label('Line flux ($10^{-16}$ erg)', size=20)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel('x (arcsec)', fontsize=fontsize)
    ax.set_ylabel('y (arcsec)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    ax = plt.subplot(132)
    im = display_map(maps, 1., ax, min_SN=-1., log=False, cmap=plt.cm.RdYlBu)
    cbar_ax = fig.add_axes([0.368, 0.92, 0.28, 0.05])
    maximum = int(np.max(maps))
    cbar = plt.colorbar(im, orientation="horizontal", ticks=np.arange(maximum)+.5, cax=cbar_ax)
    cbar.ax.set_xticklabels(2**np.arange(maximum))
    cbar.set_label('Map number', size=20)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel('x (arcsec)', fontsize=fontsize)
    ax.set_ylabel('y (arcsec)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    ax = plt.subplot(133)
    im = display_map(rec_s, rec_n, ax, min_SN=config.min_SN, vrange=vrange)
    cbar_ax = fig.add_axes([0.695, 0.92, 0.28, 0.05])
    cbar = plt.colorbar(im, orientation="horizontal", cax=cbar_ax, ticks=ticks)
    cbar.ax.set_xticklabels(ticklabels)
    cbar.set_label('Line flux ($10^{-16}$ erg)', size=20)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel('x (arcsec)', fontsize=fontsize)
    ax.set_ylabel('y (arcsec)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    if server_mode:
      plt.savefig(savefig_path + 'recon_' + name + '.pdf')




def two_point_corr_plot(name, eline='SII6731', fontsize=20):

    eline_data = fits.open(fits_path + 'flux_elines.' +  name + '.cube.fits')[0].data
    signal = eline_data[config.eline_dict[eline]]
    signal = np.where(signal > 0., signal, 0.)
    noise = eline_data[config.eline_dict[eline] + config.diff]
    noise = np.where(noise > 0., noise, np.mean(noise))
    height, width = signal.shape
    min_SN = config.min_SN
    rec_s, rec_n, maps = multi_order_bin(signal, noise, min_SN)

    beam = get_beam(name)[0]
    distance = get_morph(name)[-1]
    kpc_per_pix = distance * arcsec * arcsec_per_pix

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.reshape(-1) * kpc_per_pix, y.reshape(-1) * kpc_per_pix

    if isfile(output_path + 'mask/mask_' + name + '_binning_False.npy'):
        mask2 = np.load(output_path + 'mask/mask_' + name + '_binning_False.npy')
    else:
        raise ValueError("Mask file does not exist. Adding mask file would be necessary.")
    if isfile(output_path + 'mask/mask_' + name + '_binning_True.npy'):
        mask1 = np.load(output_path + 'mask/mask_' + name + '_binning_True.npy')
    else:
        raise ValueError("Mask file does not exist. Adding mask file would be necessary.")

    plt.subplots(figsize=(8, 4))
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.85)

    s1, s2 = [], []
    for i in range(50):
        blue_noise = GaussianBlur(np.random.normal(0, 1, (height, width)), (15, 15),
                                  beam / kpc_per_pix)
        recon_blue_noise, _ = reconstruct_maps(blue_noise, blue_noise, maps)
        
        bin_d2, bin_s2 = tpcf(blue_noise.reshape(-1)[mask2], x[mask2], y[mask2],
                              bin_size=kpc_per_pix)
        s2.append(bin_s2)
        bin_d1, bin_s1 = tpcf(recon_blue_noise.reshape(-1)[mask1], x[mask1], y[mask1],
                              bin_size=kpc_per_pix)
        s1.append(bin_s1)
        print("Bootstrap for blue noise #%d finished." %(i))

    bn2_50, bn2_16, bn2_84 = (np.percentile(s2, 50, axis=0),
                              np.percentile(s2, 16, axis=0),
                              np.percentile(s2, 84, axis=0))
    bn2_mean, bn2_std = np.mean(s2, axis=0), np.std(s2, axis=0)

    bn1_50, bn1_16, bn1_84 = (np.percentile(s1, 50, axis=0),
                              np.percentile(s1, 16, axis=0),
                              np.percentile(s1, 84, axis=0))
    bn1_mean, bn1_std = np.mean(s1, axis=0), np.std(s1, axis=0)

    plt.fill_between(bin_d1, bn1_16, bn1_84, color='r', alpha=.3, edgecolors='none',
                     zorder=2, label='binned blue noise')
    plt.fill_between(bin_d2, bn2_16, bn2_84, color='b', alpha=.3, edgecolors='none',
                     zorder=1, label='original blue noise')

    fit_once(bin_d1, bn1_mean, bn1_std, name)
    fit_once(bin_d2, bn2_mean, bn2_std, name)

    plt.xlim(-.05, 1.1)
    plt.ylim(-.1, 1.1)
    plt.xlabel("$r$ (kpc)", fontsize=fontsize)
    plt.ylabel("$\\xi$($r$)", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc='upper right')

    if server_mode:
        plt.savefig(savefig_path + 'blue_noise_2p_corr_' + name + '.pdf')



name = 'SN2011hb'
recon_plot(name, vrange=(-3, -1))
two_point_corr_plot(name)








