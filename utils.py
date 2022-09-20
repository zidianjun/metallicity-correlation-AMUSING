
import constant
import config
from cdll import bin_stat
from paths import *

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')


# ========== read table ========== 

def read_file(name='AMUSING_galaxy.csv', path=obj_path):
    return pd.read_csv(path + name)

def get_one_value(name, col_name, obj_catalog=read_file()):
    return float(obj_catalog[obj_catalog.name == name].iloc[0,
           obj_catalog.columns.get_loc(col_name)])

# ========== deprojection ========== 

def get_morph(name):
    distance = get_one_value(name, 'dist')
    special = read_file(name='deproject.csv')
    if name in list(special.name):
        PA = get_one_value(name, 'PA', obj_catalog=special)
        b2a = get_one_value(name, 'b2a', obj_catalog=special)
    else:
        PA = get_one_value(name, 'PA')
        b2a = get_one_value(name, 'b2a')
    return (PA, b2a, distance)

def find_center(name, obj_catalog=read_file()):
    return get_one_value(name, 'cx'), get_one_value(name, 'cy')

def b2a_to_cosi(b2a, q0=config.q0):
    return np.sqrt((b2a ** 2 - q0 ** 2) / (1 - q0 ** 2)) #if b2a > q0 else 0.

def deproject(name, height, width, x1=0, y1=0, q0=config.q0):
    '''
    Deproject the galaxy coordinates using rotation matrix.
    '''
    cx, cy = find_center(name)
    PA, b2a, distance = get_morph(name)
    cosi = b2a_to_cosi(b2a, q0=q0)
    theta = PA * np.pi / 180
    dep_mat = np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta) / cosi, np.cos(theta) / cosi]])
    
    x0, y0 = np.meshgrid(range(width), range(height))
    x0, y0 = x0.reshape(-1), y0.reshape(-1)
    xy_mat = np.stack([x0 - (cx - x1), y0 - (cy - y1)], axis=0)
    X, Y = np.dot(dep_mat, xy_mat)
    return (X, Y)

def get_beam(name, kpc_per_arcsec=None, PSF=read_file(name='/PSF/PSF.csv')):
    if kpc_per_arcsec is None:
        kpc_per_arcsec = get_morph(name)[2] * constant.arcsec
    FWHM = get_one_value(name, 'FWHM', obj_catalog=PSF)
    sigma_FWHM = get_one_value(name, 'FWHM', obj_catalog=PSF)
    return (  FWHM / np.sqrt(np.log(256)) * kpc_per_arcsec,
        sigma_FWHM / np.sqrt(np.log(256)) * kpc_per_arcsec)


# ========== utils for 1d np arrays ==========

def inv_where(val_arr, bool_arr, padding=np.nan):
    if int(np.sum(bool_arr)) != len(val_arr):
        raise ValueError("The number of 'True' in bool_arr should be equal to" +
                         "the length of val_arr!")
    res = np.ones(len(bool_arr)) * padding
    flag = 0
    for i in range(len(bool_arr)):
        if bool_arr[i]:
            res[i] = val_arr[flag]
            flag += 1
    return res

def bin_array(r, f, bin_size=.1, adp=config.adp_bin): # phase=0
    if adp:
        hist, bin_edge = np.histogram(r, bins='auto')
    else:
        bin_edge = np.arange(min(r), max(r) + bin_size, bin_size)
    stat = binned_statistic(r, f, bins=bin_edge)
    mask = ~np.isnan(stat.statistic)
    bin_r = stat.bin_edges[:-1][mask]
    bin_f = stat.statistic[mask]
    return bin_r, bin_f

def step(rad, met, bin_rad, bin_met):
    rad_matrix = abs(np.subtract.outer(rad, bin_rad))
    min_value = np.expand_dims(np.min(rad_matrix, axis=1), axis=1)
    x, y = np.where(rad_matrix == min_value)  # np.where will return ALL the min values!
    y = y[np.insert(np.diff(x) != 0, 0, True)]  # If having recurring items, remove them.
    step_func = bin_met[y]
    fluc = met - step_func
    return fluc

def tpcf(f, x, y, bin_size=.1, max_kpc=5.):
    mean2, sigma2 = np.mean(f) ** 2, np.std(f) ** 2  # mean is 0
    bin_ind, bin_scorr = bin_stat(f, x, y, bin_size=bin_size, max_kpc=max_kpc)
    bin_d = np.arange(0, max_kpc + bin_size, bin_size)[:len(bin_scorr)]
    bin_s = (bin_scorr - mean2) / sigma2
    return bin_d, bin_s


# ========== utils for 2d np arrays ==========


def bin_map(matrix, factor, keep_shape=True):
    mat = np.where(~np.isnan(matrix), matrix, 0.)
    height, width = mat.shape
    res_h, res_w = int(height / factor), int(width / factor)
    tail_h, tail_w = height % factor, width % factor
    in_mat = mat[:height - tail_h, :width - tail_w]
    in_h, in_w = in_mat.shape
    out_mat = in_mat.reshape(res_h, factor, res_w, factor).mean(1).mean(2)
    if keep_shape:
        res = np.kron(out_mat, np.ones((factor, factor)))
        if width % factor != 0:
            res = np.concatenate([res, np.tile(res[:, -1], (tail_w, 1)).T], axis=1)
        if height % factor != 0:
            res = np.concatenate([res, np.tile(res[-1], (tail_h, 1))], axis=0)
        return res
    else:
        return out_mat
    
def multi_order_bin(signal, noise, min_SN, keep_zero=True):
    shape = signal.shape
    
    res_signal = signal.copy()
    res_noise = noise.copy()
    maps = np.zeros(shape)
    sn = signal / noise

    for i in range(1, int(np.log(min(shape)) / np.log(2)) + 1):
        k = 2 ** i
        s = bin_map(signal, k)
        n = np.sqrt(bin_map(noise**2, k)) / k
        res_signal = np.where(sn > min_SN, res_signal, s)
        res_noise = np.where(sn > min_SN, res_noise, n)
        maps[res_signal == s] = i
        sn = s / n

    if not keep_zero:
        res_signal = np.where(signal > 0., res_signal, 0.)
        res_noise = np.where(signal > 0., res_noise, -1.)
    return res_signal, res_noise, maps

def reconstruct_maps(signal, noise, maps):
    s_list, n_list = [signal], [noise]
    s, n = np.zeros((signal.shape)), np.zeros((noise.shape))
    for i in range(1, int(np.max(maps) + 1)):
        k = 2 ** i
        s_list.append(bin_map(signal, k))
        n_list.append(np.sqrt(bin_map(noise**2, k)) / k)
    for i in range(int(np.max(maps) + 1)):
        s = np.where(maps == i, s_list[i], s)
        n = np.where(maps == i, n_list[i], n)
    return s, n

def display_map(signal, raw_noise, ax,
                log=True, vrange=(-3, 0), center=None, min_SN=config.min_SN,
                cmap=plt.cm.RdYlBu_r, fontsize=20, long_return=False, skip_arcsec=False):
    
    noise = np.where(raw_noise != 0., raw_noise, np.inf)
    img = np.flipud(np.where(signal / noise > min_SN, signal, np.nan))
    h, w = signal.shape
    if not skip_arcsec:
        h *= constant.arcsec_per_pix; w *= constant.arcsec_per_pix
    if log:
        img = np.log10(img)

    if center is None:
        extent = [-.5 * w, .5 * w, -.5 * h, .5 * h]
    else:
        cx, cy = center
        cx *= constant.arcsec_per_pix; cy *= constant.arcsec_per_pix
        extent = [-cx, w - cx, -cy, h - cy]

    if min_SN < 0.:
        im = ax.imshow(img, cmap=plt.cm.get_cmap('RdYlBu', int(np.ptp(img))), extent=extent)
        if long_return:
            return im, np.array(extent)
        else:
            return im
    else:
        print("[display_map]: %d pixels with S/N > %d.\n" %(np.sum(signal / noise > min_SN), min_SN))
        im = ax.imshow(img, cmap=cmap, vmin=vrange[0], vmax=vrange[1], extent=extent)
        if long_return:
            return im, np.array(extent)
        else:
            return im

def plot_cross(ax, name, cen_coords, bar_length=5.):
    x0, y0 = cen_coords
    PA, b2a, distance = get_morph(name)
    kpc_per_arcsec = distance * constant.arcsec
    cosi = b2a_to_cosi(b2a)
    k1, k2 = np.tan(PA * np.pi / 180), -1. / np.tan(PA * np.pi / 180)
    dx1 = bar_length / kpc_per_arcsec * np.abs(np.cos(PA * np.pi / 180))
    dx2 = bar_length / kpc_per_arcsec * np.abs(np.sin(PA * np.pi / 180)) * cosi
    x1 = np.arange(x0 - .5 * dx1, x0 + .5 * dx1, 1e-4)
    x2 = np.arange(x0 - .5 * dx2, x0 + .5 * dx2, 1e-4)
    ax.plot(x1, k1 * x1 - k1 * x0 + y0, color='k', lw=2)
    ax.plot(x2, k2 * x2 - k2 * x0 + y0, color='k', lw=2)

def read_eline(name, eline, recon=False, min_SN=config.min_SN):
    eline_data = fits.open(fits_path + 'flux_elines.' +  name + '.cube.fits')[0].data
    signal = eline_data[config.eline_dict[eline]]
    signal = np.where(signal > 0., signal, 0.)
    noise = eline_data[config.eline_dict[eline] + config.diff]
    noise = np.where(noise > 0., noise, np.mean(noise))
    if recon:
        SII, SII_err = read_eline(name, 'SII6731', recon=False)
        maps = multi_order_bin(SII, SII_err, min_SN)[2]
        return reconstruct_maps(signal, noise, maps)
    else:
        return signal, noise







