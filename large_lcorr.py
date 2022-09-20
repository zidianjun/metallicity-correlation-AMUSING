
import config
from constant import arcsec_per_pix as app
from paths import *
from utils import display_map, multi_order_bin, reconstruct_maps
from utils import get_morph, b2a_to_cosi, read_eline, plot_cross, inv_where
from galmet import Galaxy

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
if server_mode:
    plt.switch_backend('agg')


fontsize = 20

def _file_path(name):
    return obj_path + 'dss-poss1red-' +  name + '.fits'

def _read_photometry(name):
    return fits.open(_file_path(name))[0].data

def _get_wcs(file_path):
    return WCS(fits.open(file_path)[0].header)

def _convert_coord(coord, name):
    '''
    This function could only be used when
    number of from_wcs axes is 3 and number of to_wcs axes is 2!
    '''
    coords = [coord[0], coord[1], 0]
    from_wcs = _get_wcs(fits_path + 'flux_elines.' +  name + '.cube.fits')
    to_wcs = _get_wcs(_file_path(name))
    return to_wcs.world_to_pixel(from_wcs.pixel_to_world(*coords)[0])

def _draw_box(ax, coords, color):

    x1, y1, x2, y2 = coords
    ax.vlines(x1, y1, y2, color=color)
    ax.vlines(x2, y1, y2, color=color)
    ax.hlines(y1, x1, x2, color=color)
    ax.hlines(y2, x1, x2, color=color)

def _arrow_cross_ax(fig, ax1, ax2, shape1, shape2, name, scale_factor=1.,
                    opt=dict(color='gray', connectionstyle='arc3, rad=0.',
                        arrowstyle='simple, head_length=1e-3, head_width=1e-3, tail_width=.4')):
    h1, w1 = shape1
    h2, w2 = shape2

    XY2 = (0, .5 * h2 * (1 - scale_factor))
    xy2 = (XY2[0] * app - .5 * w2 * app, XY2[1] * app - .5 * h2 * app)
    XY1 = _convert_coord((0, 0), name)
    XY1 = (XY1[0], XY1[1])
    xy1 = (XY1[0] - .5 * w1, XY1[1] - .5 * h1)
    transFigure = fig.transFigure.inverted()
    ax1.plot(*xy1, alpha=0.)
    ax2.plot(*xy2, alpha=0.)
    ax1.scatter(*xy1, color='r', alpha=0.)
    ax2.scatter(*xy2, color='r', alpha=0.)
    coord1 = transFigure.transform(ax1.transData.transform(xy1))
    coord2 = transFigure.transform(ax2.transData.transform(xy2))
    arrow = patches.FancyArrowPatch(coord1, coord2, transform=fig.transFigure, **opt)
    fig.patches.append(arrow)

    XY2 = (0, .5 * h2 * (1 + scale_factor))
    xy2 = (XY2[0] * app - .5 * w2 * app, XY2[1] * app - .5 * h2 * app)
    XY1 = _convert_coord((0, h2), name)
    XY1 = (XY1[0], XY1[1])
    xy1 = (XY1[0] - .5 * w1, XY1[1] - .5 * h1)
    transFigure = fig.transFigure.inverted()
    ax1.plot(*xy1, alpha=0.)
    ax2.plot(*xy2, alpha=0.)
    ax1.scatter(*xy1, color='r', alpha=0.)
    ax2.scatter(*xy2, color='r', alpha=0.)
    coord1 = transFigure.transform(ax1.transData.transform(xy1))
    coord2 = transFigure.transform(ax2.transData.transform(xy2))
    arrow = patches.FancyArrowPatch(coord1, coord2, transform=fig.transFigure, **opt)
    fig.patches.append(arrow)

    XY1 = _convert_coord((0, 0), name)
    XY1 = (XY1[0], XY1[1])
    xy1 = (XY1[0] - .5 * w1, XY1[1] - .5 * h1)
    coords = [xy1[0], xy1[1]]
    XY1 = _convert_coord((w2, h2), name)
    XY1 = (XY1[0], XY1[1])
    xy1 = (XY1[0] - .5 * w1, XY1[1] - .5 * h1)
    coords += [xy1[0], xy1[1]]
    _draw_box(ax1, coords, color='#df1d27')


def _ra2hex(ra):
    x = ra / 15
    hh = int(x)
    mm = int((x - hh) * 60)
    ss = (x - hh - mm / 60) * 3600
    return hh, mm, ss

def _dec2hex(dec):
    x = np.abs(dec)
    dd = int(x) if dec > 0 else -int(x)
    mm = int((x - np.abs(dd)) * 60)
    ss = (x - np.abs(dd) - mm / 60) * 3600
    return dd, mm, ss

def _pixel2hex(name, coord):
    coords = [coord[0], coord[1], 0]
    wcs = _get_wcs(fits_path + 'flux_elines.' +  name + '.cube.fits')
    str_coords = wcs.pixel_to_world(*coords)[0].to_string()
    ind = str_coords.index(' ')
    ra, dec = float(str_coords[:ind]), float(str_coords[ind+1:])
    return _ra2hex(ra), _dec2hex(dec)


def show_map(name1, name2):

    fig = plt.figure(figsize=(13.3, 10))
    plt.subplots_adjust(left=.04, bottom=.02, right=.98, top=.85)
    
    ax1 = plt.subplot(231)
    bbim = _read_photometry(name1)
    display_map(bbim, 1., ax1, cmap=plt.cm.binary, log=False,
                vrange=(np.min(bbim), np.max(bbim)), skip_arcsec=True)
    ax1.tick_params(axis='both', labelleft=False, labelbottom=False, labelsize=fontsize)

    ax = plt.subplot(232)
    signal, noise = read_eline(name1, 'Halpha')
    im = display_map(signal, noise, ax, cmap=plt.cm.binary, log=True, vrange=(-2, 1))
    ax.annotate(name1, xy=(40, 5), fontsize=fontsize)
    cbar_ax = fig.add_axes([0.37, 0.94, 0.28, 0.03])
    cbar = plt.colorbar(im, orientation='horizontal', ticks=[-2, -1, 0, 1.], cax=cbar_ax)
    cbar.ax.set_xticklabels(['10$^{-2}$', '10$^{-1}$', '10$^{0}$', '10$^{1}$'])
    cbar.set_label('Line flux ($10^{-16}$ erg)', size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.set_xlim(-.5 * signal.shape[1] * app, .5 * signal.shape[1] * app)
    ax.set_ylim(-.5 * signal.shape[0] * app, .5 * signal.shape[0] * app)
    x_ticks = (np.array([12, 162, 312]) - .5 * signal.shape[1]) * app
    x_labels = ['08s', '20h33m06s', '04s']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('RA (J2000)', fontsize=fontsize)
    ax.set_ylabel('Dec. (J2000)', fontsize=fontsize)
    ax.tick_params(axis='both', labelleft=False, labelsize=fontsize)
    _arrow_cross_ax(fig, ax1, ax, bbim.shape, signal.shape, name1, scale_factor=.96)

    ax = plt.subplot(233)
    galaxy = Galaxy(name1, map_mode=True)
    met_fluc = inv_where(galaxy.met_fluc, galaxy.mask_list[0]).reshape(signal.shape)
    im = display_map(met_fluc, met_fluc, ax, min_SN=0., log=False, vrange=(-.4, .4))
    plot_cross(ax, name1, (21, -15))
    ax.annotate('5 kpc', xy=(16, -25), fontsize=fontsize)
    cbar_ax = fig.add_axes([0.7, 0.94, 0.28, 0.03])
    cbar = plt.colorbar(im, orientation='horizontal', cax=cbar_ax)
    cbar.set_label('$\\Delta$ (O/H)', size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.set_xlim(-.5 * signal.shape[1] * app, .5 * signal.shape[1] * app)
    ax.set_ylim(-.5 * signal.shape[0] * app, .5 * signal.shape[0] * app)
    y_ticks = (np.array([1.3, 151.3, 301.3]) - .5 * signal.shape[0]) * app
    y_labels = ["$-02^{\circ}$\n$02'$\n$30''$", "$02'$\n$00''$", "$01'$\n$30''$"]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    x_ticks = (np.array([12, 162, 312]) - .5 * signal.shape[1]) * app
    x_labels = ['08s', '20h33m06s', '04s']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('RA (J2000)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    ax4 = plt.subplot(234)
    bbim = _read_photometry(name2)
    display_map(bbim, 1., ax4, cmap=plt.cm.binary, log=False,
                vrange=(np.min(bbim), np.max(bbim)), skip_arcsec=True)
    ax4.tick_params(axis='both', labelleft=False, labelbottom=False, labelsize=fontsize)

    ax = plt.subplot(235)
    signal, noise = read_eline(name2, 'Halpha')
    im = display_map(signal, noise, ax, cmap=plt.cm.binary, log=True, vrange=(-2, 1))
    ax.annotate(name2, xy=(75, 10), fontsize=fontsize)
    ax.set_xlim(-.5 * signal.shape[1] * app, .5 * signal.shape[1] * app)
    ax.set_ylim(-.5 * signal.shape[0] * app, .5 * signal.shape[0] * app)
    x_ticks = (np.array([106, 412]) - .5 * signal.shape[1]) * app
    x_labels = ['48s', '21h13m44s']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('RA (J2000)', fontsize=fontsize)
    ax.set_ylabel('Dec. (J2000)', fontsize=fontsize)
    ax.tick_params(axis='both', labelleft=False, labelsize=fontsize)
    _arrow_cross_ax(fig, ax4, ax, bbim.shape, signal.shape, name2, scale_factor=.52)

    ax = plt.subplot(236)
    galaxy = Galaxy(name2, map_mode=True)
    met_fluc = inv_where(galaxy.met_fluc, galaxy.mask_list[0]).reshape(signal.shape)
    im = display_map(met_fluc, met_fluc, ax, min_SN=0., log=False, vrange=(-.4, .4))
    plot_cross(ax, name2, (35, -10))
    ax.annotate('5 kpc', xy=(30, -25), fontsize=fontsize)
    ax.set_xlim(-.5 * signal.shape[1] * app, .5 * signal.shape[1] * app)
    ax.set_ylim(-.5 * signal.shape[0] * app, .5 * signal.shape[0] * app)
    y_ticks = (np.array([146.5, 296.5]) - .5 * signal.shape[0]) * app
    y_labels = ["$02^{\circ}$\n$28'$\n$30''$", "$29'$\n$00''$"]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    x_ticks = (np.array([106, 412]) - .5 * signal.shape[1]) * app
    x_labels = ['48s', '21h13m44s']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('RA (J2000)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    if server_mode:
        plt.savefig(savefig_path + 'large_lcorr.pdf')




name1, name2 = 'NGC6926-S', 'JO206'
show_map(name1, name2)
plt.show()








