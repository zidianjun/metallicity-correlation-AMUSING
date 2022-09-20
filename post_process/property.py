
from hub import DataHub, red, green, blue
from paths import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr
if server_mode:
    plt.switch_backend('agg')


califa_output_path = obj_path + '/prev/'

x_info = {'Mass': ((9., 11.7, 20), (8e7, 9e11), 'Stellar mass (M$_{\odot}$)'),
          'lSFR': ((-1.2, 1.6, 20), (5e-3, 7e1), 'SFR (M$_{\odot}$ yr$^{-1}$)'),
          'Re': ((-1.5, 1.2, 20), (2e-1, 20), '$R_e$ (kpc)')}

y_info = {'lcorr': ((-1.4, 1, 20), (2e-2, 9e1), 'Correlation length (kpc)', (2e-2, 6e0)),
          'winj': ((-1.4, 1, 20), (1.5e0, 2e3), 'Injection width (pc)', (2e0, 5e2))}

opt = dict(color='gray', alpha=.5, connectionstyle='arc3, rad=0',
           arrowstyle='simple, head_width=.25, head_length=.5, tail_width=.03')

scale_height = .15  # kpc
tau_dep_const = 2.2  # Gyr
n0 = 1  # cm^{-3}


def transport_coefficient(f_sf, f_g, v_phi_out,
                          Q=1., phi_a=2, phi_mp=1.4, eff=0.02,
                          G=6.67e-20, # km/s
                          unit=6.3e22, # Convert from kg/s to Msun/yr
                          ):
    return 16./np.sqrt(3*phi_mp) * phi_a * eff / (np.pi * G * Q**2) * (
           f_sf * f_g**1.5 * v_phi_out**2) / unit

def feedback_coefficient(f_g, v_phi_out,
                         Q=1., phi_a=2, phi_mp=1.4, phi_nt=1.,
                         phi_Q=2., p2m=3000, eta=1.5, G=6.67e-20, # km/s
                         unit=6.3e22, # Convert from kg/s to Msun/yr
                         ):
    return 4 * eta * np.sqrt(phi_mp * phi_nt**3 * phi_Q**2 * phi_a**2) / (
           G * Q * p2m) * f_g * v_phi_out**2 / unit

def K18_model(SFR, gtype='spiral', v_phi_out=200.):
    if gtype not in ['spiral', 'dwarf', 'high-z']:
        raise ValueError("label should only be 'spiral', " +
                         "'dwarf', or 'high-z'!")
    # Default T+F
    par_dict = {'spiral': (.5, .5, 10.), 'dwarf': (.2, .9, 5.)}
    f_sf, f_g, min_vel_disp = par_dict[gtype]

    T_coeff = transport_coefficient(f_sf, f_g, v_phi_out=v_phi_out)
    
    return np.where(SFR / T_coeff > min_vel_disp, SFR / T_coeff, min_vel_disp)


def hist2d(xdata, ydata, x_range, y_range=(-1.4, 1, 20)):
    x0, x1, nx = x_range
    y0, y1, ny = y_range
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny

    x, y = np.meshgrid(np.linspace(*x_range), np.linspace(*y_range))

    count = np.zeros(nx * ny)
    for i in range(len(xdata)):
        if xdata[i] > 0. and ydata[i] > 0.:
            ind_x = int((np.log10(xdata[i]) - x0) / dx)
            ind_y = int((np.log10(ydata[i]) - y0) / dy)
            ind = ind_y * nx + ind_x
            if 0 < ind < nx * ny:
                count[ind] += 1

    c = np.log10(np.where(count > 0, count, 1) / np.max(count))
    return (x, y, c.reshape((nx, ny)))

def scatter_map(i, x_axis, y_axis, dh=DataHub()):
    x_range, x_lim, x_label = x_info[x_axis]
    y_range, y_lim, y_label, v_range = y_info[y_axis]
    
    if 'corr' in y_axis:
        v50, v16, v84, v10 = dh.corr_len(long_return=True)
        uplim = (v10 ** 2 < v50 ** 2 / 5.) | (v16 ** 2 < v50 ** 2 / 3.)
        # corr_len if from sqrt(kappat_*)
    else:
        v50, v16, v84, v10 = dh.inj_width(long_return=True)
        uplim = (v10 < v50 / 5.) | (v16 < v50 / 3.)
    
    ax = plt.subplot(131 + i)
    ax.set_xscale('log')
    ax.set_yscale('log')

    L21_flag = 0
    lines0, labels0 = [], []
    if x_axis in ['Mass', 'lSFR', 'Re']:
        # L21 for correlation length
        if y_axis == 'lcorr':
            L21_flag = 1
            if x_axis == 'Re':
                x_plot = dh.col(x_axis)
            else:
                x_plot = 10 ** dh.col(x_axis)
                ax.hlines(v50[~uplim], 10 ** (dh.col(x_axis) - dh.col('error_' + x_axis))[~uplim],
                      10 ** (dh.col(x_axis) + dh.col('error_' + x_axis))[~uplim], color='gray')

            lines0.append(ax.scatter([-1], [-1], color='gray', edgecolor='gray', marker='s', s=100))
            labels0.append('CALIFA (L21)')       

            xdata, ydata = np.load(califa_output_path + x_axis + '.npy')
            x_grid, y_grid, c_grid = hist2d(xdata, ydata, x_range=x_range, y_range=y_range)
            im = ax.pcolor(10 ** x_grid, 10 ** y_grid, c_grid, cmap=plt.cm.binary,
                            vmin=-1.5, zorder=0)

        else:
            im = None
            if x_axis == 'Re':
                x_plot = dh.col(x_axis)
            else:
                x_plot = 10 ** dh.col(x_axis)
                ax.hlines(v50[~uplim],
                          10 ** (dh.col(x_axis) - dh.col('error_' + x_axis))[~uplim],
                          10 ** (dh.col(x_axis) + dh.col('error_' + x_axis))[~uplim],
                          color='gray')

    else:
        im = None
        x_plot = dh.col(x_axis)

    lines0.append(ax.scatter(x_plot[~uplim], v50[~uplim], color='w',
                  edgecolor='k', s=80, zorder=4))
    fig = ax.scatter(x_plot[~uplim], v50[~uplim], c=SFRSD[~uplim], edgecolor='k', s=80, zorder=4)
    lines0.append(fig)
    labels0.append('AMUSING++' + ' (this work)' * L21_flag)
    ax.vlines(x_plot[~uplim], v16[~uplim], v84[~uplim], color='gray', alpha=.4)

    for x, y in zip(x_plot[uplim], v84[uplim]):
        ax.scatter(x, y, color='w', edgecolor='gray', s=40, zorder=2)
        ax.annotate("", xy=(x, .75*y), xytext=(x, y), arrowprops=opt)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel(x_label, fontsize=20)
    if i < 1:
        ax.set_ylabel(y_label, fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
    else:
        ax.tick_params(axis='both', labelsize=20, labelleft=False)
    
    legend0 = ax.legend(lines0, labels0, loc='upper left', fontsize=15)
    plt.gca().add_artist(legend0)

    return im



def SFR_insight(dh=DataHub()):
    x_range, x_lim, x_label = x_info['lSFR']
    y_range, y_lim, y_label, v_range = y_info['lcorr']
    
    v50, v16, v84, v10 = dh.corr_len(long_return=True)
    uplim = (v10 ** 2 < v50 ** 2 / 5.) | (v16 ** 2 < v50 ** 2 / 3.)

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    lines0, labels0 = [], []
    
    x_plot = 10 ** dh.col('lSFR')
    ax.hlines(v50[~uplim],
              10 ** (dh.col('lSFR') - dh.col('error_lSFR'))[~uplim],
              10 ** (dh.col('lSFR') + dh.col('error_lSFR'))[~uplim],
              color='gray')

    xx = np.arange(1e-5, 1e2, 1e-3)
    lines, labels = [], []

    for t, color in zip([.1, .5, 2.], [red, green, blue]):
        vel_disp = K18_model(xx)
        ax.plot(xx, np.sqrt(1./3 * scale_height * vel_disp * t),
                color=color, ls='-', alpha=.5, zorder=10)
        lines.append(ax.plot(np.nan, np.nan, color=color, ls='-')[0])
        labels.append(r'$\tau_{\mathrm{eq}} = $' + str(t) + ' Gyr')


    # ax.scatter(x_plot[~uplim], v50[~uplim], color='w', edgecolor='k', s=80, zorder=4)
    SFRSD = np.log10(10 ** dh.col('lSFR') / np.pi / dh.col('Re') ** 2)
    fig = ax.scatter(x_plot[~uplim], v50[~uplim], c=SFRSD[~uplim], edgecolor='k', s=80,
                     cmap=plt.cm.RdYlBu_r, zorder=4)
    ticks = range(-3, 1)
    ticklabels = []
    for tick in ticks:
        ticklabels.append('10$^{%d}$' %(tick))
    cbar = plt.colorbar(fig, ticks=ticks)
    cbar.ax.set_yticklabels(ticklabels, fontsize=15)
    cbar.set_label('$\Sigma_{\mathrm{SFR}}$ (M$_{\odot}$ yr$^{-1}$ kpc$^{-2}$)', size=15)
    ax.vlines(x_plot[~uplim], v16[~uplim], v84[~uplim], color='gray')   

    for x, y in zip(x_plot[uplim], v84[uplim]):
        ax.scatter(x, y, color='w', edgecolor='gray', s=40, zorder=2)
        ax.annotate("", xy=(x, .75*y), xytext=(x, y), arrowprops=opt)


    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim[0], 3e1)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    
    legend = ax.legend(lines, labels, loc='upper left', fontsize=15)
    plt.gca().add_artist(legend)

    if server_mode:
        plt.savefig(savefig_path + '/property/SFR.pdf')



def properties(y_axis='lcorr'):
    fig = plt.figure(figsize=(18, 6))
    plt.subplots_adjust(left=.07, bottom=.15, right=.9, top=.95, wspace=0.)

    for i, x_axis in enumerate(['Mass', 'lSFR', 'Re']):
        im = scatter_map(i, x_axis, y_axis)

    if y_axis == 'lcorr':
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.8])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Log probability density', size=20)
        cbar.ax.tick_params(labelsize=20)

    if server_mode:
        plt.savefig(savefig_path + '/property/property_' + y_axis + '.pdf')



properties(y_axis='lcorr')
properties(y_axis='winj')
SFR_insight()

plt.show()



