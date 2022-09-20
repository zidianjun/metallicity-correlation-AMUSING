
import config
from paths import *
from utils import read_file

from astropy.io import fits
import numpy as np
from os.path import isfile
import pandas as pd
from scipy.optimize import leastsq
from scipy.odr import Model, RealData, ODR
from urllib.request import urlopen
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')

def _extract_info(line, mark1, mark2):
    i1 = line.find(mark1) + len(mark1)
    l = line[i1:]
    i2 = l.find(mark2)
    return l[:i2]

def _name_matches(name, line):
    extract_name = _extract_info(line, '<td>&nbsp;</td><td>', '<').replace('&nbsp;', '')
    if len(name) <= len(extract_name):
        return name.lower() in extract_name.lower()
    else:
        return extract_name.lower() in name.lower()
    
def _ao_info(header):
    if 'TITLE' in header.keys():
        return _extract_info(header['TITLE'], 'FM-', '-')
    else:
        return None

def _ao_matches(header, line):
    ao = _ao_info(header)
    if ao is not None:
        return ao.lower() == _extract_info(line, 'fm-', '_').lower()
    else:
        return True

def _pull_down(s):
    try:
        sgs = float(_extract_info(s, 'HIERARCH ESO OCS SGS FWHM MED = ', ' / '))
    except:
        sgs = 0.
    try:
        tel = float(_extract_info(s, 'HIERARCH ESO TEL IA FWHMLINOBS = ', ' / '))
    except:
        tel = 0.
    return sgs, tel

def write_allPSF(allpsf_dir):
    line_list = open(obj_path + '/PSF/query_source.html', 'r').read().splitlines()
    name_list, sgs_list, tel_list = [], [], []
    for name in read_file().name:
        header = fits.open(fits_path + 'flux_elines.' + name + '.cube.fits')[0].header
        i1 = line_list.index('<P>') + 1
        line_list = line_list[i1:]
        temp = line_list[:line_list.index('<P>')]
        for line in temp:
            if _name_matches(name, line) and _ao_matches(header, line):
                url = 'http://archive.eso.org' + _extract_info(line, '" href="', '"')
                sgs, tel = _pull_down(str(urlopen(url).read()))
                name_list.append(name)
                sgs_list.append(sgs)
                tel_list.append(tel)
                print('%s,%.3f,%.2f' %(name, sgs, tel))

        df = pd.DataFrame({'name': name_list, 'SGS': sgs_list, 'TEL': tel_list})
        df[['name', 'SGS', 'TEL']].to_csv(allpsf_dir, index=False)

def _linear(par, x):
    k, b = par
    return k * x + b

def _ols(x, y):
    return leastsq(lambda par, x, y: _linear(par, x) - y, [.9, .05], args=(x, y))[0]

def _tls(x, y):
    return ODR(RealData(x, y), Model(_linear), beta0=[.9, .05]).run().beta

def plot_fit(allpsf_dir):
    total = pd.read_csv(allpsf_dir)
    good = (total.TEL < 2.) & (total.SGS > 0.) & (total.SGS < 2.)
    x, y = total.TEL[good], total.SGS[good]
    # k1, b1 = _ols(x, y)  # k1 = 0.610, b1 = 0.131
    # print('k1 = %.3f, b1 = %.3f' %(k1, b1))
    k, b = _tls(x, y)  # k2 = 0.787, b2 = -0.082
    print('k = %.3f, b = %.3f' %(k, b))

    plt.figure(figsize=(8, 8))
    ax = plt.axes([.15, .15, .8, .8])
    xx = np.arange(.3, 3., .01)
    ax.scatter(x, y, color='gray', s=4)
    # ax.plot(xx, k1 * xx + b1, color='r', alpha=.75, label='OLS')
    ax.plot(xx, k * xx + b, color='k', alpha=.75, label='$y = 0.787x - 0.082$')
    ax.plot(xx, xx, color='gray', ls='--')
    ax.set_xlim(.3, 2.1)
    ax.set_ylim(.3, 2.1)
    ax.set_xlabel('TEL IA FWHMLINOBS (arcsec)', fontsize=20)
    ax.set_ylabel('SGS FWHM (arcsec)', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(loc='upper left', fontsize=20)

    if server_mode:
        plt.savefig(savefig_path + '/PSF/FWHM.pdf')

def write_PSF(psf_dir, allpsf_dir):
    name_list = read_file().name
    a = pd.read_csv(allpsf_dir)
    value, sigma = [], []
    for name in name_list:
        ind = a.name == name
        arr = np.where(a.SGS[ind] > 0., a.SGS[ind], 0.787 * a.TEL[ind] - 0.082)
        value.append(np.sqrt(np.mean(arr ** 2)))
        sigma.append(np.std(arr))
    df = pd.DataFrame({'name': name_list, 'FWHM': value, 'sigma_FWHM': sigma})
    df[['name', 'FWHM', 'sigma_FWHM']].round(3).to_csv(psf_dir, index=False)


allpsf_dir = obj_path + '/PSF/allPSF.csv'
if isfile(allpsf_dir):
    pass
else:
    write_allPSF(allpsf_dir)  # Roughly takes 3h.

psf_dir = obj_path + '/PSF/PSF.csv'
write_PSF(psf_dir, allpsf_dir)

plot_fit(allpsf_dir)


