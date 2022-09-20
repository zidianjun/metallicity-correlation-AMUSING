
import constant
from cdll import iter_met
from paths import *

import numpy as np


# ========== utils for derendening ==========

def _kappa(wavelength):
    '''
    Cardelli et al. (1989) has already been corrected in AMUSING++.
    Here we only correct extinction caused by the source galaxy,
        using the extinction curve proposed by Calzetti et al. (2000).
    '''
    Rv = 4.05  # Rv will be cancelled once _kappa(a) - _kappa(b)
    x = 1e4 / wavelength
    if wavelength >= 6300:
        return 2.659 * (-1.857 + 1.040*x) + Rv
    else:
        return 2.659 * (-2.156 + 1.509*x - 0.198*x**2 + 0.011*x**3) + Rv

def dered_f(EBV, wavelength, reddest_wavelength):
    kappa_a = _kappa(wavelength)
    kappa_b = _kappa(reddest_wavelength)
    factor = 10 ** (0.4 * (kappa_a - kappa_b) * EBV)
    return factor
 

# ========== metallicity diagnostics ==========

def PPN2(galaxy):
    N2 = np.log10(galaxy.ratio(['NII6584'], ['Halpha']))
    met = 9.37 + 2.03 * N2 + 1.26 * N2 ** 2 + 0.32 * N2 ** 3
    met = np.nan_to_num(met)
    mask = (galaxy.mask_AGN() & galaxy.mask_EW() & galaxy.mask_Ha() &
            (met > constant.min_Z) & (met < constant.max_Z))
    return met, mask

def PPO3N2(galaxy):
    O3 = galaxy.ratio(['OIII5007'], ['Hbeta'])
    N2 = galaxy.ratio(['NII6584'], ['Halpha'])
    O3N2 = np.log10(O3 / N2)
    met = 8.73 - 0.32 * O3N2
    met = np.nan_to_num(met)
    mask = (galaxy.mask_AGN() & galaxy.mask_EW() & galaxy.mask_Ha() &
            (met > constant.min_Z) & (met < constant.max_Z))
    return met, mask

def Scal(galaxy):
    N2 = galaxy.ratio(['NII6549', 'NII6584'], ['Hbeta'])
    S2 = galaxy.ratio(['SII6717', 'SII6731'], ['Hbeta'])
    R3S2 = galaxy.ratio(['OIII4959', 'OIII5007'], ['SII6717', 'SII6731'])

    met_upper = 8.424 + 0.030 * np.log10(R3S2) + 0.751 * np.log10(N2) + np.log10(S2) * (
               -0.349 + 0.182 * np.log10(R3S2) + 0.508 * np.log10(N2))
    met_lower = 8.072 + 0.789 * np.log10(R3S2) + 0.726 * np.log10(N2) + np.log10(S2) * (
                1.069 - 0.170 * np.log10(R3S2) + 0.022 * np.log10(N2))

    N2_06 = np.where(~np.isnan(N2), np.log10(N2), -.6)
    met_upper = np.where(N2_06 >= -.6, met_upper, 0.)
    met_lower = np.where(N2_06 <= -.6, met_lower, 0.)
    met = met_upper + met_lower
    met = np.nan_to_num(met)
    mask = (galaxy.mask_AGN() & galaxy.mask_EW() & galaxy.mask_Ha() &
            (met > constant.min_Z) & (met < constant.max_Z))
    return met, mask

def D16(galaxy):
    y = (np.log10(galaxy.ratio(['NII6584'], ['SII6717', 'SII6731'])) +
         .264 * np.log10(galaxy.ratio(['NII6584'], ['Halpha'])))
    met = 8.77 + y + 0.45 * (y + 0.3) ** 5
    met = np.nan_to_num(met)
    mask = (galaxy.mask_AGN() & galaxy.mask_EW() & galaxy.mask_Ha() &
            (met > constant.min_Z) & (met < constant.max_Z))
    return met, mask

def K19_diag(galaxy):
    S32 = np.log10((1 + constant.SIII9531_to_SIII9069) *
                   galaxy.ratio(['SIII9069'], ['SII6717', 'SII6731']))

    if galaxy.diag == 'K19N2S2':
        x = np.log10(galaxy.ratio(['NII6584'], ['SII6717', 'SII6731']))
    elif galaxy.diag == 'K19D16':
        x = (np.log10(galaxy.ratio(['NII6584'], ['SII6717', 'SII6731'])) +
             .264 * np.log10(galaxy.ratio(['NII6584'], ['Halpha'])))
    elif galaxy.diag == 'K19N2':
        x = np.log10(galaxy.ratio(['NII6584'], ['Halpha']))
    else: #'K19O3N2':
        x = np.divide(galaxy.ratio(['OIII5007'], ['Hbeta']),
                      galaxy.ratio(['NII6584'], ['Halpha']),
                      out=np.zeros_like(galaxy.ratio(['OIII5007'], ['Hbeta'])),
                      where=galaxy.ratio(['NII6584'], ['Halpha'])!=0)

    met, ion = iter_met(x, S32, diag=galaxy.diag)
    mask = (galaxy.mask_AGN() & galaxy.mask_EW() & galaxy.mask_Ha() & ~np.isnan(met))
    return met, ion, mask




