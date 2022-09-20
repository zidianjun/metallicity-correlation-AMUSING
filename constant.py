
import numpy as np

# In MAD survey, 1 pixel equals to 0.2 arcsec
arcsec = 4.848e-3
arcsec_per_pix = .2

line_rest_wavelength_dict = {
    'OII3727': 3727.092,
    'Hbeta': 4862.683,
    'OIII4959': 4960.295,
    'OIII5007': 5008.240,
    'NII6549': 6549.840,
    'Halpha': 6564.610,
    'NII6584': 6585.230,
    'SII6717': 6718.294,
    'SII6731': 6732.674,
    'SIII9069': 9071.1
}

coeff_logU_S23 = [38.1897, 14.8776, -14.5589, -3.2898, 1.7002, 1.6837,
                  0.1966, -0.1710, 0.7205, -0.0644]
coeff_logOH_N2S2 = [5.8892, 3.1688, -3.5991, 1.6394, -2.3939, -1.6764,
                    0.4455, -0.9302, -0.0966, -0.2490]
coeff_logOH_D16 = [8.1964, 1.1850, -0.9534, 0.1477, 0.1358, -0.5363,
                    0.1096, -0.2857, 0.4872, -0.0913]
coeff_logOH_N2 = [10.6383, 4.4704, -1.0809, 0.6063, 1.9142, -0.7227,
                  0.0184, 0.1514, 0.3414, -0.0947]
coeff_logOH_O3N2 = [10.1121, -0.4162, 1.6365, -0.4426, -0.1605, 0.8803,
                    -0.1397, -0.0022, -0.0119, 0.1539]

SIII9531_to_SIII9069 = 2.47

min_Z, max_Z, min_U, max_U = 7.63, 9.23, -3.98, -1.98

f_Ha_NII = 1.00912  # 10 ** (.4 * delta_kappa)
f_Hb_OIII = 1.13344

