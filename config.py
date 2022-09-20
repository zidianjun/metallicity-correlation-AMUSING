
# AMUSING columns
eline_dict = {'OIII5007': 1, 'OIII4959': 2, 'Hbeta': 3, 'Halpha': 20, 'NII6584': 21,
              'NII6549': 22, 'SII6717': 24, 'SII6731': 25, 'SIII9069': 29}

diff = 120

# metallicity diagnostics
available_diag_list = ['PPN2', 'PPO3N2', 'D16',
                       'Scal', # Do not recommend! It requires too many faint lines.
                       'K19N2S2', 'K19D16', 'K19N2', 'K19O3N2']
                       # Do not recommend! S32 ionization parameter diagostic varies.
diag_list = ['PPN2', 'PPO3N2', 'D16']

# galaxy properties
min_SN = 3. # defaulted 3. (changeable)

min_fill_f_A = 4e-2
min_fill_f_p = 4e-1
min_b2a = .4
# Above three do not directly appear in Galaxy class.
# Selection should be done before the loop.
q0 = .13  # .13 for AMUSING thick-disk de-projection (changeable)
EW_criterion = -6  # Sanchez
AGN_criterion = 'Kauffmann'  # or Kewley

# Two-point correlation
adp_bin = False
max_separation = .7 # 0.7 * median of the radii
times = 10

# MCMC
n_dim = 4
n_walker = 100
n_step = 1000
n_sample = 500

