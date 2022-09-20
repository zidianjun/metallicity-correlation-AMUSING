
import config
from paths import *

import numpy as np


diag = 'D16'
suffix = ''
# suffix = '_q0_35'

lcorr = open(output_path + '/correlation_length_' + diag + suffix + '.csv', 'w')
lcorr.write('name,l_50,l_16,l_84,l_10\n')
winj = open(output_path + '/injection_width_' + diag + suffix + '.csv', 'w')
winj.write('name,w_50,w_16,w_84,w_10\n')
beam = open(output_path + '/sigma_beam_' + diag + suffix + '.csv', 'w')
beam.write('name,s_50,s_16,s_84\n')

chains = open(output_path + '/total_chain_' + diag + suffix + '.txt', 'r')

for line in chains.readlines():
    name = line[:line.index(' ')]
    s = line[(1+line.index(' ')):]
    array = np.array([float(n) for n in s.split()])
    data_cube = np.reshape(array, [config.n_sample, config.n_walker, 3])
    L = data_cube[:, :, 2].reshape(-1)
    l_50, l_16, l_84 = (np.sqrt(np.percentile(L, 50)),
                        np.sqrt(np.percentile(L, 16)),
                        np.sqrt(np.percentile(L, 84)))
    l_10 = np.sqrt(np.percentile(L, 10))
    lcorr.write('%s,%.3f,%.3f,%.3f,%.3f\n' %(name, l_50, l_16, l_84, l_10))
    W = data_cube[:, :, 1].reshape(-1) * 1e3
    w_50, w_16, w_84 = (np.percentile(W, 50),
                        np.percentile(W, 16),
                        np.percentile(W, 84))
    w_10 = np.percentile(W, 10)
    winj.write('%s,%.0f,%.0f,%.0f,%.0f\n' %(name, w_50, w_16, w_84, w_10))
    S = data_cube[:, :, 0].reshape(-1)
    s_50, s_16, s_84 = (np.percentile(S, 50),
                        np.percentile(S, 16),
                        np.percentile(S, 84))
    beam.write('%s,%.3f,%.3f,%.3f\n' %(name, s_50, s_16, s_84))

lcorr.close()
winj.close()
beam.close()
chains.close()


