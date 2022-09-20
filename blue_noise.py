
from utils import tpcf
import constant
from utils import reconstruct_maps

import numpy as np
from cv2 import GaussianBlur


def _gen_blue_noise(mask, kpc_per_pix, beam,
                    maps=None, height=325, width=325):
    '''
    Height and width are in the unit of pixel.
    1 pixel in MUSE is 0.2 arcsec, thus converting beam into the unit of pixel.
    The kernel size must be odd and here the default value is 15.
    '''
    noise = GaussianBlur(np.random.normal(0, 1, (height, width)), (15, 15),
                         beam / kpc_per_pix)
    if maps is not None:
        noise, _ = reconstruct_maps(noise, noise, maps)
    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    return (noise.reshape(-1)[mask],
    	    x.reshape(-1)[mask] * kpc_per_pix,
    	    y.reshape(-1)[mask] * kpc_per_pix)

def gen_blue_noise_band(mask_list, kpc_per_pix, beam,
                        maps=None, height=325, width=325, times=10):
    ksi_b = None
    len_mask = len(mask_list) - 1
    for i in range(times):
        x, ksi = tpcf(*_gen_blue_noise(mask_list[min(i, len_mask)], kpc_per_pix, beam,
                      maps=maps, height=height, width=width), bin_size=kpc_per_pix)
        ksi = np.expand_dims(ksi, axis=0)
        ksi_b = np.concatenate((ksi_b, ksi), axis=0) if ksi_b is not None else ksi
        print("Bootstrap for blue noise #%d finished." %(i))
    ksi_b_mean = np.mean(ksi_b, axis=0)
    ksi_b_std = np.std(ksi_b, axis=0)
    return x, ksi_b_mean, ksi_b_std




