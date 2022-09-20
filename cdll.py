
import constant

from ctypes import *
import numpy as np
import matplotlib.pyplot as plt

lib = CDLL("./lib.so")

def bin_stat(f, x, y, bin_size=.1, max_kpc=5.):
    three_arrays = np.concatenate([f, x, y], axis=0)
    lib.group_by.argtypes = (POINTER(c_float), c_int, c_int, c_float, c_float)
    lib.group_by.restype = POINTER(c_float)
    c_array = np.ctypeslib.as_ctypes(three_arrays.astype(np.float32))
    length = int(max_kpc / bin_size) + 1
    c_res = lib.group_by(c_array, len(f), length, bin_size, max_kpc)
    py_res = cast(c_res, POINTER(c_float * (length * 2))).contents
    res = np.array(list(py_res), dtype=float).reshape(2, -1)
    return res[0], res[1]

def iter_met(r_arr, S23_arr, diag='K19D16'):
    coeff_logOH = getattr(constant, 'coeff_logOH_' + diag[3:])
    coeff_logU =  getattr(constant, 'coeff_logU_S23')
    four_arrays = np.concatenate([r_arr, S23_arr, coeff_logOH, coeff_logU], axis=0)

    lib.iteration.argtypes = (POINTER(c_float), c_int)
    lib.iteration.restype = (POINTER(c_float))
    c_array = np.ctypeslib.as_ctypes(four_arrays.astype(np.float32))
    length = len(r_arr)
    c_res = lib.iteration(c_array, length)
    py_res = cast(c_res, POINTER(c_float * (length * 2))).contents
    res = np.array(list(py_res), dtype=float).reshape(2, -1)

    met, ion = res
    nan_mask = ~np.isnan(met)  # nan and 0. are blended.
    met, ion = np.where(nan_mask, met, 0.), np.where(nan_mask, ion, 0.)
    mask = ((met > constant.min_Z) & (met < constant.max_Z) &
            (ion > constant.min_U) & (ion < constant.max_U))
    return np.where(mask, met, np.nan), np.where(mask, ion, np.nan)





