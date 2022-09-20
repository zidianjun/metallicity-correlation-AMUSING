
import config
from paths import *
from utils import get_beam

import numpy as np
from scipy.integrate import quad
from scipy.special import jv
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.stats import gaussian_kde
import emcee
import corner
from os.path import isfile
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')


x_range, y_range = (-12, 6, .01), (-12, 14, .01)

def gauss_kde_pdf(posterior):
    # Return a function, computing the pdf with the given posterior.
    return gaussian_kde(posterior).pdf

def _KT18_integ(alpha, beta):
    return (2. / np.log(1 + beta / alpha) * quad(lambda x:
            np.exp(-alpha * x**2) * (1 - np.exp(-beta * x**2)) *
            jv(0, x) / x, 0, np.inf)[0])

def tab(name='KT18_table.npy', x_range=x_range, y_range=y_range):
    a, b = np.arange(*x_range), np.arange(*y_range)
    x, y = np.meshgrid(a, b)
    c = np.zeros(len(a) * len(b))
    n = 0
    for ln_alpha in a:
        for ln_beta in b:
            c[n] = max(_KT18_integ(np.e ** ln_alpha, np.e ** ln_beta), 1e-10)
            n += 1
            if n % 10000 == 0: print(n)
    mat = np.stack([x.T.reshape(-1), y.T.reshape(-1), c], axis=1)
    np.save(name, mat)

if isfile('KT18_table.npy'):
    _, _, z = np.load('KT18_table.npy').T
    x, y = np.arange(*x_range), np.arange(*y_range)
    rbs_func = RBS(x, y, z.reshape([len(x), len(y)]))
else:
    tab()

def KT18_model(x_array, beam, x0, KappaTstar):
    alpha = (beam ** 2 / 2 + x0 ** 2) / x_array ** 2
    beta = 2 * KappaTstar / x_array ** 2
    ln_alpha, ln_beta = np.log(alpha[::-1]), np.log(beta[::-1])
    return np.diagonal(rbs_func(ln_alpha, ln_beta))[::-1]

def _L1_norm(x, y):
    return np.mean(np.abs(x - y))

def _ln_likelihood(theta, x, y, yerr):
    sigma, x0, KappaTstar, f = theta
    model = KT18_model(x, sigma, x0, KappaTstar)
    ln_prob = -.5 * np.sum((y - model / f) ** 2 / yerr ** 2 + 2 * np.log(yerr))
    return ln_prob

def _ln_prior_1st(theta, beam):
    sigma, x0, KappaTstar, f = theta
    sigma_0, sigma_u = beam
    if 0 < x0 < 5 and 0 < KappaTstar < 500 and 1 < f < 20 and sigma > 0:
        return -.5 * ((sigma - sigma_0) ** 2 / sigma_u ** 2 +
                      np.log(sigma_u ** 2 * f ** 2))
    return -np.inf

def _ln_prior_2nd(theta, beam, kt_prior):
    sigma, x0, KappaTstar, f = theta
    sigma_0, sigma_u = beam
    if 0 < x0 < 5 and 0 < KappaTstar < 500 and 1 < f < 20 and sigma > 0:
        return -.5 * ((sigma - sigma_0) ** 2 / sigma_u ** 2 +
                      np.log(sigma_u ** 2 * f ** 2)) + np.log(kt_prior(KappaTstar))
    return -np.inf

def _ln_prob_1st(theta, x, y, yerr, beam):
    p = _ln_prior_1st(theta, beam)
    l = _ln_likelihood(theta, x, y, yerr)
    if not np.isfinite(p) or np.isnan(p) or not np.isfinite(l) or np.isnan(l):
        return -np.inf
    return p + l

def _ln_prob_2nd(theta, x, y, yerr, beam, kt_prior):
    p = _ln_prior_2nd(theta, beam, kt_prior)
    l = _ln_likelihood(theta, x, y, yerr)
    if not np.isfinite(p) or np.isnan(p) or not np.isfinite(l) or np.isnan(l):
        return -np.inf
    return p + l

def fit(x, y, yerr, name, diag, kt_prior=None, plot=False,
        n_walker=config.n_walker, n_step=config.n_step, n_sample=config.n_sample):
    
    beam = get_beam(name)
    n_dim = config.n_dim

    if kt_prior is None:  # The first fitting only freezes kappatstar    
        pos = (np.array([beam[0], 0.07, 0.3, 2]) + np.random.randn(n_walker, n_dim) *
               np.array([1e-4, 0.01, 0.03, 0.2]))
        sampler = emcee.EnsembleSampler(n_walker, n_dim, _ln_prob_1st,
                  args=(x[1:], y[1:], yerr[1:], beam))
        # The first term of y should always be zero and
        # will never be affected by the factor of f.
        sampler.run_mcmc(pos, n_step, progress=True)
        samples = sampler.get_chain()
        flat_samples = samples[-n_sample:, :, :]
        perc_samples = flat_samples.reshape((-1, n_dim))
        par50 = np.percentile(perc_samples, 50, axis=0)
        par16 = np.percentile(perc_samples, 16, axis=0)
        par84 = np.percentile(perc_samples, 84, axis=0)

        model_y = np.append([1.], KT18_model(x[1:], *par50[:3]) / par50[3])
        l1_norm = _L1_norm(model_y, y) * 100

        print("The first MCMC is done!\n")
        print(np.percentile(perc_samples, 50, axis=0))
        print(np.percentile(perc_samples, 16, axis=0))
        print(np.percentile(perc_samples, 84, axis=0))
        return flat_samples[:, :, 2:3], par50[2], par16[2], par84[2], model_y, l1_norm
    # Above is the first-time fitting and aims to get an estimation of kappatstar.
    # It returns only the posterior of kappatstar.
    else: 
        # kt_prior is a tuple of (initial guess, prior function).
        # Prior function (kt_prior[1]) will be the prior in the second-time fitting.
        pos = (np.array([beam[0], 0.07, max(kt_prior[0], 1e-3), 2]) +
               np.random.randn(n_walker, n_dim) * np.array([1e-4, 0.01, 1e-4, 0.2]))
        sampler = emcee.EnsembleSampler(n_walker, n_dim, _ln_prob_2nd,
                  args=(x[1:], y[1:], yerr[1:], beam, kt_prior[1]))

        sampler.run_mcmc(pos, n_step, progress=True)
        samples = sampler.get_chain()
        flat_samples = samples[-n_sample:, :, :]
        perc_samples = flat_samples.reshape((-1, n_dim))
        par50 = np.percentile(perc_samples, 50, axis=0)
        model_y = np.append([1.], KT18_model(x[1:], par50[0], par50[1], kt_prior[0]) / par50[3])

        labels = ["$\\sigma_{\mathrm{beam}}$", "$\\sigma_{\mathrm{inj}}$",
                  "$\\kappa t_*$", "$f$"]
        if plot and diag == 'D16':
            fig, axes = plt.subplots(n_dim, figsize=(10, 7), sharex=True)
            for i in range(n_dim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

            axes[-1].set_xlabel("step number")
            plt.savefig(savefigs_path + name + '_' + diag + '_chains.pdf')
            
            corner.corner(flat_samples.reshape((n_sample * n_walker, n_dim)),
                          labels=labels, range=[.9] * n_dim)
            plt.savefig(savefigs_path + name + '_' + diag + '_corner.pdf')

        print("The second MCMC is done!\n")
        print(np.percentile(perc_samples, 50, axis=0))
        print(np.percentile(perc_samples, 16, axis=0))
        print(np.percentile(perc_samples, 84, axis=0))

        return flat_samples[:, :, :2], np.percentile(perc_samples, 50, axis=0)[:2], model_y


def fit_once(x, y, yerr, name, plot=False,
             n_walker=config.n_walker, n_step=config.n_step, n_sample=config.n_sample):

    beam = get_beam(name)
    n_dim = config.n_dim

    pos = (np.array([beam[0], 0.07, 0.3, 2]) + np.random.randn(n_walker, n_dim) *
           np.array([1e-4, 0.02, 0.08, 0.25]))
    sampler = emcee.EnsembleSampler(n_walker, n_dim, _ln_prob_1st,
              args=(x[1:], y[1:], yerr[1:], beam))
    # The first term of y should always be zero and
    # will never be affected by the factor of f.
    sampler.run_mcmc(pos, n_step, progress=True)
    samples = sampler.get_chain()
    flat_samples = samples[-n_sample:, :, :]
    perc_samples = flat_samples.reshape((-1, n_dim))
    med_par = np.percentile(perc_samples, 50, axis=0)

    print(med_par)





