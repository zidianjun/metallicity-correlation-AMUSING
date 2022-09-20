
import constant
import config
from paths import *
import utils
from blue_noise import gen_blue_noise_band
import diagnostics
from mcmc import fit, KT18_model, gauss_kde_pdf

from astropy.io import fits
import numpy as np
from scipy.spatial import ConvexHull
import time
import matplotlib.pyplot as plt
if server_mode:
    plt.switch_backend('agg')


class Galaxy(object):

    def __init__(self, name, diag='D16', min_SN=config.min_SN, fit_once=False,
                 map_mode=False, binning=True, mc_plot=False, save_mask=False,
                 truncate=(0, 10000, 0, 10000)):
        
        time0 = time.time()
        print("Start analyzing " + name + " with " + diag + " diagnostics.\n")

        self.name, self.diag, self.min_SN = name, diag, min_SN
        x1, x2, y1, y2 = truncate

        self.eline_data = fits.open(fits_path + 'flux_elines.' + name +
                                    '.cube.fits')[0].data[:, y1:y2, x1:x2]
        channel, self.height, self.width = self.eline_data.shape

        self.EW = self.eline_data[110].reshape(-1)
        # self.EBV = fits.open(fits_path + name +
                             # '.SSP.cube.fits')[0].data[11, y1:y2, x1:x2].reshape(-1)
        Ka, Kb = 3.33, 4.60  # Calzetti et al. (2000)
        self.EBV = 2.5 / (Kb - Ka) * np.log10(
            self.line_flux('Halpha') / self.line_flux('Hbeta') / 2.86)

        self.maps = self.build_maps()

        self.X, self.Y = utils.deproject(self.name, self.height, self.width, x1, y1)

        self.b2a, self.distance = utils.get_morph(self.name)[-2:]
        self.kpc_per_arcsec = self.distance * constant.arcsec
        self.kpc_per_pix = self.kpc_per_arcsec * constant.arcsec_per_pix
        self.beam = utils.get_beam(name, kpc_per_arcsec=self.kpc_per_arcsec)[0]
        print("One arcsec is %.3fkpc. Physical beam size is %.3fkpc.\n" %(
              self.kpc_per_arcsec, self.beam))

        if not map_mode: # If only figures are required, skip the correlation function.
            self.mcmc_fit(mc_plot=mc_plot, fit_once=fit_once)
        # Take self.error as a "global" parameter in self.
        self.main(err_mode=False, binning=binning, save_mask=save_mask)

        time1 = time.time()
        print("Initialization time: %.3fs.\n" %(time1 - time0))

    def main(self, err_mode=True, binning=True, save_mask=False, times=config.times):
        '''
        The parameter err_mode is global.
        If err_mode is False, the loop will only run once,
            and in the eline function only central values will be used.
            This records the 2p correlation and its error.
        If err_mode is True, the loop will run for times,
            and in the eline function Gaussian-generated values will be used.
            This records the original metallicity map.
        '''
        self.err_mode = err_mode
        self.binning = binning
        self.mask_list = []
        # The mask changes every time when err_mode is True.

        value = None
        for i in range(max(times * err_mode, 1)):
            met, ion, mask = self.metallicity()

            if not err_mode:
                print("Below is the original map.")
                self.total_number = np.sum(mask)
                if save_mask:
                    np.save(output_path + 'mask/mask_' + self.name +
                            '_binning_' + str(binning) + '.npy', mask)
            print("%d pixels in realization %d of %s." %(np.sum(mask), i, self.name))

            self.mask_list.append(mask)

            self.met, self.ion = met[mask], ion[mask]
            self.x = self.X[mask] * self.kpc_per_pix
            self.y = self.Y[mask] * self.kpc_per_pix

            self.rad = np.sqrt(self.x ** 2 + self.y ** 2)

            if np.sum(mask) < 3:
                (self.met, self.ion, self.x, self.y, self.rad,
                 self.bin_rad, self.bin_met, self.met_fluc,
                 self.dist1, self.ksi1, self.ksi_u1,
                 self.dist2, self.ksi2, self.ksi_u2) = [np.nan] * 14
                self.fill_f_A, self.fill_f_p, self.l1_norm = 0., 0., np.nan
                break

            self.bin_rad, self.bin_met = utils.bin_array(self.rad, self.met,
                                                         bin_size=self.kpc_per_pix)
            self.met_fluc = utils.step(self.rad, self.met, self.bin_rad, self.bin_met)

            if err_mode:  # Errors of 2p correlation are only available using bootstrap.
                dist, ksi = utils.tpcf(self.met_fluc, self.x, self.y,
                                       bin_size=self.kpc_per_pix)
                v = np.expand_dims(ksi, axis=0)
                value = np.concatenate([value, v], axis=0) if value is not None else v
            else:
                area = ConvexHull(self.kpc_per_pix * np.stack([self.X, self.Y],
                                  axis=1)[self.mask_Ha()]).volume
                # area = ConvexHull(np.stack([self.x, self.y], axis=1)).volume
                dA = self.kpc_per_pix ** 2 / utils.b2a_to_cosi(self.b2a)
                self.fill_f_A = len(self.x) * dA / area
                self.fill_f_p = len(self.x) * 1. / np.sum(self.mask_Ha())
        
        if err_mode and value is not None:  # dist is only available when err_mode is True
            mean, std = value.mean(0), value.std(0)
            self.max_sep = config.max_separation * np.median(self.rad)
            valid = ((np.abs(mean) > 1e-4) & (dist < self.max_sep))
            # Remove null and keep it shorter to avoid edge effect.
            if binning:
                self.dist1 = dist[valid]
                self.ksi1 = mean[valid]
                self.ksi_u1 = std[valid]
            else:
                self.dist2 = dist[valid]
                self.ksi2 = mean[valid]
                self.ksi_u2 = std[valid]


    # ========== reconstruction ========== 
    def build_maps(self):
        label = 'SIII9069' if self.diag.startswith('K19') else 'SII6731'
        # If diagnostic is K19, it requires ionization parameter, and thus SIII.
        # If not, the bottleneck is generally SII6731.
        f_map = self.eline_data[config.eline_dict[label]]
        f_map = np.where(f_map > 0., f_map, 0.)
        f_err_map = self.eline_data[config.eline_dict[label] + config.diff]
        f_err_map = np.where(f_err_map > 0., f_err_map, np.mean(f_err_map))
        return utils.multi_order_bin(f_map, f_err_map, self.min_SN)[2]

    def reload_maps(self, signal, noise):
        # self.maps records the way of binning.
        return utils.reconstruct_maps(signal, noise, self.maps)

    # ========== line flux and ratio ========== 
    def line_flux(self, label, error=False):
        mat = self.eline_data[config.eline_dict[label] + config.diff * error]
        if error:
            return np.where(mat > 0., mat, np.mean(mat)).reshape(-1)
        else:
            return np.where(mat > 0., mat, np.nan).reshape(-1)

    def eline(self, label):
        # First generate a new line flux map and then apply the S/N threshold.
        # If err_mode is True, the mask changes every time.
        f_map = self.eline_data[config.eline_dict[label]]
        f_err_map = self.eline_data[config.eline_dict[label] + config.diff]
        f_err_map = np.where(f_err_map > 0., f_err_map, np.mean(f_err_map))
        r_map = np.random.normal(f_map, f_err_map * self.err_mode)
        r_map = np.where(r_map > 0., r_map, 0.)
        
        if self.binning:  # Use the binned map for estimating l_corr
            signal, noise = self.reload_maps(r_map, f_err_map)
        else:  # Use the original map for estimating w_inj
            signal, noise = r_map.copy(), f_err_map.copy()

        res = np.where(signal / noise > self.min_SN, signal, np.nan)
        # print("%s map has %d pixels" %(label, np.sum(~np.isnan(res))))
        
        return res.reshape(-1)

    def ratio(self, label_up_list, label_down_list, static=False):
        label_list = label_up_list + label_down_list
        reddest_wavelength = 0.
        for label in label_list:
            if constant.line_rest_wavelength_dict[label] > reddest_wavelength:
                reddest_wavelength = constant.line_rest_wavelength_dict[label]
        flux_up, flux_down = 0., 0.

        for label in label_up_list:
            f = self.line_flux(label) if static else self.eline(label)
            flux_up += f * diagnostics.dered_f(self.EBV,
                constant.line_rest_wavelength_dict[label], reddest_wavelength)
        for label in label_down_list:
            f = self.line_flux(label) if static else self.eline(label)
            flux_down += f * diagnostics.dered_f(self.EBV,
                 constant.line_rest_wavelength_dict[label], reddest_wavelength)
        return flux_up / flux_down

    # ========== masks ========== 
    def mask_AGN(self, c=config.AGN_criterion):

        N = np.log10(self.ratio(['NII6584'], ['Halpha'], static=True))
        N = np.where(~np.isnan(N), N, 99)

        O = np.log10(self.ratio(['OIII5007'], ['Hbeta'], static=True))
        O = np.where(~np.isnan(O), O, 99)

        if c == 'Kewley':
            return ((O < 0.61 / (N - 0.47) + 1.19) & (N < .4))
        elif c == 'Kauffmann':
            return ((O < 0.61 / (N - 0.05) + 1.30))
        else:
            print("AGN criterion must be either 'Kewley' or 'Kauffmann'!")
            return False

    def mask_EW(self, c=config.EW_criterion):
        EW = np.where(~np.isnan(self.EW), self.EW, 0)
        return EW < c

    def mask_Ha(self):
        return np.nan_to_num(self.line_flux('Halpha') /
               self.line_flux('Halpha', error=True)) > self.min_SN
    
    # ========== metallicity & 2-point correlation ========== 
    def metallicity(self):
        if self.diag in config.available_diag_list:
            if self.diag.startswith('K19'):
                return diagnostics.K19_diag(self)
            else:
                met, mask = getattr(diagnostics, self.diag)(self)
                return met, np.nan * np.ones(len(met)), mask
        else:
            raise ValueError("Unavailable diagnostics!")

    def met_grad(self, re=True):
        # if re is True, Re ** re = Re, otherwise Re ** re = 1
        Re = utils.get_one_value(self.name, 'Re')
        inner = self.bin_rad < Re
        try:
            slope = utils.grad(self.bin_rad[inner], self.bin_met[inner]) * Re ** re
        except:
            slope = np.nan
        return slope

    def mcmc_fit(self, mc_plot=False, fit_once=False):
        self.main(err_mode=True, binning=True)
        self.mask_list1 = self.mask_list
        # Take self.reconstruction as a "global" parameter in self.
        if type(self.dist1) == float:
            print("The data quality of the original map of %s" %(self.name))
            print("is too low, thus no first-stage MCMC fit for correlation length.\n")
            samples = np.zeros((config.n_sample, config.n_walker, 1))
            par = np.zeros((1, ))
        else:
            samples_kt, par_kt, par_kt16, par_kt84, model_y, self.l1_norm = fit(
                self.dist1, self.ksi1, self.ksi_u1, self.name, self.diag)
            # The first fit. This time for all the binned maps with/without PSFs.
            print("The estimation of $\\kappa t_*$ is %.3f " %(par_kt))
            print("(with 68%% range from %.3f to %.3f)" %(par_kt16, par_kt84))
            print("L1 norm is %.1f" %(self.l1_norm))

        if fit_once:
            self.par_kt = par_kt
            self.model_y = model_y
            self.dist = self.dist1
        else:
            self.main(err_mode=True, binning=False)
            if type(self.dist2) == float:  # if self.met is not np.nan
                print("The data quality of the original map of %s" %(self.name))
                print("is too low, thus no second-stage MCMC fit for injection width.\n")
                samples = np.zeros((config.n_sample, config.n_walker, 2))
                par = np.zeros((2, ))
            else:
                prior = (par_kt, gauss_kde_pdf(posterior=samples_kt.reshape(-1)))
                samples, par, model_y = fit(self.dist2, self.ksi2, self.ksi_u2,
                    self.name, self.diag, kt_prior=prior, plot=mc_plot)
                # The second fit. This time only for original maps with measured PSFs.
            self.mask_list2 = self.mask_list
            self.samples = np.concatenate([samples, samples_kt], axis=2)
            self.par = np.concatenate([par, [par_kt]], axis=0)
            self.model_y = model_y
            self.dist = self.dist2



class GalaxyFigure(object):
    def __init__(self, galaxy, savefig_path=None):
        self.cmap = plt.cm.RdYlBu_r
        self.g = galaxy
        '''
        Directly inheriting class Galaxy will double the time
        consumed to compute the 2-point correlation function.
        Note that met_fluc_corr function is only available for
        Galaxy with map_mode=False. Only in this case would
        the two-point correlation be computed.
        '''
        self.savefig_path = savefig_path

    def map(self, ax=None, dtype='met', vrange='auto', FoV_size=250, deproj=True):
        if ax is None:
            plt.figure(figsize=(12, 10))
            ax = plt.axes([.12, .12, .86, .86])
        
        if dtype in ['met', 'met_fluc', 'ion']:
            if dtype == 'ion' and not self.g.diag.startswith('K19'):
                print("Warning: Only K19 diagnostics have an ionization parameter map!")
            map_type = getattr(self.g, dtype)
        else:
            raise ValueError("dtype should only be 'met', 'met_fluc', or 'ion'!")

        vmin = np.percentile(map_type, .1) if vrange == 'auto' else vrange[0]
        vmax = np.percentile(map_type, 99.9) if vrange == 'auto' else vrange[1]
        if dtype == 'met_fluc':
            abs_max = max(-vmin, vmax)
            vmin, vmax = -abs_max, abs_max

        if deproj:
            fig = ax.scatter(self.g.x, self.g.y, c=map_type, alpha=0.75, vmin=vmin,
                    vmax=vmax, s=1, edgecolors='face', marker='o', cmap=self.cmap)
        else:
            maps = utils.inv_where(map_type, self.g.mask_list[0]).reshape(
                (self.g.height, self.g.width))
            fig = utils.display_map(maps, maps, ax,
                min_SN=0., log=False, vrange=(vmin, vmax))

        cbar = plt.colorbar(fig)
        if dtype == 'met':
            cbar.set_label('12 + log(O/H)', size=25)
        elif dtype == 'met_fluc':
            cbar.set_label('$\\Delta$ (O/H)', size=25)
        else:
            cbar.set_label('Ionization parameter', size=25)
        cbar.ax.tick_params(labelsize=25)
        ax.set_xlim(-FoV_size * self.g.kpc_per_pix, FoV_size * self.g.kpc_per_pix)
        ax.set_ylim(-FoV_size * self.g.kpc_per_pix, FoV_size * self.g.kpc_per_pix)
        ax.set_xlabel('x (kpc)', fontsize=25)
        ax.set_ylabel('y (kpc)', fontsize=25)
        ax.tick_params(axis='both', labelsize=25)

        if self.savefig_path is not None:
            plt.savefig(self.savefig_path + self.g.name + '_' +
                        self.g.diag + '_' + dtype + '_map.pdf')

    def met_fluc(self, show_bin=False):
        fig, axes = plt.subplots(2, figsize=(20, 10), sharex=True)
        ax = axes[0]
        ax.scatter(self.g.rad, self.g.met, marker='.', color='gray')
        if show_bin:
            ax.scatter(self.g.bin_rad, self.g.bin_met, marker='o', color='k', s=1)
        ax.tick_params(axis='both', labelsize=15)
        ax.set_ylabel('12 + log(O/H)', fontsize=15)

        ax = axes[1]
        ax.scatter(self.g.rad, self.g.met_fluc, marker='.', color='gray')
        ax.tick_params(axis='both', labelsize=15)
        ax.set_xlabel('Radius (kpc)', fontsize=15)
        ax.set_ylabel('$\\Delta$(O/H)', fontsize=15)
        
        if self.savefig_path is not None:
            plt.savefig(self.savefig_path + self.g.name + '_' +
                        self.g.diag + '_met_fluc.pdf')

    def met_fluc_corr(self, ax=None, stage=3):
        if ax is None:
            plt.subplots(figsize=(8, 4))
            plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.85)
            ax = plt.subplot(111)
        
        # If stage == 1, plot ksi1; if stage == 2, plot ksi2; If stage == 3, plot both.
        if stage % 2 == 1:
            dist, ksi, ksi_u = gen_blue_noise_band(self.g.mask_list1, self.g.kpc_per_pix,
                self.g.beam, maps=self.g.maps, height=self.g.height, width=self.g.width)
            ax.errorbar(self.g.dist1, self.g.ksi1, yerr=self.g.ksi_u1,
                        linestyle='none', marker='o', ms=8, color='black', alpha=.8,
                        label='Two-point correlation (binned maps)')
        if stage > 1:
            # When plotting both stage 1 & 2, blue noise is defaulted to show stage 2.
            # The blue noises of stage 1 & 2 are slightly different.
            dist, ksi, ksi_u = gen_blue_noise_band(self.g.mask_list2, self.g.kpc_per_pix,
                self.g.beam, maps=None, height=self.g.height, width=self.g.width)
            ax.errorbar(self.g.dist2, self.g.ksi2, yerr=self.g.ksi_u2,
                        linestyle='none', marker='o', ms=8, color='gray', alpha=.8,
                        label='Two-point correlation (unbinned maps)')

        ax.fill_between(dist, ksi - ksi_u, ksi + ksi_u,
                        color='gray', alpha=.3, edgecolors='none')

        ax.plot(self.g.dist, self.g.model_y, color='black', linestyle='--')

        ax.set_xlim(-.05, 3.1)
        ax.set_ylim(-.1, 1.1)
        ax.set_xlabel("$r$ (kpc)", fontsize=25)
        ax.set_ylabel("$\\xi$($r$)", fontsize=25)
        ax.tick_params(axis='both', labelsize=25)
        if self.g.name == 'ASASSN-18oa':
            ax.annotate('ESO 476-G 016', xy=(1.8, .4), xytext=(2.2, .6), fontsize=20)
        else:
            ax.annotate('NGC 0613', xy=(1.8, .4), xytext=(2.2, .6), fontsize=20)
        ax.legend(loc='upper right', fontsize=20)
        
        if self.savefig_path is not None:
            plt.savefig(self.savefig_path + self.g.name + '_' +
                        self.g.diag + '_2p_corr_stage' + str(stage) + '.pdf')
        









