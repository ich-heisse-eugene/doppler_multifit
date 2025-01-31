#!/usr/local/bin/python3.11

from sys import argv,exit
import argparse
import numpy as np
import numpy.ma as ma
from lmfit import minimize, Parameters, fit_report, report_fit
from scipy.interpolate import splrep, splev
from scipy import signal
from numpy.polynomial.legendre import legfit, legval
from numpy.polynomial.chebyshev import chebfit, chebval
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

# block of constants and initialized variables
oversample = 3 # oversampling of the profile
defresol = 17.6 # spectral resolution in km/s [Default is 17.6 km/s => R = 17000, see desc.]
eps = 0.6     # Limb darkening [Default is 0.6]
Ncomp = 1     # Number of components [Default is 1]
minRV = -500  # Minimum RV [Default is -500]
maxRV = 500   # Maximum RV [Default is 500]
maxV = 450    # Maximum vsin i [Default is 450]
c = 299792.458 # speed of light in km/s

fontsize = 12

mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['text.usetex'] = True

def import_data(input_file):
    block = np.loadtxt(input_file, unpack=True, dtype=float)
    if block.shape[0] >= 3:
        vel, proff, err = block[0,:], block[1, :], block[2, :]
    else:   # set fixed errors
        vel, proff = block[0,:], block[1, :]
        err = 0.002 * np.ones(len(vel))
    return vel, proff, err

def fill_param(args, conf, ncomp):
    for item in vars(args):
        attr = getattr(args, item)
        if isinstance(attr, str) and attr is not None:
            val = attr.split(',')
            if len(val) > 1 and len(val) % 2 == 0 and item.find('init') != -1:
                for x in range(0, len(val), 2):
                    if int(val[x]) > ncomp:
                        print(f"Wrong component number {val[x]} in args.{item}")
                    else:
                        conf[int(val[x])-1][item] = float(val[x+1])
            elif len(val) > 1 and len(val) % 2 == 0 and item.find('min') != -1:
                for x in range(0, len(val), 2):
                    if int(val[x]) > ncomp:
                        print(f"Wrong component number {val[x]} in args.{item}")
                    else:
                        conf[int(val[x])-1][item] = float(val[x+1])
            elif len(val) > 1 and len(val) % 2 == 0 and item.find('max') != -1:
                for x in range(0, len(val), 2):
                    if int(val[x]) > ncomp:
                        print(f"Wrong component number {val[x]} in args.{item}")
                    else:
                        conf[int(val[x])-1][item] = float(val[x+1])
            elif item.find('fix') != -1:
                for x in range(0, len(val)):
                    if not val[x].isnumeric() or int(val[x]) > ncomp:
                        print(f"Wrong component number {val[x]} in args.{item}")
                    else:
                        conf[int(val[x])-1][item.replace('fix','var')] = False
    out_params = Parameters()
    for i in range(ncomp):
        out_params.add('radvel'+str(i+1), value=conf[i]['initRV'], min=conf[i]['minRV'], \
                   max=conf[i]['maxRV'], vary=conf[i]['varRV'])
        out_params.add('vsini'+str(i+1), value=conf[i]['initV'], min=conf[i]['minV'], \
                   max=conf[i]['maxV'], vary=conf[i]['varV'])
        out_params.add('amp'+str(i+1), value=0.001, vary=True)
    return out_params

def interp(vel_ref, vel_new, flux_new):
    tck = splrep(vel_new, flux_new)
    return splev(vel_ref, tck)

def doppler(vel_grid, dv, vsini, eps):
    e1 = 2. * (1. - eps)
    e2 = np.pi * eps / 2.
    e3 = np.pi * (1. - eps/3.)
    npts =  int(np.floor(2. * vsini / dv))
    if npts % 2 == 0:
        npts = npts + 1
    nwid = npts / 2
    x = (np.arange(npts) - nwid) * dv / vsini
    vel_init = x * vsini
    x1 = np.abs(1. - x**2)
    broad_init = (e1 * np.sqrt(x1) + e2 * x1) / e3
    vel_lim = np.abs(vel_grid[0]) if np.abs(vel_grid[0]) > np.abs(vel_grid[-1]) else np.abs(vel_grid[-1])
    vel_left = np.linspace(-vel_lim, vel_init[0]-dv, int(np.abs((-1)*vel_lim - (vel_init[0]+dv))/dv))
    vel_right = np.linspace(vel_init[-1]+dv, vel_lim, int(np.abs(vel_lim - (vel_init[-1]+dv))/dv))
    vel_grid = np.hstack((vel_left, vel_init, vel_right))
    broad = np.hstack((np.zeros(len(vel_left)), broad_init, np.zeros(len(vel_right))))
    broad = broad/np.max(broad)
    vel_tmp = np.linspace(vel_grid[0], vel_grid[-1], oversample*len(vel_grid))
    broad_tmp = interp(vel_tmp, vel_grid, broad)
    return vel_tmp, broad_tmp

def gauss(vel, fwhm):
    gs = np.exp(-(4.*np.log(2.)*(vel)**2.)/(fwhm**2.))
    return gs/np.max(gs)

def prof(vel, amp, vr, dvel, vsini, eps, fwhm):
    vel_dop, dop = doppler(vel, dvel, vsini, eps)
    gs = gauss(vel_dop, fwhm)
    conv = np.convolve(gs/np.max(gs), dop/np.max(dop), mode='same')
    prof = interp(vel, vel_dop+vr, conv)
    return amp * prof

def residual(params, vel_obs, obs_prof, obs_err, dv, eps, fwhm):
    comb_prof = np.zeros(len(vel_obs))
    for i in range(len(params)//3):
        rv = params['radvel'+str(i+1)].value
        a = params['amp'+str(i+1)].value
        vsini = params['vsini'+str(i+1)].value
        comb_prof += prof(vel_obs, a, rv, dv, vsini, eps, fwhm)
    return (obs_prof - interp(vel_obs, vel_obs, comb_prof)) / obs_err**2

def make_prof(params, vel_obs, dv, eps, fwhm):
    comb_prof = np.zeros(len(vel_obs))
    for i in range(len(params.params)//3):
        rv = params.params['radvel'+str(i+1)].value
        a = params.params['amp'+str(i+1)].value
        vsini = params.params['vsini'+str(i+1)].value
        comb_prof += prof(vel_obs, a, rv, dv, vsini, eps, fwhm)
    return comb_prof

def report_result(params, args, logfile):
    print("Result\n===============")
    with open(logfile, 'a') as fp:
        for i in range(len(params.params)//3):
            # print report
            if params.redchi is not None:
                scale = 1 # params.redchi / (params.ndata - params.nvarys)
#                print(f"Warning: Errors will be multiplied by a scale factor {scale:.4f}")
            if params.params['radvel'+str(i+1)].stderr is not None \
                         and params.params['vsini'+str(i+1)].stderr is not None:
                print(f"RV{i+1} = {params.params['radvel'+str(i+1)].value:.2f} ± {params.params['radvel'+str(i+1)].stderr * scale:.2f} km/s")
                print(f"v{i+1}sin i = {params.params['vsini'+str(i+1)].value:.2f} ± {params.params['vsini'+str(i+1)].stderr * scale:.2f} km/s")
                print(f"{params.params['radvel'+str(i+1)].value:.2f} ± {params.params['radvel'+str(i+1)].stderr * scale:.2f} km/s\t", end='', file=fp)
                print(f"{params.params['vsini'+str(i+1)].value:.2f} ± {params.params['vsini'+str(i+1)].stderr * scale:.2f} km/s\t", end='', file=fp)
            else:
                print(f"RV = {params.params['radvel'+str(i+1)].value:.2f} ± None km/s")
                print(f"vsin i = {params.params['vsini'+str(i+1)].value:.2f} ± None km/s")
                print(f"{params.params['radvel'+str(i+1)].value:.2f} ± None km/s\t", end='', file=fp)
                print(f"{params.params['vsini'+str(i+1)].value:.2f} ± None km/s\t", end='', file=fp)
        if params.redchi is not None:
            print(f"\n-------\nReduced chisq = {params.redchi:.5f}")
            if not args.batch:
                print(f"\n-------\nReduced chisq = {params.redchi:.5f}\n", file=fp)
        else:
            print(f"The result is uncertain")
            if not batch_key:
                print(f"The result is uncertain\n", file=fp)
        print("====== Full report =======")
        report_fit(params)
        if not args.batch:
            print("====== Full report =======\n", file=fp)
            print(fit_report(params), file=fp)
    fp.close()
    return None

def plot_profile(obs_vel, obs_prof, obs_err, mask, diff, fit_obs, plot_file, batch_key):
    fig = plt.figure(figsize=(6,5), tight_layout=True)
    if len(fit_obs) == 0:
        ax = fig.add_subplot(1,1,1)
        ax.plot(obs_vel, obs_prof, 'ks', ms=1.)
        plt.errorbar(obs_vel, obs_prof, obs_err, capthick=0, capsize=0, \
                     ecolor='k', elinewidth=0.4, xerr=None, fmt="none")
        ax.set_xlabel(r"Velocity [km\,s$^{-1}$]")
        ax.set_ylabel("Mean LSD $I$")
        plt.show(block=False)
    else:
        grid = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[4, 1])
        ax = fig.add_subplot(grid[0])
        # plt.errorbar(obs_vel, obs_prof, obs_err, capthick=0, capsize=0, \
        #              ecolor='k', elinewidth=0.4, xerr=None, fmt="none")
        ax.plot(obs_vel, obs_prof, 'ks', ms=1.)
        ax.plot(obs_vel[np.logical_not(mask)], obs_prof[np.logical_not(mask)], 'bx', ms=3.1)
        ax.plot(obs_vel, fit_obs, 'r-', lw=1.1)
        ax.set_ylabel("Mean LSD")
        ax.set_xlim(obs_vel[0], obs_vel[-1])
        ax_1 = fig.add_subplot(grid[1])
        ax_1.plot(obs_vel[mask], diff, 'ks', ms=0.7)
        ax_1.set_ylabel("Resid.")
        ax_1.set_xlabel(r"Velocity [km\,s$^{-1}$]")
        lim = np.max(np.abs(diff))*1.1
        ax_1.set_xlim(obs_vel[0], obs_vel[-1])
        ax_1.set_ylim(-lim, lim)
        plt.savefig(plot_file, dpi=300)
        if not batch_key:
            plt.show()
    return None

def process_file(infile, args, logfile, bary):
    global c, minV, maxV, minRV, maxRV
    if args.resol > 1000:
        resol = c / args.resol
    elif args.resol > 0 and args.resol < 1:
        resol = 2.5 * args.resol * c / 5500. # reference wavelength = 5500A
    else:
        resol = args.resol
    obs_vel, obs_prof, obs_err = import_data(infile)
    if args.renormalize:
        obs_prof = renormalize_iraf(obs_vel, obs_prof, "chebyshev", 1, 7, 1.2, 6., 5)
    obs_vel += bary
    if args.plot == None:
        plotfile = infile + "_fit.pdf"
    else:
        plotfile = args.plot
    if maxV >= np.max(np.abs(obs_vel)): maxV = np.max(np.abs(obs_vel)) - resol
    if args.zoomRV is not None:
        rv_lim = args.zoomRV.split(',')
        idx = np.where((obs_vel >= float(rv_lim[0])) & (obs_vel <= float(rv_lim[1])))[0]
        obs_vel = obs_vel[idx]
        obs_prof = obs_prof[idx]
        obs_err = obs_err[idx]
    obs_vel_full = obs_vel.copy()
    obs_prof_full = obs_prof.copy()
    obs_err_full = obs_err.copy()
    obs_prof = 1. - obs_prof
    minRV = obs_vel[0]
    maxRV = obs_vel[-1]
    dv = np.abs(obs_vel[-1] - obs_vel[-2])
    conf = []
    for cmp in range(args.ncomp):
        conf_single = {
            "number": cmp+1,
            "initRV": 0,
            "initV": resol,
            "minRV": minRV,
            "maxRV": maxRV,
            "minV": 0.5 * resol,
            "maxV": maxV,
            "varRV": True,
            "varV": True,
        }
        conf.append(conf_single)
    # masking data
    mask = np.ones(len(obs_vel), dtype='bool')
    if args.exclude != "":
        regs = args.exclude.split(',')
        for i in range(len(regs)):
            if regs[i].find(':') == -1:
                print(f"Wrong format of the region {regs[i]}. Skipping...")
            else:
                rvi1,rvi2 = regs[i].split(':')
                idx = np.where((obs_vel >= float(rvi1)) & (obs_vel <= float(rvi2)))
                mask[idx] = False
        obs_vel = obs_vel[mask]
        obs_prof = obs_prof[mask]
        obs_err = obs_err[mask]
    # Minimization
    params = fill_param(args, conf, args.ncomp)
    try:
        out = minimize(residual, params, args=(obs_vel, obs_prof, obs_err, dv, args.eps, resol), nan_policy='omit')
    except Exception as e:
        print(f"Failed: {e}")
        return False
    else:
        model_fit = make_prof(out, obs_vel, dv, args.eps, resol)
        diff = obs_prof - model_fit
        model_full = 1.-make_prof(out, obs_vel_full, dv, args.eps, resol)
        report_result(out, args, logfile)
        plot_profile(obs_vel_full, obs_prof_full, obs_err_full, mask, diff, model_full, plotfile, args.batch)
    return True

# Mean LSD profile re-normalization block
def filter_result(input_signal, window):
	""" Function smooths input data using Savitsky-Golay filter
	"""
	return signal.savgol_filter(input_signal, window_length=window, polyorder=2, axis=0, mode='nearest')
	# return input_signal

def median_smooth(input_signal, npix):
	""" This function performs smoothing of the input array using median filter
	"""
	return signal.medfilt(input_signal, npix)

def fit_poly(w, r, type, order):
    if type == "legendre":
        coef = legfit(w, r, order)
        return coef
    elif type == "chebyshev":
        coef = chebfit(w, r, order)
        return coef

def fit_cont(w, type, coef):
    if type == "legendre":
        return legval(w, coef)
    elif type == "chebyshev":
        return chebval(w, coef)

def reject_points(w, r, cont, low_rej, high_rej, func, ord):
    resid = r - cont
    stdr = np.std(resid)
    idx = np.where((resid >= -low_rej * stdr) & (resid <= high_rej * stdr))
    coef = fit_poly(w[idx], r[idx], func, ord)
    return w[idx], r[idx], cont, coef

def renormalize_iraf(w_init, r_init, fit_func, fit_ord, fit_niter, fit_low_rej, fit_high_rej, window):
	r_0 = r_init.copy()
	r_init = filter_result(r_0, window)
	cont_lev = np.zeros(len(r_init))
	if fit_niter <= 1:
		fit_niter = fit_niter + 1
	w_tmp = w_init
	r_tmp = r_init
	for j in range(fit_niter-1):
		coef = fit_poly(w_tmp, r_tmp, fit_func, fit_ord)
		cont = fit_cont(w_tmp, fit_func, coef)
		w_tmp, r_tmp, cont, coef = reject_points(w_tmp, r_tmp, cont, fit_low_rej, fit_high_rej, fit_func, fit_ord)
		cont_full = fit_cont(w_init, fit_func, coef)
	cont_lev = cont_full
	idx_wrong = np.where(cont_lev == 0.)[0]
	cont_lev[idx_wrong] = r_init[idx_wrong]
	plt.plot(w_init, r_init, 'k-')
	plt.plot(w_init, cont_lev, 'r--')
	plt.show()
	return r_0/cont_lev
# end of re-normalization


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file with mean LSD profile", type=str, default="")
    parser.add_argument("--eps", help="Limb-darkening coefficient", type=float, default=eps)
    parser.add_argument("--resol", help="Spectral resolution. Values > 1000 correspond to R, \
                                     values < 1 correspond to dl in A/pix, otherwise it is in km/s", \
                                     type=float, default=defresol)
    parser.add_argument("--ncomp", help="Number of spectroscopic components in LSD profile", type=int, \
                                        default=Ncomp)
    parser.add_argument("--initRV", help="Initial value(s) of RV for selected components. Format: \
                                         n1,val1,n2,val2,...", type=str)
    parser.add_argument("--minRV", help="Minimal value(s) of RV for selected components. Format: \
                                         n1,val1,n2,val2,...", type=str)
    parser.add_argument("--maxRV", help="Maximal value(s) of RV for selected components. Format: \
                                         n1,val1,n2,val2,...", type=str)
    parser.add_argument("--initV", help="Initial value(s) of vsin i for selected components. Format: \
                                         n1,val1,n2,val2,...", type=str)
    parser.add_argument("--minV", help="Minimal value(s) of vsin i for selected components. Format: \
                                         n1,val1,n2,val2,...", type=str)
    parser.add_argument("--maxV", help="Maximal value(s) of vsin i for selected components. Format: \
                                         n1,val1,n2,val2,...", type=str)
    parser.add_argument("--fixRV", help="Fixed value(s) of RV for selected components. Format: \
                                         n1,n2,...", type=str)
    parser.add_argument("--fixV", help="Fixed value(s) of vsin i for selected components. Format: \
                                         n1,n2,...", type=str)
    parser.add_argument("--zoomRV", help="Range of velocities for fitting. Format: RV1,RV2", type=str)
    parser.add_argument("--plot", help="Save plot to the output file [with extension]", type=str, default=None)
    parser.add_argument("--exclude", help="Regions to exclude in fitting. Format: RV01:RV02,RV11:RV12", \
                        default="", type=str)
    parser.add_argument("--renormalize", help="Renormalizer profile before fitting?", action="store_true")
    parser.add_argument("--batch", help="Batch processing. An input file must contain a list of LSD profiles", action="store_true")
    parser.add_argument("--hastime", help="Batch processin. An input file must contain a column with time marks separated by semicolon from a list of LSD profiles", action="store_true")
    parser.add_argument("--bary", help="An input file contain a column with barycentric correction in km/s", action="store_true")
    args = parser.parse_args()
    infile = args.input
    logfile = infile + "_fit.log"
    # Let's go
    with open(logfile, 'w') as fp:
        print("# Results of fitting\n# ", file=fp, end='')
        for i in range(args.ncomp):
            print(f"RV{i+1} ± sigma\t v{i+1}sin i ± sigma \t", end='', file=fp)
        print("\n#  ---  ", file=fp)
    fp.close()

    if args.batch:
        if args.hastime:
            if args.bary:
                tmark, filelist, barycor = np.loadtxt(infile, unpack=True, usecols=(0,1,2), delimiter=';', dtype=str)
            else:
                tmark, filelist = np.loadtxt(infile, unpack=True, usecols=(0,1), delimiter=';', dtype=str)
        else:
            if args.bary:
                filelist, barycor = np.loadtxt(infile, unpack=True, dtype=str, usecols=(0,1), delimiter=';')
            else:
                filelist = np.loadtxt(infile, unpack=True, dtype=str)
        if not args.bary:
            barycor = np.zeros(len(filelist), dtype=float)
        else:
            barycor = np.asarray(barycor, dtype=float)
        for f in range(len(filelist)):
            result = process_file(filelist[f].strip(), args, logfile, barycor[f])
            if args.hastime:
                with open(logfile, 'a') as fp:
                    print(f"# Filename - time (barycor): {filelist[f].strip()} - {tmark[f].strip()} ({barycor[f]:.2f})", file=fp)
                fp.close()
            else:
                with open(logfile, 'a') as fp:
                    print(f"# Filename: {filelist[f].strip()}", file=fp)
                fp.close()
            if result:
                print(f"File {filelist[f]} has been succesfully processed\n")
    else:
        result = process_file(infile, args, logfile, 0.0)
        if result:
            print(f"File {infile} has been succesfully processed\n")
    exit(0)
