#!/usr/bin/env python3

from sys import argv,exit
import argparse
import numpy as np
from lmfit import minimize, Parameters, report_fit
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

# block of constants and initialized variables
c = 299792.458 # speed of light in km/s
oversample = 3 # oversampling of the profile
resol = 17.6 # spectral resolution in km/s [Default is 17.6 km/s => R = 17000, see desc.]
eps = 0.6     # Limb darkening [Default is 0.6]
Ncomp = 1     # Number of components [Default is 1]
minRV = -500  # Minimum RV [Default is -500]
maxRV = 500   # Maximum RV [Default is 500]
maxV = 450    # Maximum vsin i [Default is 450]

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
            if item.find('fix') != -1:
                for x in range(0, len(val)):
                    if not val[x].isnumeric() or int(val[x]) > ncomp:
                        print(f"Wrong component number {val[x]} in args.{item}")
                    else:
                        conf[int(val[x])-1][item.replace('fix','var')] = False
            elif len(val) > 1 and len(val) % 2 == 0 and item.find('fit') == -1:
                for x in range(0, len(val), 2):
                    if int(val[x]) > ncomp:
                        print(f"Wrong component number {val[x]} in args.{item}")
                    else:
                        conf[int(val[x])-1][item] = float(val[x+1])
    out_params = Parameters()
    for i in range(ncomp):
        out_params.add('radvel'+str(i+1), value=conf[i]['initRV'], min=conf[i]['minRV'], \
                   max=conf[i]['maxRV'], vary=conf[i]['varRV'])
        out_params.add('vsini'+str(i+1), value=conf[i]['initV'], min=conf[i]['minV'], \
                   max=conf[i]['maxV'], vary=conf[i]['varV'])
        out_params.add('amp'+str(i+1), value=0.01, vary=True)
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
    prof = interp(vel, vel_dop+vr, conv/np.max(conv))
    return amp * prof

def residual(params, vel_obs, obs_prof, obs_err, dv, eps, fwhm):
    comb_prof = np.zeros(len(vel_obs))
    for i in range(len(params)//3):
        rv = params['radvel'+str(i+1)]
        a = params['amp'+str(i+1)]
        vsini = params['vsini'+str(i+1)]
        comb_prof += prof(vel_obs, a, rv, dv, vsini, eps, fwhm)
    return (obs_prof - interp(vel_obs, vel_obs, comb_prof)) / obs_err**2

def report_prof(params, vel_obs, obs_prof, obs_err, dv, eps, fwhm, file_rep):
    comb_prof = np.zeros(len(vel_obs))
    print("Result\n===============")
    with open(file_rep, 'w') as fp:
        for i in range(len(params.params)//3):
            rv = params.params['radvel'+str(i+1)].value
            a = params.params['amp'+str(i+1)].value
            vsini = params.params['vsini'+str(i+1)].value
            comb_prof += prof(vel_obs, a, rv, dv, vsini, eps, fwhm)
            # print report
            fp.write(f"Component #{i+1}:\n")
            print(f"Component #{i+1}:")
            if params.params['radvel'+str(i+1)].stderr is not None \
                             and params.params['vsini'+str(i+1)].stderr is not None:
                print(f"RV = {params.params['radvel'+str(i+1)].value:.1f} ± {params.params['radvel'+str(i+1)].stderr:.1f} km/s")
                fp.write(f"RV = {params.params['radvel'+str(i+1)].value:.1f} ± {params.params['radvel'+str(i+1)].stderr:.1f} km/s\n")
                print(f"vsin i = {params.params['vsini'+str(i+1)].value:.1f} ± {params.params['vsini'+str(i+1)].stderr:.1f} km/s")
                fp.write(f"vsin i = {params.params['vsini'+str(i+1)].value:.1f} ± {params.params['vsini'+str(i+1)].stderr:.1f} km/s\n")
            else:
                print(f"RV = {params.params['radvel'+str(i+1)].value:.1f} ± None km/s")
                fp.write(f"RV = {params.params['radvel'+str(i+1)].value:.1f} ± None km/s\n")
                print(f"vsin i = {params.params['vsini'+str(i+1)].value:.0f} ± None km/s")
                fp.write(f"vsin i = {params.params['vsini'+str(i+1)].value:.0f} ± None km/s\n")
        if params.redchi is not None:
            print(f"-------\nReduced chisq = {params.redchi:.5f}")
            fp.write(f"-------\nReduced chisq = {params.redchi:.5f}\n")
        else:
            print("The result is uncertain")
            fp.write("The result is uncertain\n")
        print("====== Full report =======")
        report_fit(params)
        fp.close()
    return comb_prof

def plot_profile(obs_vel, obs_prof, obs_err, obs_fit, file_out):
    fig = plt.figure(figsize=(6,5), tight_layout=True)
    if len(obs_fit) == 0:
        ax = fig.add_subplot(1,1,1)
        ax.plot(obs_vel, obs_prof, 'k-', lw=0.7)
        plt.errorbar(obs_vel, obs_prof, obs_err, capthick=0, capsize=0, \
                     ecolor='k', elinewidth=0.4, xerr=None, fmt="none")
        ax.set_xlabel(r"Velocity, km\,s$^{-1}$")
        ax.set_ylabel("Mean LSD")
        plt.show(block=False)
    else:
        grid = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[4, 1])
        ax = fig.add_subplot(grid[0])
        # plt.errorbar(obs_vel, obs_prof, obs_err, capthick=0, capsize=0, \
        #              ecolor='k', elinewidth=0.4, xerr=None, fmt="none")
        ax.plot(obs_vel, obs_prof, 'k-', lw=0.7)
        ax.plot(obs_vel, obs_fit, 'r-', lw=1.1)
        ax.set_ylabel("Mean LSD")
        ax_1 = fig.add_subplot(grid[1])
        ax_1.plot(obs_vel, obs_prof-obs_fit, 'k-', lw=0.7)
        ax_1.set_ylabel("Resid")
        ax_1.set_xlabel(r"Velocity, km\,s$^{-1}$")
        lim = np.max(np.abs(obs_prof-obs_fit))*1.1
        ax_1.set_ylim(-lim, lim)
        plt.savefig(file_out, dpi=300)
        plt.show()
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file with mean LSD profile", type=str, default="")
    parser.add_argument("--eps", help="Limb-darkening coefficient", type=float, default=eps)
    parser.add_argument("--resol", help="Spectral resolution. Values > 1000 correspond to R, \
                                     values < 1 correspond to dl in A/pix, otherwise it is in km/s", \
                                     type=float, default=resol)
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
    parser.add_argument("--fitRV", help="Range of velocities for fitting. Format: RV1,RV2", type=str)
    parser.add_argument("--interac", help="Enter initial values of radial velocities interactively", \
                                     action="store_true")
    args = parser.parse_args()
    # Let's go
    Ncomp = args.ncomp
    if args.resol > 1000:
        resol = c / args.resol
    elif args.resol > 0 and args.resol < 1:
        resol = 2.5 * args.resol * c / 5500. # reference wavelength = 5500A
    # Default configuration is a single star
    file_plot = args.input+'fit.pdf'
    file_rep = args.input+'fit.report'
    obs_vel, obs_prof, obs_err = import_data(args.input)
    obs_prof = 1 - obs_prof
    if args.fitRV is not None:
        rv_lim = args.fitRV.split(',')
        idx = np.where((obs_vel >= float(rv_lim[0])) & (obs_vel <= float(rv_lim[1])))[0]
        obs_vel = obs_vel[idx]
        obs_prof = obs_prof[idx]
        obs_err = obs_err[idx]
    minRV = obs_vel[0]
    maxRV = obs_vel[-1]
    maxV = maxRV*0.9
    dv = np.abs(obs_vel[-1] - obs_vel[-2])
    conf = []
    for c in range(Ncomp):
        conf_single = {
            "number": c+1,
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
    if args.interac:
        plot_profile(obs_vel, obs_prof, obs_err, [])
        for i in range(Ncomp):
            guess = input(f"Enter expected RV of the component #{i+1} -> ")
            conf[i]['initRV'] = float(guess)
        plt.close()
    # Minimization
    params = fill_param(args, conf, Ncomp)
    try:
        out = minimize(residual, params, args=(obs_vel, obs_prof, obs_err, dv, args.eps, resol))
    except Exception:
        print("Failed")
        exit(1)
    else:
        model_fit = report_prof(out, obs_vel, obs_prof, obs_err, dv, args.eps, resol, file_rep)
        vel = np.linspace(obs_vel[0], obs_vel[-1], len(obs_vel)*oversample)
        plot_profile(vel, 1.-interp(vel, obs_vel, obs_prof), obs_err, 1.-interp(vel, obs_vel, model_fit), file_plot)

    exit(0)
