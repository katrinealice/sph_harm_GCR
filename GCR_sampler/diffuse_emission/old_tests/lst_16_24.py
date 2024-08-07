import os

import numpy as np
import scipy as sp

# Plotting
#import matplotlib.pyplot as plt
#import cmocean
#import cmocean.cm as cmo
#import seaborn as sns
#import pylab as plt
#import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import ticker

# Mapping
import healpy as hp

# GSM  (NOTE: GSM is deprecated, they update GDSM now: https://github.com/telegraphic/pygdsm)
from pygdsm import GlobalSkyModel2016
from pygdsm import GlobalSkyModel

# Wigner D matrices
import spherical, quaternionic

# Simulation
import pyuvsim

# Hydra
import sys
sys.path.append("/cosma8/data/dp270/dc-glas1/Hydra") # change this to your own path
import hydra
from hydra.utils import build_hex_array

import argparse

# Linear solver 
from scipy.sparse.linalg import cg, LinearOperator

# All things astropy
from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.coordinates.builtin_frames import AltAz, ICRS
from astropy.time import Time

# Pandas, dataframe
import pandas as pd

# Time
from numba import jit
import time

# Multiprocessing
from multiprocessing import Pool

# Printing full arrays
np.set_printoptions(threshold=sys.maxsize)

# Construct the argument parser
AP = argparse.ArgumentParser()
AP.add_argument("-dir", "--directory", required=False,
   help="output directory")
ARGS = vars(AP.parse_args())

## Functions
def vis_proj_operator_no_rot(freqs, lsts, beams, ant_pos, lmax, nside, latitude=-0.5361913261514378, include_autos=False, autos_only=False):
    """
    Precompute the real and imaginary blocks of the visibility response 
    operator. This should only be done once and then "apply_vis_response()"
    is used to get the actual visibilities.
    
    Parameters
    ----------
   
    * freqs (array_like):
            Frequencies, in MHz.
    
    * lsts (array):
            lsts (times) for the simulation. In radians.
    
    * beams (list of pyuvbeam):
            List of pyuveam objects, one for each antenna
            
    * ant_pos (dict):
            Dictionary of antenna positions, [x, y, z], in m. The keys should
            be the numerical antenna IDs.    
            
    * lmax (int):
            Maximum ell value. Determines the number of modes used.
             
    * nside (int):
            Healpix nside to use for the calculation (longer baselines should 
            use higher nside).
    * latitude (optional) (float):
            Latitude in decimal format of the simulated array/visibilities. 
            Default: -30.7215 * np.pi / 180 = -0.5361913261514378 (HERA)
            
    * include_autos (optional) (Boolean):
            If True, the auto baselines are included. Default: False. 
    
    Returns
    -------
    
    * vis_response_2D (array_like):
            Visibility operator (δV_ij) for each (l,m) mode, frequency, 
            baseline and lst. Shape (Nvis,N_alms) where Nvis is N_bl x N_times x N_freqs.
            
    * ell (optional: set lm_index = True) (array of int):
            Array of ell-values for the visiblity simulation
            
    * m (optional: set lm_index = True) (array of int):
        Array of ell-values for the visiblity simulation
    
    """
       
    ell, m, vis_alm = hydra.vis_simulator.simulate_vis_per_alm(lmax=lmax, 
                                                               nside=nside, 
                                                               ants=ant_pos, 
                                                               freqs=freqs, 
                                                               lsts=lsts, 
                                                               beams=beams,
                                                               latitude=latitude)
    
    # Removing visibility responses corresponding to the m=0 imaginary parts 
    vis_alm = np.concatenate((vis_alm[:,:,:,:,:len(ell)],vis_alm[:,:,:,:,len(ell)+(lmax+1):]), axis=4)
    
    ants = list(ant_pos.keys())
    antpairs = []
    if autos_only == False and include_autos == False:
        auto_ants = []
    for i in ants:
        for j in ants:
            # Toggle via keyword argument if you want to keep the auto baselines/only have autos
            if include_autos == True:
                if j >= i:
                    antpairs.append((ants[i],ants[j]))
            elif autos_only == True:
                if j == i:
                    antpairs.append((ants[i],ants[j]))
            else:
                if j == i:
                    auto_ants.append((ants[i],ants[j]))
                if j > i:
                    antpairs.append((ants[i],ants[j]))
                
    vis_response = np.zeros((len(antpairs),len(freqs),len(lsts),2*len(ell)-(lmax+1)), dtype=np.complex128)
    # vis_response = np.zeros((len(antpairs),*vis_alm.shape[:-3],2*len(ell)-lmax), dtype=np.complex128)
    
    ## Collapse the two antenna dimensions into one baseline dimension
    # Nfreqs, Ntimes, Nant1, Nant2, Nalms --> Nbl, Nfreqs, Ntimes, Nalms 
    for i, bl in enumerate(antpairs):
        idx1 = ants.index(bl[0])
        idx2 = ants.index(bl[1])
        vis_response[i, :] = vis_alm[:, :, idx1, idx2, :]  
        
    ## Reshape to 2D                                      ## TODO: Make this into a "pack" and "unpack" function
    # Nbl, Nfreqs, Ntimes, Nalms --> Nvis, Nalms
    Nvis = len(antpairs) * len(freqs) * len(lsts)
    # Nvis = np.prod([len(antpairs),*vis_alm.shape[:-3]])
    vis_response_2D = vis_response.reshape(Nvis, 2*len(ell)-(lmax+1))
    
    
    
    if autos_only == False and include_autos == False:
        autos = np.zeros((len(auto_ants),len(freqs),len(lsts),2*len(ell)-(lmax+1)), dtype=np.complex128)
        ## Collapse the two antenna dimensions into one baseline dimension
        # Nfreqs, Ntimes, Nant1, Nant2, Nalms --> Nbl, Nfreqs, Ntimes, Nalms 
        for i, bl in enumerate(auto_ants):
            idx1 = ants.index(bl[0])
            idx2 = ants.index(bl[1])
            autos[i, :] = vis_alm[:, :, idx1, idx2, :]   

        ## Reshape to 2D                                      ## TODO: Make this into a "pack" and "unpack" function
        # Nbl, Nfreqs, Ntimes, Nalms --> Nvis, Nalms
        Nautos = len(auto_ants) * len(freqs) * len(lsts)
        # Nvis = np.prod([len(antpairs),*vis_alm.shape[:-3]])
        autos_2D = autos.reshape(Nautos, 2*len(ell)-(lmax+1))

    
    if autos_only == False and include_autos == False:
        return vis_response_2D, autos_2D, ell, m
    else:
        return vis_response_2D, ell, m

def alms2healpy(alms, lmax):
    """
    Takes a real array split as [real, imag] (without the m=0 modes 
    imag-part) and turns it into a complex array of alms (positive 
    modes only) ordered as in HEALpy.
      
     Parameters
    ----------
    * alms (ndarray (floats))
            Array of zeros except for the specified mode. 
            The array represents all positive (+m) modes including zero 
            and has double length, as real and imaginary values are split. 
            The first half is the real values.

    
    Returns
    -------
    * healpy_modes (ndarray (complex)):
            Array of zeros except for the specified mode. 
            The array represents all positive (+m) modes including zeroth modes.
            
    """
    
    real_imag_split_index = int((np.size(alms)+(lmax+1))/2)
    real = alms[:real_imag_split_index]
    
    add_imag_m0_modes = np.zeros(lmax+1)
    imag = np.concatenate((add_imag_m0_modes, alms[real_imag_split_index:]))
    
    healpy_modes = real + 1.j*imag
    
    return healpy_modes
    
    
def healpy2alms(healpy_modes):
    """
    Takes a complex array of alms (positive modes only) and turns into
    a real array split as [real, imag] making sure to remove the 
    m=0 modes from the imag-part.
      
    Parameters
    ----------
    * healpy_modes (ndarray (complex)):
            Array of zeros except for the specified mode. 
            The array represents all positive (+m) modes including zeroth modes.
    
    Returns
    -------
    * alms (ndarray (floats))
            Array of zeros except for the specified mode. 
            The array represents all positive (+m) modes including zero 
            and is split into a real (first) and imag (second) part. The
            Imag part is smaller as the m=0 modes shouldn't contain and 
            imaginary part. 
    """
    lmax = hp.sphtfunc.Alm.getlmax(healpy_modes.size) # to remove the m=0 imag modes
    alms = np.concatenate((healpy_modes.real,healpy_modes.imag[(lmax+1):]))
        
    return alms   

def get_healpy_from_gsm(freq, lmax, nside=64, resolution="low", output_model=False, output_map=False):
    """
    Generate an array of alms (HEALpy ordered) from gsm 2016 (https://github.com/telegraphic/pygdsm)
    
    Parameters
    ----------
    * freqs: (float or np.array)
        Frequency (in MHz) for which to return GSM model
        
    * lmax: (int)
        Maximum l value for alms
        
    * nside: (int)
        The NSIDE you want to upgrade/downgrade the map to. Default is nside=64.

    * resolution: (str)
        if "low/lo/l":  The GSM nside = 64  (default)
        if "hi/high/h": The GSM nside = 1024 

    * output_model: (Boolean) optional
        If output_model=True: Outputs model generated from the GSM data. 
        If output_model=False (default): no model output.

    * output_map: (Boolean) optional
        If output_map=True: Outputs map generated from the GSM data. 
        If output_map=False (default): no map output.

    Returns
    -------
    *healpy_modes: (np.array)
        Complex array of alms with same size and ordering as in healpy (m,l)
    
    *gsm_2016: (PyGDSM 2016 model) optional
        If output_model=True: Outputs model generated from the GSM data. 
        If output_model=False (default): no model output.

    *gsm_map: (healpy map) optional
        If output_map=True: Outputs map generated from the GSM data. 
        If output_map=False (default): no map output.
    
    """
    gsm_2016 = GlobalSkyModel2016(freq_unit='MHz', resolution=resolution) 
    gsm_map = gsm_2016.generate(freqs=freq)
    gsm_upgrade = hp.ud_grade(gsm_map, nside)
    healpy_modes_gal = hp.map2alm(maps=gsm_upgrade,lmax=lmax)

    # Per default it is in gal-coordinates, convert to equatorial
    rot_gal2eq = hp.Rotator(coord="GC")
    healpy_modes_eq = rot_gal2eq.rotate_alm(healpy_modes_gal)

    if output_model == False and output_map == False: # default
        return healpy_modes_eq
    elif output_model == False and output_map == True:
        return healpy_modes_eq, gsm_map 
    elif output_model == True and output_map == False:
        return healpy_modes_eq, gsm_2016 
    else:
        return healpy_modes_eq, gsm_2016, gsm_map

def get_alms_from_gsm(freq, lmax, nside=64, resolution='low', output_model=False, output_map=False):
    """
    Generate a real array split as [real, imag] (without the m=0 modes 
    imag-part) from gsm 2016 (https://github.com/telegraphic/pygdsm)
    
    Parameters
    ----------
    * freqs: (float or np.array)
        Frequency (in MHz) for which to return GSM model
        
    * lmax: (int)
        Maximum l value for alms
        
    * nside: (int)
        The NSIDE you want to upgrade/downgrade the map to. Default is nside=64.
        
    * resolution: (str)
        if "low/lo/l":  nside = 64  (default)
        if "hi/high/h": nside = 1024 
        
    * output_model: (Boolean) optional
        If output_model=True: Outputs model generated from the GSM data. 
        If output_model=False (default): no model output.
        
    * output_map: (Boolean) optional
        If output_map=True: Outputs map generated from the GSM data. 
        If output_map=False (default): no map output.

    Returns
    -------
    * alms (ndarray (floats))
            Array of zeros except for the specified mode. 
            The array represents all positive (+m) modes including zero 
            and has double length, as real and imaginary values are split. 
            The first half is the real values.
            
    * gsm_2016: (PyGDSM 2016 model) optional
        If output_model=True: Outputs model generated from the GSM data. 
        If output_model=False (default): no model output.
            
    * gsm_map: (healpy map) optional
        If output_map=True: Outputs map generated from the GSM data. 
        If output_map=False (default): no map output.
    
    """
    return healpy2alms(get_healpy_from_gsm(freq, lmax, nside, resolution, output_model, output_map))

def construct_rhs_no_rot(data, inv_noise_cov, inv_prior_cov, omega_0, omega_1, a_0, vis_response):
    
    real_data_term = vis_response.real.T @ (inv_noise_cov*data.real + np.sqrt(inv_noise_cov)*omega_1.real)
    imag_data_term = vis_response.imag.T @ (inv_noise_cov*data.imag + np.sqrt(inv_noise_cov)*omega_1.imag)
    prior_term = inv_prior_cov*a_0 + np.sqrt(inv_prior_cov)*omega_0

    right_hand_side = real_data_term + imag_data_term + prior_term 
    
    return right_hand_side

def apply_lhs_no_rot(a_cr, inv_noise_cov, inv_prior_cov, vis_response):
    
    real_noise_term = vis_response.real.T @ ( inv_noise_cov[:,np.newaxis]* vis_response.real ) @ a_cr
    imag_noise_term = vis_response.imag.T @ ( inv_noise_cov[:,np.newaxis]* vis_response.imag ) @ a_cr
    signal_term = inv_prior_cov * a_cr
    
    left_hand_side = (real_noise_term + imag_noise_term + signal_term) 
    
    return left_hand_side

def radiometer_eq(auto_visibilities, ants, delta_time, delta_freq, Nnights = 1, include_autos=False):
    nbls = len(ants)
    indx = auto_visibilities.shape[0]//nbls
    
    sigma_full = np.empty((0))#, autos.shape[-1]))

    for i in ants:
        vis_ii = auto_visibilities[i*indx:(i+1)*indx]#,:]

        for j in ants:
            if include_autos == True:
                if j >= i:
                    vis_jj = auto_visibilities[j*indx:(j+1)*indx]#,:]
                    sigma_ij = ( vis_ii*vis_jj ) / ( Nnights*delta_time*delta_freq )
                    sigma_full = np.concatenate((sigma_full,sigma_ij))
            else:
                if j > i:  # only keep this line if you don't want the auto baseline sigmas
                    vis_jj = auto_visibilities[j*indx:(j+1)*indx]#,:]
                    sigma_ij = ( vis_ii*vis_jj ) / ( Nnights*delta_time*delta_freq )
                    sigma_full = np.concatenate((sigma_full,sigma_ij))
                    
    return sigma_full

## Plotting functions:
def map_plot(skymap, title, file_name, vmin=-450, vmax=5000, logged=False):
    """
    file_name = name of file
    title = title of figure
    logged: bool
        Take the np.log2 of the data before plotting. Defaults to False.
    """
    fig, (ax1) = plt.subplots(ncols=1, figsize=(10,6))
    plt.axes(ax1)

    cmap=cmo.haline
    
    if logged == False:
        cbar = False
    else:
        skymap = np.log2(skymap)
        cbar = True
    
    hp.mollview(skymap, title=title,
                cmap=cmap,
                hold=True,
                notext=True,
                #bgcolor="#d5bc5e00",
                cbar = cbar
               )
    hp.graticule()
    
    if logged == False:
        # ax2 = fig.add_subplot(ax1)   #used to check actual ax height, as cbar position depends on this
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("bottom", size="5%", pad="1%")

        cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap),
                 cax=cax, orientation='horizontal')

        # ticks = np.linspace(skymap.min(),skymap.max(),6)    
        ticks = np.linspace(vmin,vmax,6)    
        labels = [f'{tick:.3f}' for tick in ticks]

        cbar.locator = ticker.MaxNLocator(nbins=6)
        cbar.update_ticks()
        cbar.ax.set_xticklabels(labels) 
    
    fig.savefig(file_name+'.png', 
                bbox_inches='tight', 
                #transparent=True, 
                dpi=fig.dpi,
                facecolor='white')
    return None
    
def mag_plot(alms, y, title, file_name, y2=[], l1=False, l2=False):
    
    mode_indices = np.arange(len(alms))

    fig = plt.figure(figsize=(12,6)) 

    plt.title(title) 
    
    if len(y2) == 0:
        if l1 == False:
            plt.plot(mode_indices,y)
        else:
            plt.plot(mode_indices,y, label=l1)
            plt.legend()
    else:
        if l1 and l2 == False:
            plt.plot(mode_indices,y)
            plt.plot(mode_indices,y2)
        else:
            plt.plot(mode_indices,y, label=l1)
            plt.plot(mode_indices,y2, label=l2)
            plt.legend()
            
    
    plt.yscale('log')
    plt.grid('minor')
    plt.xlabel('modes')
    
    plt.savefig(file_name+'.png',
                bbox_inches='tight',
                transparent=False,
                dpi=fig.dpi)
    
    return None
    
def lin_plot(alms, y, title, file_name, y2=np.empty((0)), l1=False, l2=False):
    
    mode_indices = np.arange(len(alms))

    fig = plt.figure(figsize=(12,6)) 

    plt.title(title) 
    
    if len(y2) == 0:
        if l1 == False:
            plt.plot(mode_indices,y)
        else:
            plt.plot(mode_indices,y, label=l1)
            plt.legend()
    else:
        if l1 and l2 == False:
            plt.plot(mode_indices,y)
            plt.plot(mode_indices,y2)
        else:
            plt.plot(mode_indices,y, label=l1)
            plt.plot(mode_indices,y2, label=l2)
            plt.legend()
            
    plt.grid('minor')
    plt.xlabel('modes')
    
    plt.savefig(file_name+'.png',
                bbox_inches='tight',
                transparent=False,
                dpi=fig.dpi)
    return None
    
def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map

    Notes:
    
    If base_cmap is a string or None, you can simply do return plt.cm.get_cmap(base_cmap, N)
    The following works for string, None, or a colormap instance:
    """

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    
    return base.from_list(cmap_name, color_list, N)
    
def discrete_log_map_plot(residual_map, title, file_name):
    """
    residual_map = map made from subtracting the true solution from the solver solution using hp.alm2map. 
    file_name = name of file
    title = title of figure
    """
    
    skymap = np.log10(np.abs(residual_map))
    
    # # Set the min/max to be exactly 10 magnitudes. 
    # if np.abs(skymap.min()) >= np.abs(skymap.max()):
    #     #vmin = np.floor(skymap.min())
    #     vmin = np.floor(skymap.min())
    #     vmax = vmin + 10
    # else:
    #     #vmax = np.ceil(skymap.max())
    #     vmax = np.ceil(skymap.max())
    #     vmin = vmax - 10
        
    cmap = discrete_cmap(12, cmo.thermal)

    vmin = -6
    vmax = 6
    
    # Plot
    fig, (ax1) = plt.subplots(ncols=1, figsize=(10,6))
    plt.title('test')
    
    hp.mollview(skymap, title=title,
                min = vmin,
                max = vmax,
                cmap = cmap,
                hold = True,
                notext = True,
                #bgcolor = "#d5bc5e00",
                cbar = False
               )
    hp.graticule()

    # Everything for the custom colourbar with tick labels and 10 magnitudes:
    bounds = np.arange(vmin,vmax+1,1)    
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    #ax2 = fig.add_subplot(ax1)   #used to check actual ax height, as cbar position depends on this
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("bottom", size="5%", pad="1%")


    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal')
    
    ## In case you wanna control the formatting of the labels:
    labels = [f'{bound:.3f}' for bound in bounds]
    cbar.ax.set_xticklabels(labels) 
    

    fig.savefig(file_name+'.png', 
                bbox_inches='tight', 
                #transparent=True, 
                dpi=fig.dpi,
                facecolor='white')
    return None


# MAIN    
if __name__ == "__main__":
    start_time = time.time()
    
    # Creating directory for output
    if ARGS['directory']: 
        directory = str(ARGS['directory'])
    else:
        directory = "output"

    path = f'/cosma8/data/dp270/dc-glas1/{directory}/'
    if not os.path.isdir(path): os.makedirs(path)

    ant_pos = build_hex_array(hex_spec=(3,4), d=14.6)  #builds array with (3,4,3) ants = 10 total
    
    ## Randomising the ants:
    #for key in ant_pos.keys():    
    #        ant_pos[key] = (ant_pos[key][0] * 5*np.random.rand(1),
    #                        ant_pos[key][1] * 5*np.random.rand(1), 
    #                        0.)
    ants = list(ant_pos.keys())
    lmax = 20
    nside = 128
    beam_diameter = 14.
    beams = [pyuvsim.AnalyticBeam('gaussian', diameter=beam_diameter) for ant in ants]
    freqs = np.linspace(100e6, 102e6, 2)
    lsts_hours = np.linspace(16.,24.,10)      # in hours for easy setting
    lsts = np.deg2rad((lsts_hours/24)*360) # in radian, used by HYDRA (and this code)
    delta_time = 60 #s
    delta_freq = 1e+06 # (M)Hz
    latitude = 31.7215 * np.pi / 180  # HERA loc in decimal numbers ## There's some sign error in the code, so this missing sign is a quick fix
    solver = cg

    vis_response, autos, ell, m = vis_proj_operator_no_rot(freqs=freqs, 
                                                        lsts=lsts, 
                                                        beams=beams, 
                                                        ant_pos=ant_pos, 
                                                        lmax=lmax, 
                                                        nside=nside,
                                                        latitude=latitude)

    np.random.seed(10)
    x_true = get_alms_from_gsm(freq=100,lmax=lmax, nside=nside)
    model_true = vis_response @ x_true

    # Inverse noise covariance and noise on data
    noise_cov = radiometer_eq(autos@x_true, ants, delta_time, delta_freq)
    inv_noise_cov = 1/noise_cov
    data_noise = np.random.randn(noise_cov.size)*np.sqrt(noise_cov) 
    data_vec = model_true + data_noise

    # Inverse signal covariance
    zero_value = 0.001
    prior_cov = (x_true*0.1)**2     # if 0.1 = 10% prior
    prior_cov[prior_cov == 0] = zero_value
    inv_prior_cov = 1/prior_cov
    a_0 = np.random.randn(x_true.size)*np.sqrt(prior_cov) + x_true # gaussian centered on alms with S variance 

    # Define left hand side operator 
    def lhs_operator(x):
        y = apply_lhs_no_rot(x, inv_noise_cov, inv_prior_cov, vis_response)

        return y

    # Wiener filter solution to provide initial guess:
    omega_0_wf = np.zeros_like(a_0)
    omega_1_wf = np.zeros_like(model_true, dtype=np.complex128)
    rhs_wf = construct_rhs_no_rot(data_vec,
                                  inv_noise_cov, 
                                  inv_prior_cov, 
                                  omega_0_wf, 
                                  omega_1_wf, 
                                  a_0, 
                                  vis_response)
    
    # Build linear operator object 
    lhs_shape = (rhs_wf.size, rhs_wf.size)
    lhs_linear_op = LinearOperator(matvec = lhs_operator,
                                       shape = lhs_shape)

    # Get the Wiener Filter solution for initial guess
    wf_soln, wf_convergence_info = solver(A = lhs_linear_op,
                                          b = rhs_wf,
                                          # tol = 1e-07,
                                          maxiter = 15000)
    
    def samples(key):
        t_iter = time.time()

        # Set a random seed defined by the key
        np.random.seed(key)
        #random_seed = np.random.get_state()[1][0] #for test/output purposes

        # Generate random maps for the realisations
        omega_0 = np.random.randn(a_0.size)
        omega_1 = (np.random.randn(model_true.size) + 1.j*np.random.randn(model_true.size))/np.sqrt(2)

        # Construct the right hand side
        rhs = construct_rhs_no_rot(data_vec,
                                   inv_noise_cov, 
                                   inv_prior_cov,
                                   omega_0,
                                   omega_1,
                                   a_0,
                                   vis_response)

        # Run and time solver
        time_start_solver = time.time()
        x_soln, convergence_info = solver(A = lhs_linear_op,
                                          b = rhs,
                                          # tol = 1e-07,
                                          maxiter = 15000,
                                          x0 = wf_soln) #initial guess
        solver_time = time.time() - time_start_solver
        iteration_time = time.time()-t_iter
        
        # Save output
        np.savez(path+'results_'+f'{key}',
                 omega_0=omega_0,
                 omega_1=omega_1,
                 key=key,
                 x_soln=x_soln,
                 rhs=rhs,
                 convergence_info=convergence_info,
                 solver_time=solver_time,
                 iteration_time=iteration_time
        )
        
        return key, iteration_time
    
    # Time for all precomputations
    precomp_time = time.time()-start_time
    print(f'\nprecomputation took:\n{precomp_time}\n')
    
    avg_iter_time = 0

    # Multiprocessing, getting the samples    
    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    print(f'\nSLURM_CPUS_PER_TASK = {number_of_cores}')

    with Pool(number_of_cores) as pool:
        # issue tasks and process results
        for result in pool.map(samples, range(100)):
            key, iteration_time = result
            avg_iter_time += iteration_time
            #print(f'Iteration {key} completed in {iteration_time:.2f} seconds')

    avg_iter_time /= (key+1)
    print(f'average_iter_time:\n{avg_iter_time}\n')

    total_time = time.time()-start_time
    print(f'total_time:\n{total_time}\n')
    print(f'All output saved in folder {path}\n')
    print(f'Note, ant_pos (dict) is saved in own file in {path}\n')
    
    # Saving all globally calculated data
    np.savez(path+'precomputed_data',
             vis_response=vis_response,
             x_true=x_true,
             inv_noise_cov=inv_noise_cov,
             zero_value=zero_value,
             inv_prior_cov=inv_prior_cov,
             wf_soln=wf_soln,
             nside=nside,
             lmax=lmax,
             ants=ants,
             beam_diameter=beam_diameter,
             freqs=freqs,
             lsts_hours=lsts_hours,
             precomp_time=precomp_time,
             total_time=total_time
            )
    # creating a dictionary with string-keys as required by .npz files
    ant_dict = dict((str(ant), ant_pos[ant]) for ant in ant_pos)
    np.savez(path+'ant_pos',**ant_dict)
    
    
