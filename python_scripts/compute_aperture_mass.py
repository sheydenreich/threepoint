import numpy as np
from astropy.io import fits
import os
import sys
from utility import aperture_mass_computer
"""
positions of files:
millennium simulations: 
- /vol/aibn1113/data1/sven/millenium/ (can be read by using the function get_millennium)
- ascii-files, can best be accessed with np.loadtxt
- 4096x4096 grid of gamma1, gamma2 and kappa values at z=1 for 64 lines of sight spanning 4x4 deg
- can be brought into that form by np.reshape(get_millennium(los),(4096,4096,5)), then
    - 1st column: x-position
    - 2nd column: y-position
    - 3rd column: gamma1
    - 4th column: gamma2
    - 5th column: kappa

SLICS without masks(euclid-like number density and redshift distribution):
- /vol/euclid7/euclid7_2/sven/slics_lsst
- .fits files, can best be accessed with fits from astropy.io
- example: get_slics function
- dictionary galaxy catalogue, relevant entries are x_arcmin,y_arcmin,shear1 and shear2
- in total 958 lines-of-sight spanning 10x10 degree

cosmo-SLICS without masks (euclid-like number density and redshift distribution):
- /vol/euclid7/euclid7_2/sven/cosmoslics_lsst/$cosmo_$lett
- file format same as SLICS
- 26 cosmologies, 2 N-body simulations each (denoted by _a/ and _f/), and 10 semi-independent lines-of-sight per N-body simulation

SLICS and cosmo-SLICS tiled to the DES_Y1 data are on /vol/euclid7/euclid7_1, but they have a different file format, if you need them just ask :)
"""

def compute_map_slics(los,npix=4096,theta_ap_array=[0.5,1,2,4,8,16,32],force=True,shapenoise=True):
    ac = aperture_mass_computer(npix,1,10*60)
    try:
        positions, shears = get_slics(los,shapenoise=shapenoise)
    except Exception as inst:
        sys.stderr.write(str(inst))
        print(inst)
        return 0
        
    shears,norm = ac.normalize_shear(positions[0],positions[1],shears)

    for theta_ap in theta_ap_array:
        savepath = "/vol/euclid2/euclid2_raid2/sven/maps_slics_euclid_npix_4096_estimator_weighted_aperture/theta_"+str(theta_ap)+"_los_"+str(los)
        if(shapenoise):
            savepath += "_with_shapenoise"
        if not os.path.exists(savepath+".npy") or force:
            ac.change_theta_ap(theta_ap)
            result = ac.Map_fft(shears,norm=norm,return_mcross=True)
            np.save(savepath,result)
    return 0


def get_slics(los,shapenoise=True,
             prefix='/vol/euclid7/euclid7_2/llinke/HOWLS/shear_catalogues/SLICS_LCDM/SLICS/GalCatalog_LOS_cone',
             suffix1='.fits_s333',
             suffix2='_zmin0.0_zmax3.0.fits'):
    hdul = fits.open(prefix+str(los)+suffix1+str(los)+suffix2)
    data = hdul[1].data

    Xs = data['x_arcmin']
    Ys = data['y_arcmin']

    shear = data['shear1'] + 1.0j*data['shear2']

    if(shapenoise):
        noise = data['e1_intr'] + 1.0j*data['e2_intr']
        
        shear = (shear + noise)/(1+shear*np.conj(noise))
    return [Xs,Ys],shear







if(__name__=='__main__'):
    import multiprocessing as mp
    from tqdm import tqdm
    n_processes = 64
    arglist = range(74,74+500)
    with mp.Pool(processes=n_processes) as p:
        for i in tqdm(p.imap_unordered(compute_map_slics, arglist),total=len(arglist)):
            pass
