import numpy as np
from astropy.io import fits

def get_kappa_millennium(los):
    return get_millennium(los)[:,:,4]


def get_millennium(los):
    los_no1 = los//8
    los_no2 = los%8
    ms_field = np.loadtxt("/vol/euclid2/euclid2_raid2/sven/millennium_maps/41_los_8_"+ str(los_no1) +"_"+ str(los_no2) +".ascii")
    ms_field = ms_field.reshape(4096,4096,5)
    return ms_field

def get_gamma_millennium(los):
    data = get_millennium(los)
    return data[:,:,2] + 1.0j*data[:,:,3]

def get_kappa_slics(los):
    filestr = "/vol/euclid7/euclid7_2/llinke/HOWLS/convergence_maps/SLICS_LCDM/kappa_noise_GalCatalog_LOS_cone"+str(los)+".fits_s333"+str(los)+"_zmin0.0_zmax3.0_sys_3.fits_ks_nomask_shear.fits"
    return fits.open(filestr)[0].data


def get_slics(los):
    hdul = fits.open('/vol/euclid7/euclid7_2/sven/slics_lsst/GalCatalog_LOS_cone'+str(los)+'.fits')
    data = hdul[1].data

    Xs = data['x_arcmin']
    Ys = data['y_arcmin']
    shears1 = data['shear1']
    shears2 = data['shear2']

    return Xs,Ys,shears1,shears2

