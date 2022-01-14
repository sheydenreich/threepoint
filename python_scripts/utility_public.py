import numpy as np
import multiprocessing.managers
from scipy import stats
from lenstools import GaussianNoiseGenerator
from astropy import units as u
class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def D(npix = 4096,pixsize = 1.):
    xs1,xs2 = np.indices((npix,npix))
    xs1 = (xs1 - npix/2)*pixsize
    xs2 = (xs2 - npix/2)*pixsize
    a = (-xs1**2+xs2**2-xs1*xs2*2.j)/((xs1**2+xs2**2)**2)
    a[(xs1**2+xs2**2==0)] = 0
    return a

def Dhat_func(npix = 4096,pixsize = 1.):
    xs1,xs2 = np.indices((npix,npix))
    xs1 = (xs1 - npix/2)*pixsize
    xs2 = (xs2 - npix/2)*pixsize
    with np.errstate(divide="ignore",invalid="ignore"):
        a = (xs1**2-xs2**2+2.j*xs1*xs2)/(xs1**2+xs2**2)
    a[(xs1**2+xs2**2==0)] = 0
    return a

def create_gaussian_random_field(power_spectrum, n_pix=4096,fieldsize=4.*np.pi/180,random_seed=None):
    """creates gaussian random field from given power spectrum, with mean 0 and variance sigma"""
    ell_min = 0
    two_ell_max = 2.*np.pi/fieldsize*n_pix
    ell_array = np.linspace(ell_min,two_ell_max,2*n_pix)
    gen = GaussianNoiseGenerator(shape=(n_pix,n_pix),side_angle=fieldsize*u.rad)
    if random_seed is None:
        random_seed = np.random.randint(0,2**32)
    gaussian_map = gen.fromConvPower(np.array([ell_array,power_spectrum(ell_array)]),seed=random_seed,kind="linear",bounds_error=False,fill_value=0.0)
    return gaussian_map.data

def create_gaussian_random_field_array(ell_array, power_spectrum_array, n_pix=4096,fieldsize=4.*np.pi/180,random_seed=None):
    """creates gaussian random field from given power spectrum array, with mean 0 and variance sigma"""
    gen = GaussianNoiseGenerator(shape=(n_pix,n_pix),side_angle=fieldsize*u.rad)
    if random_seed is None:
        random_seed = np.random.randint(0,2**32)
    gaussian_map = gen.fromConvPower(np.array([ell_array,power_spectrum_array]),seed=random_seed,kind="linear",bounds_error=False,fill_value=0.0)
    return gaussian_map.data    

def create_gamma_field(kappa_field,Dhat=None):
    if Dhat is None:
        Dhat = Dhat_func(npix=kappa_field.shape[0])
    fieldhat = np.fft.fftshift(np.fft.fft2(kappa_field))
    gammahat = fieldhat*Dhat
    gamma = np.fft.ifft2(np.fft.ifftshift(gammahat))
    return gamma

def extract_power_spectrum(field,fieldsize,
    bins=10,linlog='log',lmin=200,lmax=10**4):
    n_pix = field.shape[0]
    pixel_size = (fieldsize/n_pix)**2
    fourier_image = np.fft.fftn(field)
    fourier_amplitudes = np.abs(fourier_image)**2*pixel_size
    kfreq = np.fft.fftfreq(n_pix)*2*np.pi*n_pix/fieldsize
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    if not hasattr(bins,"__len__"):
        if(linlog=='lin'):
            kbins = np.linspace(lmin,lmax, bins+1)
            kvals = 0.5 * (kbins[1:] + kbins[:-1])

        if(linlog=='log'):
            kbins = np.geomspace(lmin,lmax,bins+1)
            kvals = np.exp(0.5 * (np.log(kbins[1:]) + np.log(kbins[:-1])))

    else:
        kbins = bins
        kvals = 0.5 * (kbins[1:] + kbins[:-1])


    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
    return Abins*pixel_size/fieldsize**2,kvals
