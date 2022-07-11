""" Functions for the creation of Gaussian and lognormal random fields
    """
from random import gauss
import numpy as np
from lenstools import GaussianNoiseGenerator
import warnings
from astropy import units as u
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.special import jv,jn_zeros


def create_gaussian_random_field(power_spectrum, n_pix=4096, fieldsize=4.*np.pi/180, random_seed=None,
                                 fill_value=1e-25):
    """Creates Gaussian Random Field from a given power spectrum with mean 0 and variance sigma. Uses lenstools.

    Args:
        power_spectrum (function): Function describing P(ell)
        n_pix (int, optional): Pixelnumber along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [rad]. Defaults to 4.*np.pi/180.
        random_seed (int, optional): Seed for random number generator. Defaults to None, then a random seed is chosen.
        fill_value (float, optional): Value put in if P(ell)<=0. Defaults to 1e-25.

    Returns:
        np.array: Gaussian field as 2D map
    """

    # Create array for power spectrum
    ell_min = 0
    two_ell_max = 2.*np.pi/fieldsize*n_pix
    ell_array = np.linspace(ell_min, two_ell_max, 2*n_pix)
    psarray = power_spectrum(ell_array)
    if(np.any(psarray <= 0)):
        warnings.warn(
            "Psarray contains zeros/negatives. Substituting with {}".format(fill_value))
        psarray[psarray <= 0] = fill_value

    # Create Map
    gaussian_map=create_gaussian_random_field_array(ell_array, psarray, n_pix=n_pix, fieldsize=fieldsize, random_seed=random_seed)

    return gaussian_map.data


def create_gaussian_random_field_array(ell_array, power_spectrum_array, n_pix=4096, fieldsize=4.*np.pi/180, random_seed=None):
    """Creates Gaussian Random Field from a given power spectrum array with mean 0 and variance sigma. Uses lenstools.

    Args:
        ell_array (np.array): ell-values for power spectrum
        power_spectrum_array (np.array): P(ell) values
        n_pix (int, optional): _description_. Defaults to 4096.
        fieldsize (float, optional): Sidelength [rad]. Defaults to 4.*np.pi/180.
        random_seed (int, optional): Seed for random number generator. Defaults to None, then a random seed is chosen.

    Returns:
        np.array: Gaussian field as 2D map
    """
    # Create Gaussian Noise Generator
    gen = GaussianNoiseGenerator(
        shape=(n_pix, n_pix), side_angle=fieldsize*u.rad)
    if random_seed is None:  # Choose random seed
        random_seed = np.random.randint(0, 2**32)

    # Create Map
    gaussian_map = gen.fromConvPower(np.array([ell_array, power_spectrum_array]), seed=random_seed, kind="linear",
                                     bounds_error=False, fill_value=0.0)

    return gaussian_map.data


def create_lognormal_random_field_from_Gaussian_powerspec(power_spectrum_of_gaussian_random_field, alpha, sigma=1., npix=4096,
                                                          fieldsize=10.*np.pi/180., random_seed=None):
    """ Creates a Lognormal field from the corresponding Gaussian Powerspectrum and non-Gaussianity parameter alpha

    Args:
        power_spectrum_of_gaussian_random_field (function): Powerspectrum of Gaussian field corresponding to Lognormal field
        alpha (float): Alpha parameter for non-linearity (See Hilbert+ 2012 for definition)
        sigma (float, optional): standard deviation (wrt lognormal field). Defaults to 1..
        npix (int, optional): Pixelnumber along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [rad]. Defaults to 10.*np.pi/180.
        random_seed (int, optional): Seed for random number generator. Defaults to None, then a random seed is chosen.
    """

    # Create Gaussian random field
    gaussian_random_field = create_gaussian_random_field(
        power_spectrum_of_gaussian_random_field, n_pix=npix, fieldsize=fieldsize, random_seed=random_seed)

    # Get Prefactors
    c = np.exp(alpha**2/2)
    new_field_prefactor = sigma/(c*np.sqrt(c**2-1))
    new_field = new_field_prefactor * \
        (np.exp(alpha*gaussian_random_field/sigma)-c)
    return new_field


def create_lognormal_random_field_array(ell_array, power_spectrum_array, alpha, sigma=1., npix=4096, fieldsize=10.*np.pi/180.,
                                        random_seed=None):
    """Creates Lognormal Random Field from a given power spectrum array and Non-Gaussianity Parameter alpha


    Args:
        ell_array (np.array): ell-values for power spectrum
        power_spectrum_array (np.array): P(ell) values
        alpha (float): Alpha parameter for non-linearity (See Hilbert+ 2012 for definition)
        sigma (float, optional): standard deviation (wrt lognormal field). Defaults to 1..
        npix (int, optional): Pixelnumber along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [rad]. Defaults to 10.*np.pi/180.
        random_seed (int, optional): Seed for random number generator. Defaults to None, then a random seed is chosen.

    Returns:
        np.array: Lognormal field as 2D map
    """
    # Interpolate power spectrum array to function
    power_spec_func = interp1d(
        ell_array, power_spectrum_array, fill_value=0, bounds_error=False)

    # Calculate standard deviation
    sigma = np.sqrt(integrate.quad(lambda ell: ell *
                                   power_spec_func(ell), 10, 10**5)[0]/(2.*np.pi))

    # Get Power spec of corresponding Gaussian field
    power_spec_gauss = get_gaussian_power_spectrum(
        power_spec_func, alpha, sigma)

    # Generate Lognormal field
    field = create_lognormal_random_field_from_Gaussian_powerspec(
        power_spec_gauss, alpha, sigma, npix, fieldsize, random_seed)

    return field


def create_lognormal_random_field(power_spectrum, alpha, n_pix=4096, fieldsize=4.*np.pi/180, random_seed=None,
                                 fill_value=1e-25):
    """Creates Lognormal Random Field from a given power spectrum and Non-Gaussianity Parameter alpha

    Args:
        power_spectrum (function): Function describing P(ell)
        n_pix (int, optional): Pixelnumber along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [rad]. Defaults to 4.*np.pi/180.
        random_seed (int, optional): Seed for random number generator. Defaults to None, then a random seed is chosen.
        fill_value (float, optional): Value put in if P(ell)<=0. Defaults to 1e-25.

    Returns:
        np.array: Lognormal field as 2D map
    """

    # Create array for power spectrum
    ell_min = 0
    two_ell_max = 2.*np.pi/fieldsize*n_pix
    ell_array = np.linspace(ell_min, two_ell_max, 2*n_pix)
    psarray = power_spectrum(ell_array)
    if(np.any(psarray <= 0)):
        warnings.warn(
            "Psarray contains zeros/negatives. Substituting with {}".format(fill_value))
        psarray[psarray <= 0] = fill_value

    # Create Map
    lognormal_map=create_lognormal_random_field_array(ell_array, psarray, alpha, n_pix=n_pix, fieldsize=fieldsize, random_seed=random_seed)

    return lognormal_map.data


def get_gaussian_power_spectrum(power_spectrum, alpha, sigma,
                                thetas=np.arange(
                                    0.000001, 8.*np.pi/180, 2.*np.pi/180/1000),
                                ells=np.geomspace(10, 10**5, 1000)):
    """ Generates powerspectrum of Gaussian field corresponding to a lognormal field

    Args:
        power_spectrum (function): Powerspectrum (lognormal)
        alpha (float): Alpha parameter for non-linearity (See Hilbert+ 2012 for definition)
        sigma (float, optional): standard deviation (wrt lognormal field). Defaults to 1..
        thetas (np.array, optional): theta bins for correlation function [rad]. Defaults to np.arange( 0.000001, 8.*np.pi/180, 2.*np.pi/180/1000).
        ells (np.array, optional): ell-bins. Defaults to np.geomspace(10, 10**5, 1000).

    Returns:
        function: Gaussian powerspectrum
    """
    # Calculates correlation function corresponding to powerspectrum
    corr = correlation_function_from_power_spectrum(thetas, power_spectrum)

    # Calculates correlation function of Gaussian field
    corr_gaussian = lognormal_to_gaussian(corr, alpha, sigma)

    # Interpolate correlation function of Gaussian field
    corrf = interp1d(thetas, corr_gaussian, fill_value=0, bounds_error=False)

    # Get Powerspectrum corresponding to Gaussian correlation function
    ps = power_spectrum_of_correlation_function(ells, corrf)
    return interp1d(ells, ps, fill_value=0, bounds_error=False)


def get_lognormal_power_spectrum(power_spectrum, alpha, sigma,
                                thetas=np.arange(
                                    0.000001, 8.*np.pi/180, 2.*np.pi/180/1000),
                                ells=np.geomspace(10, 10**5, 1000)):
    """ Generates powerspectrum of lognormal field corresponding to a Gaussian field

    Args:
        power_spectrum (function): Powerspectrum (gaussian)
        alpha (float): Alpha parameter for non-linearity (See Hilbert+ 2012 for definition)
        sigma (float, optional): standard deviation (wrt lognormal field). Defaults to 1..
        thetas (np.array, optional): theta bins for correlation function [rad]. Defaults to np.arange( 0.000001, 8.*np.pi/180, 2.*np.pi/180/1000).
        ells (np.array, optional): ell-bins. Defaults to np.geomspace(10, 10**5, 1000).

    Returns:
        function: Gaussian powerspectrum
    """
    # Calculates correlation function corresponding to powerspectrum
    corr = correlation_function_from_power_spectrum(thetas, power_spectrum)

    # Calculates correlation function of Lognormal
    corr_lognormal = gaussian_to_lognormal(corr, alpha, sigma)

    # Interpolate correlation function of Lognormal field
    corrf = interp1d(thetas, corr_lognormal, fill_value=0, bounds_error=False)

    # Get Powerspectrum corresponding to Lognormal correlation function
    ps = power_spectrum_of_correlation_function(ells, corrf)
    return interp1d(ells, ps, fill_value=0, bounds_error=False)


def lognormal_to_gaussian(vals,alpha,sigma=1):
    """Converts Lognormal correlation function to Gaussian

    Args:
        vals (np.array): Correlation function
        alpha (int): Non-Gaussianity parameter (see Hilbert+ 2012 for a definition)
        sigma (float, optional): standard deviation (wrt lognormal field). Defaults to 1..

    Returns:
        np.array: Gaussian correlation function
    """
    c = np.exp(alpha**2/2)
    A = sigma/np.sqrt(c**2-1)
    return sigma**2*np.log(vals/A**2+1)/alpha**2

def gaussian_to_lognormal(vals,alpha,sigma=1):
    """Converts Gaussian correlation function to Lognormal

    Args:
        vals (np.array): Correlation function
        alpha (int): Non-Gaussianity parameter (see Hilbert+ 2012 for a definition)
        sigma (float, optional): standard deviation (wrt lognormal field). Defaults to 1..

    Returns:
        np.array: Gaussian correlation function
    """
    c = np.exp(alpha**2/2)
    A = sigma/np.sqrt(c**2-1)
    return A**2*(np.exp(alpha**2*vals/sigma**2)-1)


def correlation_function_from_power_spectrum(x,power_spectrum):
    """Calculates the correlation function for a power spectrum

    Args:
        x (np.array): theta-bins for correlation function [rad]
        power_spectrum (function): Powerspectrum

    Returns:
        np.array: Correlation function
    """

    # Create integrator for bessel integrals
    bi=besselintegrator(0, 0.00002, 5.)

    factor = 1./(2*np.pi)
    
    def power_spectrum_ell(ell):
        return ell*power_spectrum(ell)

    # Do integration    
    if(hasattr(x,'__len__')):
        lx = len(x)
        result = np.zeros(lx)
        for i,xi in enumerate(x):
            result[i] = bi.integrate(power_spectrum_ell,xi)
        return factor*result
    else:
        return factor*bi.integrate(power_spectrum_ell,x)


def power_spectrum_of_correlation_function(ell,correlation_function):
    """Calculates the power spectrum for a correlation function

    Args:
        ell (np.array): ell-bins for power spectrum
        correlation_function (function): Correlation functin

    Returns:
        np.array: Power spectrum
    """
    # Create integrator for bessel integrals
    bi=besselintegrator(0, 0.00002, 5.)

    factor = 2*np.pi
    def correlation_function_x(x):
        return correlation_function(x)*x

    # Do integration
    if(hasattr(ell,'__len__')):
        ll = len(ell)
        result = np.zeros(ll)
        for i,l in enumerate(ell):
            result[i] = bi.integrate(correlation_function_x,l)
        return factor*result

    else:
        return factor*bi.integrate(correlation_function_x,ell)



class besselintegrator:
    """ Class for calculating Bessel integrals
    """

    def __init__(self,n_dim_bessel,prec_h,prec_k):
        """Initialization

        Args:
            n_dim_bessel (int): Order of Bessel function
            prec_h (float): step width
            prec_k (int): Max root of Bessel function considered in the integral
        """
        self.n_dim_bessel = n_dim_bessel
        self.prec_h = prec_h
        self.prec_k = int(prec_k/prec_h)
        
        self.bessel_zeros = jn_zeros(n_dim_bessel,self.prec_k)
        self.pi_bessel_zeros = self.bessel_zeros/np.pi
        self.psiarr = np.pi*self.psi(self.pi_bessel_zeros*self.prec_h)/self.prec_h
        self.besselarr = jv(self.n_dim_bessel,self.psiarr)
        self.psiparr = self.psip(self.prec_h*self.pi_bessel_zeros)
        self.warr = 2/(np.pi*self.bessel_zeros*np.jv(n_dim_bessel+1,self.bessel_zeros)**2)
    
    def psi(self,t):
        return t*np.tanh(np.pi*np.sinh(t)/2)
    def psip(self,t):
        zahler = np.sinh(np.pi*np.sinh(t))+np.pi*t*np.cosh(t)
        nenner = np.cosh(np.pi*np.sinh(t))+1
        return zahler/nenner
    

    def integrate(self,function,R):
        """Computes the Integral int f(k)J(kR) dk"""
        return np.pi/R*np.sum(self.warr*function(self.psiarr/R)*self.besselarr*self.psiparr)
    

