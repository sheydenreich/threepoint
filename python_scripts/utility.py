import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from astropy.convolution import interpolate_replace_nans,Gaussian2DKernel#,convolve_fft
import astropy.convolution as apc
import collections
import multiprocessing.managers
#from FyeldGenerator import generate_field
from scipy import stats
from lenstools import GaussianNoiseGenerator
from astropy import units as u
import matplotlib.pyplot as plt
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
    #gaussian_map.visualize()
    #plt.show()
    #plt.clf()
    return gaussian_map.data

def create_gaussian_random_field_array(ell_array, power_spectrum_array, n_pix=4096,fieldsize=4.*np.pi/180,random_seed=None):
    """creates gaussian random field from given power spectrum array, with mean 0 and variance sigma"""
    gen = GaussianNoiseGenerator(shape=(n_pix,n_pix),side_angle=fieldsize*u.rad)
    if random_seed is None:
        random_seed = np.random.randint(0,2**32)
    gaussian_map = gen.fromConvPower(np.array([ell_array,power_spectrum_array]),seed=random_seed,kind="linear",bounds_error=False,fill_value=0.0)
    #gaussian_map.visualize()
    #plt.show()
    #plt.clf()
    return gaussian_map.data    

def create_gamma_field(kappa_field,Dhat=None):
    if Dhat is None:
        Dhat = Dhat_func(npix=kappa_field.shape[0])
    fieldhat = np.fft.fftshift(np.fft.fft2(kappa_field))
    gammahat = fieldhat*Dhat
    gamma = np.fft.ifft2(np.fft.ifftshift(gammahat))
    return gamma

class aperture_mass_computer:
    """
    a class handling the computation of aperture masses.
    initialization:
        npix: number of pixel of desired aperture mass map
        theta_ap: aperture radius of desired aperture mass map (in arcmin)
        fieldsize: fieldsize of desired aperture mass map (in arcmin)
    """
    def __init__(self,npix,theta_ap,fieldsize,use_polynomial_filter = False):
        self.theta_ap = theta_ap
        self.npix = npix
        self.fieldsize = fieldsize
        self.use_polynomial_filter = use_polynomial_filter
        if(use_polynomial_filter):
            print("WARNING! Using polynomial filter!")


        # compute distances to the center in arcmin
        idx,idy = np.indices([self.npix,self.npix])
        idx = idx - ((self.npix)/2)
        idy = idy - ((self.npix)/2)

        self.idc = idx + 1.0j*idy
        self.dist = np.abs(self.idc)*self.fieldsize/self.npix

        # compute the Q filter function on a grid
        self.q_arr = self.Qfunc_array()
        self.u_arr = self.Ufunc_array()

        self.disk = np.zeros((self.npix,self.npix))
        self.disk[(self.dist<self.theta_ap)] = 1

    def change_theta_ap(self,theta_ap):
        self.theta_ap = theta_ap
        self.q_arr = self.Qfunc_array()
        self.u_arr = self.Ufunc_array()
        self.disk = np.zeros((self.npix,self.npix))
        self.disk[(self.dist<self.theta_ap)] = 1

    def Ufunc(self,theta):
        """
        The U filter function for the aperture mass calculation from Schneider et al. (2002)
        input: theta: aperture radius in arcmin
        """
        xsq_half = (theta/self.theta_ap)**2/2
        small_ufunc = np.exp(-xsq_half)*(1.-xsq_half)/(2*np.pi)
        return small_ufunc/self.theta_ap**2

    def Qfunc(self,theta):
        """
        The Q filter function for the aperture mass calculation from Schneider et al. (2002)
        input: theta: aperture radius in arcmin
        output: Q [arcmin^-2]
        """
        thsq = (theta/self.theta_ap)**2
        if(self.use_polynomial_filter):
            res = 6/np.pi*thsq**2*(1.-thsq**2)
            res[(thsq>1)] = 0
            return res/self.theta_ap**2
        else:
            res = thsq/(4*np.pi*self.theta_ap**2)*np.exp(-thsq/2)
            return res

    def Qfunc_array(self):
        """
        Computes the Q filter function on an npix^2 grid
        fieldsize: size of the grid in arcmin
        """
        with np.errstate(divide='ignore',invalid='ignore'):
            res = self.Qfunc(self.dist)*(np.conj(self.idc)**2/np.abs(self.idc)**2)
        res[(self.dist==0)] = 0
        return res



    def Ufunc_array(self):
        """
        Computes the U filter function on an npix^2 grid
        fieldsize: size of the grid in arcmin
        """
        res = self.Ufunc(self.dist)
        return res

    
    def interpolate_nans(self,array,interpolation_method,fill_value):
        """
        method to interpolate nans. adapted from
        https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
        """
        print("starting interpolation preparations")
        x = np.arange(0, array.shape[1])
        y = np.arange(0, array.shape[0])
        #mask invalid values
        array = np.ma.masked_invalid(array)
        xx, yy = np.meshgrid(x, y)
        #get only the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]
        
        print("interpolating")
        GD1 = griddata((x1, y1), newarr.ravel(),
                                  (xx[array.mask], yy[array.mask]),
                                     method=interpolation_method, 
                      fill_value=fill_value)
        print("done")
        array[array.mask] = GD1
        #'cubic' interpolation would probalby be better, but appears to be extremely slow
        return array
    
    def filter_nans_astropy(self,array):
        filter_radius = 0.1*self.theta_ap * self.npix/self.fieldsize
#         print(filter_radius)
        kernel = Gaussian2DKernel(x_stddev=filter_radius)
        filtered_array_real = interpolate_replace_nans(array.real,kernel,convolve=convolve_fft,allow_huge=True)
        filtered_array_imag = interpolate_replace_nans(array.imag,kernel,convolve=convolve_fft,allow_huge=True)

        return filtered_array_real + 1.0j*filtered_array_imag
    
        
    def filter_nans_gaussian(self,array):
        filter_radius = 0.1*self.theta_ap * self.npix/self.fieldsize
#         print(filter_radius)
        mask = np.isnan(array)
        array[mask] = 0
        
        # fill an array with ones wherever there is data
        normalisation = np.ones(array.shape)
        normalisation[mask] = 0
        filtered_array_real = gaussian_filter(array.real,filter_radius)
        filtered_array_imag = gaussian_filter(array.imag,filter_radius)

        filtered_normalisation = gaussian_filter(normalisation,filter_radius)
        
        result = (filtered_array_real + 1.0j*filtered_array_imag)/filtered_normalisation
        
        array[mask] = result[mask]
        

        return array


    def Map_fft_from_kappa(self,kappa_arr):
        if self.u_arr is None:
            self.u_arr = self.Ufunc_array()
       
        return fftconvolve(kappa_arr,self.u_arr,'same')*self.fieldsize**2/self.npix**2

    def Map_fft(self,gamma_arr,norm=None,return_mcross=False,normalize_weighted=True, periodic_boundary=True):
        """
        Computes the signal-to-noise of an aperture mass map
        input:
            gamma_arr: npix^2 grid with sum of ellipticities of galaxies as (complex) pixel values
            norm: if None, assumes that the gamma_arr is a field with <\gamma_t> as pixel values
                    if Scalar, uses the Estimator of Bartelmann & Schneider (2001) with n as the scalar
                    if array, computes the mean number density within an aperture and uses this for n in
                                the Bartelmann & Schneider (2001) Estimator
            return_mcross: bool -- if true, also computes the cross-aperture map and returns it
            periodic_boundary: bool (default:False) 
                               if true: computes FFT with astropy's convolve_fft without zero-padding, 
                               if false: computes FFT with scipy's fftconvolve which uses zero padding 
        output:
            result: resulting aperture mass map and, if return_mcross, the cross aperture map
        this uses Map(theta) = - int d^2 theta' gamma(theta) Q(|theta'-theta|)conj(theta'-theta)^2/abs(theta'-theta)^2
        """  
 
        yr = gamma_arr.real
        yi = gamma_arr.imag
        qr = self.q_arr.real
        qi = self.q_arr.imag
        if periodic_boundary:
            rr=apc.convolve_fft(yr, qr, boundary='wrap', normalize_kernel=False, nan_treatment='fill',allow_huge=True)
            ii=apc.convolve_fft(yi, qi, boundary='wrap', normalize_kernel=False, nan_treatment='fill',allow_huge=True)
        else:
            rr = fftconvolve(yr,qr,'same')
            ii = fftconvolve(yi,qi,'same')
        
        result = (ii-rr)
        if(np.any(np.isnan(result))):
            print("ERROR! NAN in aperture mass computation!")
        if(return_mcross):
            if periodic_boundary:
                ri = apc.convolve_fft(yr, qi, boundary='wrap', normalize_kernel=False, nan_treatment='fill',allow_huge=True)
                ir = apc.convolve_fft(yi, qr,  boundary='wrap', normalize_kernel=False, nan_treatment='fill',allow_huge=True)
            else:
                ri = fftconvolve(yr,qi,'same')
                ir = fftconvolve(yi,qr,'same')
            mcross = (-ri -ir)

        if norm is None:
            result *= self.fieldsize**2/self.npix**2
            if(return_mcross):
                mcross *= self.fieldsize**2/self.npix**2
                return result,mcross
            return result
        
        if(normalize_weighted):
            if not norm.shape==gamma_arr.shape:
                print("Error! Wrong norm format")
                return None
            norm_weight = self.norm_fft(norm)
            result /= (norm_weight)
            if(return_mcross):
                mcross /= (norm_weight)
                return result,mcross
            return result
        
       
        elif isinstance(norm, (collections.Sequence, np.ndarray)):
            mean_number_within_aperture = fftconvolve(norm,self.disk,'same')
            mean_number_density_within_aperture = mean_number_within_aperture/(np.pi*self.theta_ap**2)
            result /= mean_number_density_within_aperture
            if(return_mcross):
                mcross /= mean_number_density_within_aperture
                return result,mcross
            return result
        
        else:
            result *= self.fieldsize**2 / norm
            if(return_mcross):
                mcross *= self.fieldsize**2 / norm
                return result,mcross
            return result
        

    def norm_fft(self,norm):
        q = np.abs(self.q_arr)
        result = fftconvolve(norm,q,'same')
        return result

        
    def normalize_shear(self,Xs,Ys,shears,CIC=True,normalize=False,nan_treatment=None,fill_value=0,debug=False):
        """
        distributes a galaxy catalogue on a pixel grid
        input:
            Xs: x-positions (arcmin)
            Ys: y-positions (arcmin)
            shears: measured shear_1 + 1.0j * measured shear_2
            CIC: perform a cloud-in-cell interpolation
            debug: output different stages of the CIC interpolation
        output:
            zahler_arr: npix^2 grid of sum of galaxy ellipticities
        """
        npix = self.npix
        fieldsize = self.fieldsize
        if not CIC:
            shears_grid_real = np.histogram2d(Xs,Ys,bins=np.arange(npix+1)/npix*fieldsize,weights=shears.real)[0]
            shears_grid_imag = np.histogram2d(Xs,Ys,bins=np.arange(npix+1)/npix*fieldsize,weights=shears.imag)[0]
            norm = np.histogram2d(Xs,Ys,bins=np.arange(npix+1)/npix*fieldsize)[0]



        else:
            cell_size = fieldsize/(npix-1)


            index_x = np.floor(Xs/cell_size)
            index_y = np.floor(Ys/cell_size)

            difference_x = (Xs/cell_size-index_x)
            difference_y = (Ys/cell_size-index_y)

            hist_bins = np.arange(npix+1)/(npix-1)*(fieldsize)        

            # lower left
            shears_grid_real = np.histogram2d(Xs,Ys,bins=hist_bins,
                                              weights=shears.real*(1-difference_x)*(1-difference_y))[0]
            shears_grid_imag = np.histogram2d(Xs,Ys,bins=hist_bins,
                                              weights=shears.imag*(1-difference_x)*(1-difference_y))[0]
            norm = np.histogram2d(Xs,Ys,bins=hist_bins,
                                 weights=(1-difference_x)*(1-difference_y))[0]

            # lower right
            shears_grid_real += np.histogram2d(Xs+cell_size,Ys,bins=hist_bins,
                                              weights=shears.real*(difference_x)*(1-difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs+cell_size,Ys,bins=hist_bins,
                                              weights=shears.imag*(difference_x)*(1-difference_y))[0]
            norm += np.histogram2d(Xs+cell_size,Ys,bins=hist_bins,
                                  weights=(difference_x)*(1-difference_y))[0]

            # upper left
            shears_grid_real += np.histogram2d(Xs,Ys+cell_size,bins=hist_bins,
                                              weights=shears.real*(1-difference_x)*(difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs,Ys+cell_size,bins=hist_bins,
                                              weights=shears.imag*(1-difference_x)*(difference_y))[0]
            norm += np.histogram2d(Xs,Ys+cell_size,bins=hist_bins,
                                  weights=(1-difference_x)*(difference_y))[0]

            # upper right
            shears_grid_real += np.histogram2d(Xs+cell_size,Ys+cell_size,bins=hist_bins,
                                              weights=shears.real*(difference_x)*(difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs+cell_size,Ys+cell_size,bins=hist_bins,
                                              weights=shears.imag*(difference_x)*(difference_y))[0]
            norm += np.histogram2d(Xs+cell_size,Ys+cell_size,bins=hist_bins,
                                  weights=(difference_x)*(difference_y))[0]




        result = (shears_grid_real + 1.0j*shears_grid_imag)

        if not normalize:
            return result,norm
        
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                result /= norm

        #treat the nans
        if(nan_treatment in ['linear','cubic','nearest']):
            result = self.interpolate_nans(result,nan_treatment,fill_value)
        elif (nan_treatment=='fill'):
            result[np.isnan(result)] = fill_value
        elif (nan_treatment=='gaussian'):
            result = self.filter_nans_gaussian(result)
        elif (nan_treatment=='astropy'):
            result = self.filter_nans_astropy(result)

        return result

    def compute_aperture_mass(self,galaxy_catalogue,return_mcross=False):
        """
        Computes the signal-to-noise of an aperture mass map from a galaxy catalogue
        Galaxy catalogue has the following form:
            (nx4)-array with n as number of galaxies
            0th column: X-position in arcmin
            1st column: Y-position in arcmin
            2nd column: gamma_1
            3rd column: gamma_2
        """
        Xs = galaxy_catalogue[:,0]
        Ys = galaxy_catalogue[:,1]
        gamma_1 = galaxy_catalogue[:,2]
        gamma_2 = galaxy_catalogue[:,3]
        shears = self.normalize_shear(Xs,Ys,gamma_1,gamma_2)
        Map_arr = self.Map_fft(shears,return_mcross)
        return Map_arr

class bispectrum_extractor:
    def __init__(self,field,fieldsize = (4*np.pi/180)):
        # pixsize = fieldsize/field.shape[0]
        idx,idy = np.indices(field.shape)
        idx = idx - idx.shape[0]/2
        idy = idy - idy.shape[1]/2
        dist = np.sqrt(idx**2+idy**2)*2*np.pi/fieldsize
        self.dist=dist
        self.field = np.copy(field)
        self.fftfield = np.fft.fftshift(np.fft.fft2(field))
        self.fieldshape = field.shape
        self.prefactor = fieldsize**4/field.shape[0]**6
    
    def new_field(self,field):
        self.field = np.copy(field)
        self.fftfield = np.fft.fftshift(np.fft.fft2(field))

    def extract_bispectrum(self,k1,k2,k3,delta_k1 = 0.13, delta_k2 = 0.13, delta_k3 = 0.13):
        dk1 = delta_k1*k1
        dk2 = delta_k2*k2
        dk3 = delta_k3*k3
        mask_k1 = (self.dist > k1-dk1/2) & (self.dist < k1 + dk1/2)
        mask_k2 = (self.dist > k2-dk2/2) & (self.dist < k2 + dk2/2)
        mask_k3 = (self.dist > k3-dk3/2) & (self.dist < k3 + dk3/2)

        fftones_k1 = np.zeros(self.fieldshape)
        fftones_k2 = np.zeros(self.fieldshape)
        fftones_k3 = np.zeros(self.fieldshape)

        fftones_k1[mask_k1] = 1
        fftones_k2[mask_k2] = 1
        fftones_k3[mask_k3] = 1
        
        fftfield_k1 = np.zeros(self.fieldshape,dtype=complex)
        fftfield_k2 = np.zeros(self.fieldshape,dtype=complex)
        fftfield_k3 = np.zeros(self.fieldshape,dtype=complex)


        fftfield_k1[mask_k1] = self.fftfield[mask_k1]
        fftfield_k2[mask_k2] = self.fftfield[mask_k2]
        fftfield_k3[mask_k3] = self.fftfield[mask_k3]

        field_k1 = np.fft.ifft2(np.fft.ifftshift(fftfield_k1))
        field_k2 = np.fft.ifft2(np.fft.ifftshift(fftfield_k2))
        field_k3 = np.fft.ifft2(np.fft.ifftshift(fftfield_k3))

        ones_k1 = np.fft.ifft2(np.fft.ifftshift(fftones_k1))
        ones_k2 = np.fft.ifft2(np.fft.ifftshift(fftones_k2))
        ones_k3 = np.fft.ifft2(np.fft.ifftshift(fftones_k3))

        return self.prefactor*np.sum(field_k1*field_k2*field_k3)/np.sum(ones_k1*ones_k2*ones_k3)

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
#     Abins *= (np.pi*(kbins[1:]**2-kbins[:-1]**2))
    return Abins*pixel_size/fieldsize**2,kvals

def is_triangle(l1,l2,l3):
    if(np.abs(l1-l2)>l3 or l1+l2<l3):
        return False
    if(np.abs(l2-l3)>l1 or l2+l3<l1):
        return False
    if(np.abs(l3-l1)>l2 or l3+l1<l2):
        return False
    return True

def extract_aperture_masses(Xs,Ys,shear_catalogue,npix,thetas,fieldsize,compute_mcross=False,save_map=None,same_fieldsize_for_all_theta=True,use_polynomial_filter=False):
    ac = aperture_mass_computer(npix,1.,fieldsize,use_polynomial_filter=use_polynomial_filter)
    shears,norm = ac.normalize_shear(Xs,Ys,shear_catalogue)
    result = extract_aperture_masses_of_field(shears,npix,thetas,fieldsize,norm=norm,ac=ac,compute_mcross=compute_mcross,
    save_map=save_map,same_fieldsize_for_all_theta=same_fieldsize_for_all_theta,use_polynomial_filter=use_polynomial_filter)
    return result

def extract_both_aperture_masses(Xs,Ys,shear_catalogue,npix,thetas,fieldsize,compute_mcross=False,save_map=None,same_fieldsize_for_all_theta=True,use_polynomial_filter=False):
    ac = aperture_mass_computer(npix,1.,fieldsize,use_polynomial_filter=use_polynomial_filter)
    shears,norm = ac.normalize_shear(Xs,Ys,shear_catalogue)
    result_3pt = extract_aperture_masses_of_field(shears,npix,thetas,fieldsize,norm=norm,ac=ac,compute_mcross=compute_mcross,
    save_map=save_map,same_fieldsize_for_all_theta=same_fieldsize_for_all_theta,use_polynomial_filter=use_polynomial_filter)
    result_2pt = extract_second_order_aperture_masses_of_field(shears,npix,thetas,fieldsize,norm=norm,ac=ac,compute_mcross=compute_mcross,
    save_map=save_map,same_fieldsize_for_all_theta=same_fieldsize_for_all_theta,use_polynomial_filter=use_polynomial_filter)

    return result_2pt,result_3pt

def extract_both_aperture_masses_of_field(shears,npix,thetas,fieldsize,compute_mcross=False,norm=None,save_map=None,same_fieldsize_for_all_theta=True,use_polynomial_filter=False):
    ac = aperture_mass_computer(npix,1.,fieldsize,use_polynomial_filter=use_polynomial_filter)
    result_3pt = extract_aperture_masses_of_field(shears,npix,thetas,fieldsize,norm=norm,ac=ac,compute_mcross=compute_mcross,
    save_map=save_map,same_fieldsize_for_all_theta=same_fieldsize_for_all_theta,use_polynomial_filter=use_polynomial_filter)
    result_2pt = extract_second_order_aperture_masses_of_field(shears,npix,thetas,fieldsize,norm=norm,ac=ac,compute_mcross=compute_mcross,
    save_map=save_map,same_fieldsize_for_all_theta=same_fieldsize_for_all_theta,use_polynomial_filter=use_polynomial_filter)

    return result_2pt,result_3pt


def extract_second_order_aperture_masses(Xs,Ys,shear_catalogue,npix,thetas,fieldsize,compute_mcross=False,save_map=None,same_fieldsize_for_all_theta=True,use_polynomial_filter=False):
    ac = aperture_mass_computer(npix,1.,fieldsize,use_polynomial_filter=use_polynomial_filter)
    shears,norm = ac.normalize_shear(Xs,Ys,shear_catalogue)
    result = extract_second_order_aperture_masses_of_field(shears,npix,thetas,fieldsize,norm=norm,ac=ac,compute_mcross=compute_mcross,
    save_map=save_map,same_fieldsize_for_all_theta=same_fieldsize_for_all_theta,use_polynomial_filter=use_polynomial_filter)
    return result

def extract_second_order_aperture_masses_of_field(shears,npix,thetas,fieldsize,norm=None,ac=None,compute_mcross=False,
    save_map=None,same_fieldsize_for_all_theta=True,use_polynomial_filter=False):
    n_thetas = len(thetas)
    maxtheta = np.max(thetas)
    if ac is None:
        ac = aperture_mass_computer(npix,1.,fieldsize,use_polynomial_filter=use_polynomial_filter)

 
    if(compute_mcross):
        # result = np.zeros((n_thetas,n_thetas,n_thetas,8))
        return
    else:
        result = np.zeros(n_thetas)

    aperture_mass_fields = np.zeros((npix,npix,n_thetas))
    if(compute_mcross):
        cross_aperture_fields = np.zeros((npix,npix,n_thetas))

    for x,theta in enumerate(thetas):
        ac.change_theta_ap(theta)
        if(compute_mcross):
            map,mx = ac.Map_fft(shears,norm=norm,return_mcross=True,periodic_boundary=False)
            cross_aperture_fields[:,:,x] = mx
        else:
            map = ac.Map_fft(shears,norm=norm,return_mcross=False,periodic_boundary=False)

        aperture_mass_fields[:,:,x] = map

    if(save_map is not None):
        np.save(save_map,aperture_mass_fields)

    for i in range(n_thetas):
        field1 = aperture_mass_fields[:,:,i]

        if not same_fieldsize_for_all_theta:
            maxtheta = thetas[i]
        
        if(use_polynomial_filter):
            factor_cutoff = 1. #polynomial filter is zero outside of theta_ap
        else:
            factor_cutoff = 4. #exponential filter has 99.8 percent of its power within 4*theta_ap

        index_maxtheta = int(np.round(maxtheta/(fieldsize)*npix*factor_cutoff)) #cut off boundaries
        
        field1_cut = field1[index_maxtheta:(npix-index_maxtheta),index_maxtheta:(npix-index_maxtheta)]
        result[i] = np.mean(field1_cut**2)

    return result


def extract_aperture_masses_of_field(shears,npix,thetas,fieldsize,norm=None,ac=None,compute_mcross=False,
    save_map=None,same_fieldsize_for_all_theta=True,use_polynomial_filter=False):
    n_thetas = len(thetas)
    maxtheta = np.max(thetas)
    if ac is None:
        ac = aperture_mass_computer(npix,1.,fieldsize,use_polynomial_filter=use_polynomial_filter)

 
    if(compute_mcross):
        # result = np.zeros((n_thetas,n_thetas,n_thetas,8))
        return
    else:
        result = np.zeros(n_thetas*(n_thetas+1)*(n_thetas+2)//6)

    aperture_mass_fields = np.zeros((npix,npix,n_thetas))
    if(compute_mcross):
        cross_aperture_fields = np.zeros((npix,npix,n_thetas))

    for x,theta in enumerate(thetas):
        ac.change_theta_ap(theta)
        if(compute_mcross):
            map,mx = ac.Map_fft(shears,norm=norm,return_mcross=True,periodic_boundary=False)
            cross_aperture_fields[:,:,x] = mx
        else:
            map = ac.Map_fft(shears,norm=norm,return_mcross=False,periodic_boundary=False)

        aperture_mass_fields[:,:,x] = map

    if(save_map is not None):
        np.save(save_map,aperture_mass_fields)

    counter = 0
    for i in range(n_thetas):
        field1 = aperture_mass_fields[:,:,i]
        if(compute_mcross):
            error1 = cross_aperture_fields[:,:,i]
        for j in range(i,n_thetas):
            field2 = aperture_mass_fields[:,:,j]
            if(compute_mcross):
                error2 = cross_aperture_fields[:,:,j]
            for k in range(j,n_thetas):                     
                field3 = aperture_mass_fields[:,:,k]
                if(compute_mcross):
                    error3 = cross_aperture_fields[:,:,k]

                if not same_fieldsize_for_all_theta:
                    maxtheta = thetas[k]
                
                if(use_polynomial_filter):
                    factor_cutoff = 1. #polynomial filter is zero outside of theta_ap
                else:
                    factor_cutoff = 4. #exponential filter has 99.8 percent of its power within 4*theta_ap

                index_maxtheta = int(np.round(maxtheta/(fieldsize)*npix*factor_cutoff)) #cut off boundaries
                
                field1_cut = field1[index_maxtheta:(npix-index_maxtheta),index_maxtheta:(npix-index_maxtheta)]
                field2_cut = field2[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                field3_cut = field3[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                if(compute_mcross):
                    error1_cut = error1[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                    error2_cut = error2[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                    error3_cut = error3[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]



                if(compute_mcross):
                    result[i,j,k,0] = np.mean(field1_cut*field2_cut*field3_cut)
                    result[i,j,k,1] = np.mean(field1_cut*field2_cut*error3_cut)
                    result[i,j,k,2] = np.mean(field1_cut*error2_cut*field3_cut)
                    result[i,j,k,3] = np.mean(error1_cut*field2_cut*field3_cut)
                    result[i,j,k,4] = np.mean(error1_cut*error2_cut*field3_cut)
                    result[i,j,k,5] = np.mean(error1_cut*field2_cut*error3_cut)
                    result[i,j,k,6] = np.mean(field1_cut*error2_cut*error3_cut)
                    result[i,j,k,7] = np.mean(error1_cut*error2_cut*error3_cut)
                else:
                    result[counter] = np.mean(field1_cut*field2_cut*field3_cut)
                counter += 1

    return result