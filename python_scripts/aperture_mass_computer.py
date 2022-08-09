"""Functions and classes for the extraction of aperture mass statistics from simulations using the FFT-method
"""

import numpy as np
from scipy.interpolate import griddata
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel, convolve_fft
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
import collections
from random_fields import create_gaussian_random_field, create_lognormal_random_field
from utility import create_gamma_field
import multiprocessing.managers
from multiprocessing import Pool
from tqdm import tqdm
from file_loader import get_gamma_millennium, get_gamma_millennium_shapenoise, get_kappa_millennium, get_millennium_downsampled_shapenoise, get_slics, get_kappa_slics
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class aperture_mass_computer:
    """
    a class handling the computation of aperture masses.
    The class can use both the exponential and the polynomial filter, but most tests were done for the exponential filter!
    initialization:
        npix: number of pixel of desired aperture mass map
        theta_ap: aperture radius of desired aperture mass map (in arcmin)
        fieldsize: fieldsize of desired aperture mass map (in arcmin)
    """

    def __init__(self, npix, theta_ap, fieldsize, use_polynomial_filter=False):
        """ Class constructor

        Args:
            npix (_type_): number of pixel of desired aperture mass map
            theta_ap (_type_): aperture radius of desired aperture mass map (in arcmin)
            fieldsize (_type_): fieldsize of desired aperture mass map (in arcmin)
            use_polynomial_filter (bool, optional): Whether polynomial filter should be used. Defaults to False, in which case the exponential filter is used
        """
        self.theta_ap = theta_ap
        self.npix = npix
        self.fieldsize = fieldsize
        self.use_polynomial_filter = use_polynomial_filter
        if(use_polynomial_filter):
            print("WARNING! Using polynomial filter!")

        # compute distances to the center in arcmin
        idx, idy = np.indices([self.npix, self.npix])
        idx = idx - ((self.npix)/2)
        idy = idy - ((self.npix)/2)

        self.idc = idx + 1.0j*idy
        self.dist = np.abs(self.idc)*self.fieldsize/self.npix

        # compute the Q filter function on a grid
        self.q_arr = self.Qfunc_array()
        self.u_arr = self.Ufunc_array()

        self.disk = np.zeros((self.npix, self.npix))
        self.disk[(self.dist < self.theta_ap)] = 1

    def change_theta_ap(self, theta_ap):
        """ Changes the aperture radius and recomputes Q and U for the pixel grid

        Args:
            theta_ap (_type_): new aperture radius [arcmin]
        """
        self.theta_ap = theta_ap
        self.q_arr = self.Qfunc_array()
        self.u_arr = self.Ufunc_array()
        self.disk = np.zeros((self.npix, self.npix))
        self.disk[(self.dist < self.theta_ap)] = 1

    def Ufunc(self, theta):
        """
        The U filter function for the aperture mass calculation from Schneider et al. (2002)
        input: theta: aperture radius in arcmin
        output: U [arcmin^-2]
        """
        xsq_half = (theta/self.theta_ap)**2/2
        small_ufunc = np.exp(-xsq_half)*(1.-xsq_half)/(2*np.pi)
        return small_ufunc/self.theta_ap**2

    def Qfunc(self, theta):
        """
        The Q filter function for the aperture mass calculation from Schneider et al. (2002)
        input: theta: aperture radius in arcmin
        output: Q [arcmin^-2]
        """
        thsq = (theta/self.theta_ap)**2
        if(self.use_polynomial_filter):
            res = 6/np.pi*thsq**2*(1.-thsq**2)
            res[(thsq > 1)] = 0
            return res/self.theta_ap**2
        else:
            res = thsq/(4*np.pi*self.theta_ap**2)*np.exp(-thsq/2)
            return res

    def Qfunc_array(self):
        """
        Computes the Q filter function on an npix^2 grid
        fieldsize: size of the grid in arcmin
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            res = self.Qfunc(self.dist)*(np.conj(self.idc)
                                         ** 2/np.abs(self.idc)**2)
        res[(self.dist == 0)] = 0
        return res

    def Ufunc_array(self):
        """
        Computes the U filter function on an npix^2 grid
        fieldsize: size of the grid in arcmin
        """
        res = self.Ufunc(self.dist)
        return res

    def interpolate_nans(self, array, interpolation_method, fill_value):
        """
        method to interpolate nans. adapted from
        https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
        This is needed if there are holes in the galaxy data
        """

        x = np.arange(0, array.shape[1])
        y = np.arange(0, array.shape[0])
        # mask invalid values
        array = np.ma.masked_invalid(array)
        xx, yy = np.meshgrid(x, y)
        # get only the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]

        GD1 = griddata((x1, y1), newarr.ravel(),
                       (xx[array.mask], yy[array.mask]),
                       method=interpolation_method,
                       fill_value=fill_value)

        array[array.mask] = GD1
        # 'cubic' interpolation would probalby be better, but appears to be extremely slow
        return array

    def filter_nans_astropy(self, array):
        """Interpolate nans using the built-in functions of astropy and a Gaussian Kernel

        Args:
            array (_type_): array with nans that need to be interpolated over
        """
        filter_radius = 0.1*self.theta_ap * self.npix/self.fieldsize

        kernel = Gaussian2DKernel(x_stddev=filter_radius)
        filtered_array_real = interpolate_replace_nans(
            array.real, kernel, convolve=convolve_fft, allow_huge=True)
        filtered_array_imag = interpolate_replace_nans(
            array.imag, kernel, convolve=convolve_fft, allow_huge=True)

        return filtered_array_real + 1.0j*filtered_array_imag

    def filter_nans_gaussian(self, array):
        """Interpolate nans using a Gaussian filter

        Args:
            array (_type_): array with nans that need to be interpolated over
        """
        filter_radius = 0.1*self.theta_ap * self.npix/self.fieldsize
        mask = np.isnan(array)
        array[mask] = 0

        # fill an array with ones wherever there is data
        normalisation = np.ones(array.shape)
        normalisation[mask] = 0
        filtered_array_real = gaussian_filter(array.real, filter_radius)
        filtered_array_imag = gaussian_filter(array.imag, filter_radius)

        filtered_normalisation = gaussian_filter(normalisation, filter_radius)

        result = (filtered_array_real + 1.0j*filtered_array_imag) / \
            filtered_normalisation

        array[mask] = result[mask]

        return array

    def Map_fft_from_kappa(self, kappa_arr):
        """ Calculates the Aperture Mass map from a kappa grid using FFT

        Args:
            kappa_arr (_type_): Kappa grid with npix^2 values

        """
        # If U is not yet calculated, calculate U
        if self.u_arr is None:
            self.u_arr = self.Ufunc_array()

        # Do the calculation, the normalisation is the pixel size
        result= fftconvolve(kappa_arr, self.u_arr, 'same')*self.fieldsize**2/self.npix**2

        return result

    def Map_fft(self, gamma_arr, norm=None, return_mcross=False, normalize_weighted=True, periodic_boundary=False):
        """
        Computes the Aperture Mass map from a Gamma-Grid
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

        if periodic_boundary:  # No Zeropadding
            rr = convolve_fft(yr, qr, boundary='wrap', normalize_kernel=False,
                              nan_treatment='fill', allow_huge=True)
            ii = convolve_fft(yi, qi, boundary='wrap', normalize_kernel=False,
                              nan_treatment='fill', allow_huge=True)
        else:  # With Zeropadding
            rr = fftconvolve(yr, qr, 'same')
            ii = fftconvolve(yi, qi, 'same')

        result = (ii-rr)

        # plt.imshow(result, norm=LogNorm())
        # plt.show()
        if(np.any(np.isnan(result))):
            print("ERROR! NAN in aperture mass computation!")
        if(return_mcross):
            if periodic_boundary:
                ri = convolve_fft(
                    yr, qi, boundary='wrap', normalize_kernel=False, nan_treatment='fill', allow_huge=True)
                ir = convolve_fft(
                    yi, qr,  boundary='wrap', normalize_kernel=False, nan_treatment='fill', allow_huge=True)
            else:
                ri = fftconvolve(yr, qi, 'same')
                ir = fftconvolve(yi, qr, 'same')
            mcross = (-ri - ir)

        if norm is None:
            result *= self.fieldsize**2/self.npix**2
            if(return_mcross):
                mcross *= self.fieldsize**2/self.npix**2
                return result, mcross
            return result

        if(normalize_weighted):
            if not norm.shape == gamma_arr.shape:
                print("Error! Wrong norm format")
                return None
            norm_weight = self.norm_fft(norm)
            result /= (norm_weight)
            if(return_mcross):
                mcross /= (norm_weight)
                return result, mcross
            return result

        elif isinstance(norm, (collections.Sequence, np.ndarray)):
            mean_number_within_aperture = fftconvolve(norm, self.disk, 'same')
            mean_number_density_within_aperture = mean_number_within_aperture / \
                (np.pi*self.theta_ap**2)
            result /= mean_number_density_within_aperture
            if(return_mcross):
                mcross /= mean_number_density_within_aperture
                return result, mcross
            return result

        else:
            result *= self.fieldsize**2 / norm
            if(return_mcross):
                mcross *= self.fieldsize**2 / norm
                return result, mcross
            return result

    def norm_fft(self, norm):
        q = np.abs(self.q_arr)
        result = fftconvolve(norm, q, 'same')
        return result

    def normalize_shear(self, Xs, Ys, shears, CIC=True, normalize=False, nan_treatment=None, fill_value=0, debug=False):
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
            shears_grid_real = np.histogram2d(Xs, Ys, bins=np.arange(
                npix+1)/npix*fieldsize, weights=shears.real)[0]
            shears_grid_imag = np.histogram2d(Xs, Ys, bins=np.arange(
                npix+1)/npix*fieldsize, weights=shears.imag)[0]
            norm = np.histogram2d(
                Xs, Ys, bins=np.arange(npix+1)/npix*fieldsize)[0]

        else:
            cell_size = fieldsize/(npix-1)

            index_x = np.floor(Xs/cell_size)
            index_y = np.floor(Ys/cell_size)

            difference_x = (Xs/cell_size-index_x)
            difference_y = (Ys/cell_size-index_y)

            hist_bins = np.arange(npix+1)/(npix-1)*(fieldsize)

            # lower left
            shears_grid_real = np.histogram2d(Xs, Ys, bins=hist_bins,
                                              weights=shears.real*(1-difference_x)*(1-difference_y))[0]
            shears_grid_imag = np.histogram2d(Xs, Ys, bins=hist_bins,
                                              weights=shears.imag*(1-difference_x)*(1-difference_y))[0]
            norm = np.histogram2d(Xs, Ys, bins=hist_bins,
                                  weights=(1-difference_x)*(1-difference_y))[0]

            # lower right
            shears_grid_real += np.histogram2d(Xs+cell_size, Ys, bins=hist_bins,
                                               weights=shears.real*(difference_x)*(1-difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs+cell_size, Ys, bins=hist_bins,
                                               weights=shears.imag*(difference_x)*(1-difference_y))[0]
            norm += np.histogram2d(Xs+cell_size, Ys, bins=hist_bins,
                                   weights=(difference_x)*(1-difference_y))[0]

            # upper left
            shears_grid_real += np.histogram2d(Xs, Ys+cell_size, bins=hist_bins,
                                               weights=shears.real*(1-difference_x)*(difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs, Ys+cell_size, bins=hist_bins,
                                               weights=shears.imag*(1-difference_x)*(difference_y))[0]
            norm += np.histogram2d(Xs, Ys+cell_size, bins=hist_bins,
                                   weights=(1-difference_x)*(difference_y))[0]

            # upper right
            shears_grid_real += np.histogram2d(Xs+cell_size, Ys+cell_size, bins=hist_bins,
                                               weights=shears.real*(difference_x)*(difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs+cell_size, Ys+cell_size, bins=hist_bins,
                                               weights=shears.imag*(difference_x)*(difference_y))[0]
            norm += np.histogram2d(Xs+cell_size, Ys+cell_size, bins=hist_bins,
                                   weights=(difference_x)*(difference_y))[0]

        result = (shears_grid_real + 1.0j*shears_grid_imag)

        if not normalize:
            return result, norm

        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                result /= norm

        # treat the nans
        if(nan_treatment in ['linear', 'cubic', 'nearest']):
            result = self.interpolate_nans(result, nan_treatment, fill_value)
        elif (nan_treatment == 'fill'):
            result[np.isnan(result)] = fill_value
        elif (nan_treatment == 'gaussian'):
            result = self.filter_nans_gaussian(result)
        elif (nan_treatment == 'astropy'):
            result = self.filter_nans_astropy(result)

        return result


def extract_Map3(Xs, Ys, shear_catalogue, npix, thetas, fieldsize, compute_mcross=False, save_map=None,
                 same_fieldsize_for_all_theta=True, use_polynomial_filter=False):
    """ Extracts <Map³> from a shear catalogue, using CIC distribution of galaxies on a grid and FFT

    Args:
        Xs (np.array, 1 x Ngal): X-positions [arcmin]
        Ys (np.array, 1 x Ngal): Y-positions [arcmin]
        shear_catalogue (np.array, 1 x Ngal (complex values)): Complex shears
        npix (int): number of pixels (along one side)
        thetas (np.array of floats): aperture radii [arcmin]
        fieldsize (float): side length of field [arcmin]
        compute_mcross (bool, optional): Whether B-modes should be calculated. Defaults to False.
        save_map (str, optional): Filename to which aperture mass maps should be saved. Defaults to None, then nothing is saved.
        same_fieldsize_for_all_theta (bool, optional): If the same amount of "border" is cut for all aperture radii. Defaults to False.
        use_polynomial_filter (bool, optional): Whether polynomial filter should be used. Defaults to False, then exponential filter is used.

    Returns:
        np.array: <Map3> for all thetas, if compute_mcross=True, also the <Mperp²Map> etc...
    """
    # Create aperture mass computer
    ac = aperture_mass_computer(
        npix, 1., fieldsize, use_polynomial_filter=use_polynomial_filter)

    # Distribute shears in grid
    shears, norm = ac.normalize_shear(Xs, Ys, shear_catalogue)

    # Do the extraction of <Map3> for the shear grid
    result = extract_Map3_of_field(shears, npix, thetas, fieldsize, norm=norm, ac=ac, compute_mcross=compute_mcross,
                                   save_map=save_map, same_fieldsize_for_all_theta=same_fieldsize_for_all_theta, use_polynomial_filter=use_polynomial_filter)
    return result


def extract_Map3_of_field(shears, npix, thetas, fieldsize, norm=None, ac=None, compute_mcross=False,
                          save_map=None, same_fieldsize_for_all_theta=False, use_polynomial_filter=False):
    """ Extracts <Map³> from a shear grid.

    Args:
        shears  (np.array, 1 x npix^2 (complex values)): Complex shears on a grid
        npix (int): number of pixels (along one side)
        thetas (np.array of floats): aperture radii [arcmin]
        fieldsize (float): side length of field [arcmin]
        norm: if None, assumes that the gamma_arr is a field with <\gamma_t> as pixel values
              if Scalar, uses the Estimator of Bartelmann & Schneider (2001) with n as the scalar
              if array, computes the mean number density within an aperture and uses this for n in
                                the Bartelmann & Schneider (2001) Estimator
        ac (aperture_mass_computer, optional): Computer for the calculation. Defaults to None, in which case a new one is initialized.
        compute_mcross (bool, optional):  Whether B-modes should be calculated. Defaults to False.
        save_map (str, optional): Filename to which aperture mass maps should be saved. Defaults to None, then nothing is saved.
        same_fieldsize_for_all_theta (bool, optional): If the same amount of "border" is cut for all aperture radii. Defaults to False.
        use_polynomial_filter (bool, optional): Whether polynomial filter should be used. Defaults to False, then exponential filter is used.

    Returns:
        np.array: <Map3> for all thetas, if compute_mcross=True, also the <Mperp²Map> etc...
    """
    n_thetas = len(thetas) #Number of aperture radii
    maxtheta = np.max(thetas) # Maximum aperture radius

    # Create a new aperture mass computer if none is given to the function
    if ac is None:
        ac = aperture_mass_computer(
            npix, 1., fieldsize, use_polynomial_filter=use_polynomial_filter)

    # Initialize results array
    if(compute_mcross):
        result = np.zeros((n_thetas,n_thetas,n_thetas,8))
    else:
        result = np.zeros(n_thetas*(n_thetas+1)*(n_thetas+2)//6)


    # Calculate the aperture mass maps for all aperture radii
    aperture_mass_fields = np.zeros((npix, npix, n_thetas))
    if(compute_mcross):
        cross_aperture_fields = np.zeros((npix, npix, n_thetas))

    for x, theta in enumerate(thetas): # Go through all aperture radii

        ac.change_theta_ap(theta)
        # Calculate aperture mass maps
        if(compute_mcross):
            map, mx = ac.Map_fft(
                shears, norm=norm, return_mcross=True, periodic_boundary=False)
            cross_aperture_fields[:, :, x] = mx
        else:
            map = ac.Map_fft(shears, norm=norm,
                             return_mcross=False, periodic_boundary=False)

        aperture_mass_fields[:, :, x] = map

    # Save aperture mass maps if wanted
    if(save_map is not None):
        np.save(save_map, aperture_mass_fields)


    # Calculate the <Map³> for all independent combinations of the aperture radii
    counter = 0
    for i in range(n_thetas):
        field1 = aperture_mass_fields[:, :, i]
        if(compute_mcross):
            error1 = cross_aperture_fields[:, :, i]
        for j in range(i, n_thetas):
            field2 = aperture_mass_fields[:, :, j]
            if(compute_mcross):
                error2 = cross_aperture_fields[:, :, j]
            for k in range(j, n_thetas):
                field3 = aperture_mass_fields[:, :, k]
                if(compute_mcross):
                    error3 = cross_aperture_fields[:, :, k]

                # Determine how much border needs to be cut-off
                if not same_fieldsize_for_all_theta:
                    maxtheta = thetas[k]

                if(use_polynomial_filter):
                    factor_cutoff = 1.  # polynomial filter is zero outside of theta_ap
                else:
                    factor_cutoff = 4.  # exponential filter has 99.8 percent of its power within 4*theta_ap

                # cut off boundaries
                index_maxtheta = int(
                    np.round(maxtheta/(fieldsize)*npix*factor_cutoff))

                field1_cut = field1[index_maxtheta:(
                    npix-index_maxtheta), index_maxtheta:(npix-index_maxtheta)]
                field2_cut = field2[index_maxtheta:npix -
                                    index_maxtheta, index_maxtheta:npix-index_maxtheta]
                field3_cut = field3[index_maxtheta:npix -
                                    index_maxtheta, index_maxtheta:npix-index_maxtheta]
                if(compute_mcross):
                    error1_cut = error1[index_maxtheta:npix -
                                        index_maxtheta, index_maxtheta:npix-index_maxtheta]
                    error2_cut = error2[index_maxtheta:npix -
                                        index_maxtheta, index_maxtheta:npix-index_maxtheta]
                    error3_cut = error3[index_maxtheta:npix -
                                        index_maxtheta, index_maxtheta:npix-index_maxtheta]
                # Calculate all B-modes and the E-mode
                if(compute_mcross):
                    result[i, j, k, 0] = np.mean(
                        field1_cut*field2_cut*field3_cut)
                    result[i, j, k, 1] = np.mean(
                        field1_cut*field2_cut*error3_cut)
                    result[i, j, k, 2] = np.mean(
                        field1_cut*error2_cut*field3_cut)
                    result[i, j, k, 3] = np.mean(
                        error1_cut*field2_cut*field3_cut)
                    result[i, j, k, 4] = np.mean(
                        error1_cut*error2_cut*field3_cut)
                    result[i, j, k, 5] = np.mean(
                        error1_cut*field2_cut*error3_cut)
                    result[i, j, k, 6] = np.mean(
                        field1_cut*error2_cut*error3_cut)
                    result[i, j, k, 7] = np.mean(
                        error1_cut*error2_cut*error3_cut)
                else:
                    result[counter] = np.mean(field1_cut*field2_cut*field3_cut)
                counter += 1

    return result


def extract_Map3_of_kappa_field(kappas, npix, thetas, fieldsize, norm=None, ac=None,
                          save_map=None, same_fieldsize_for_all_theta=True, use_polynomial_filter=False):
    """ Extracts <Map³> from a kappa grid.

    Args:
        kappa  (np.array, 1 x npix^2 ): kappa on a grid
        npix (int): number of pixels (along one side)
        thetas (np.array of floats): aperture radii [arcmin]
        fieldsize (float): side length of field [arcmin]
        norm: if None, assumes that the gamma_arr is a field with <\gamma_t> as pixel values
              if Scalar, uses the Estimator of Bartelmann & Schneider (2001) with n as the scalar
              if array, computes the mean number density within an aperture and uses this for n in
                                the Bartelmann & Schneider (2001) Estimator
        ac (aperture_mass_computer, optional): Computer for the calculation. Defaults to None, in which case a new one is initialized.
        save_map (str, optional): Filename to which aperture mass maps should be saved. Defaults to None, then nothing is saved.
        same_fieldsize_for_all_theta (bool, optional): If the same amount of "border" is cut for all aperture radii. Defaults to False.
        use_polynomial_filter (bool, optional): Whether polynomial filter should be used. Defaults to False, then exponential filter is used.

    Returns:
        np.array: <Map3> for all thetas, if compute_mcross=True, also the <Mperp²Map> etc...
    """
    n_thetas = len(thetas) #Number of aperture radii
    maxtheta = np.max(thetas) # Maximum aperture radius

    # Create a new aperture mass computer if none is given to the function
    if ac is None:
        ac = aperture_mass_computer(
            npix, 1., fieldsize, use_polynomial_filter=use_polynomial_filter)

    # Initialize results array
    result = np.zeros(n_thetas*(n_thetas+1)*(n_thetas+2)//6)


    # Calculate the aperture mass maps for all aperture radii
    aperture_mass_fields = np.zeros((npix, npix, n_thetas))

    for x, theta in enumerate(thetas): # Go through all aperture radii

        ac.change_theta_ap(theta)
        # Calculate aperture mass maps

        map = ac.Map_fft_from_kappa(kappas)

        aperture_mass_fields[:, :, x] = map

    # Save aperture mass maps if wanted
    if(save_map is not None):
        np.save(save_map, aperture_mass_fields)


    # Calculate the <Map³> for all independent combinations of the aperture radii
    counter = 0
    for i in range(n_thetas):
        field1 = aperture_mass_fields[:, :, i]

        for j in range(i, n_thetas):
            field2 = aperture_mass_fields[:, :, j]

            for k in range(j, n_thetas):
                field3 = aperture_mass_fields[:, :, k]


                # Determine how much border needs to be cut-off
                if not same_fieldsize_for_all_theta:
                    maxtheta = thetas[k]

                if(use_polynomial_filter):
                    factor_cutoff = 1.  # polynomial filter is zero outside of theta_ap
                else:
                    factor_cutoff = 4.  # exponential filter has 99.8 percent of its power within 4*theta_ap

                # cut off boundaries
                index_maxtheta = int(
                    np.round(maxtheta/(fieldsize)*npix*factor_cutoff))

                field1_cut = field1[index_maxtheta:(
                    npix-index_maxtheta), index_maxtheta:(npix-index_maxtheta)]
                field2_cut = field2[index_maxtheta:npix -
                                    index_maxtheta, index_maxtheta:npix-index_maxtheta]
                field3_cut = field3[index_maxtheta:npix -
                                    index_maxtheta, index_maxtheta:npix-index_maxtheta]


                result[counter] = np.mean(field1_cut*field2_cut*field3_cut)
                counter += 1

    return result



def extract_Map2(Xs, Ys, shear_catalogue, npix, thetas, fieldsize, save_map=None,
                 same_fieldsize_for_all_theta=False, use_polynomial_filter=False):
    """ Extracts <Map²> from a shear catalogue, using CIC distribution of galaxies on a grid and FFT
    Note that <Map²> contains all information for theta1=theta2, so only equal aperture radii are considered

    Args:
        Xs (np.array, 1 x Ngal): X-positions [arcmin]
        Ys (np.array, 1 x Ngal): Y-positions [arcmin]
        shear_catalogue (np.array, 1 x Ngal (complex values)): Complex shears
        npix (int): number of pixels (along one side)
        thetas (np.array of floats): aperture radii [arcmin]
        fieldsize (float): side length of field [arcmin]
        save_map (str, optional): Filename to which aperture mass maps should be saved. Defaults to None, then nothing is saved.
        same_fieldsize_for_all_theta (bool, optional): If the same amount of "border" is cut for all aperture radii. Defaults to False.
        use_polynomial_filter (bool, optional): Whether polynomial filter should be used. Defaults to False, then exponential filter is used.

    Returns:
        np.array: <Map²> for all thetas
    """
    # Create aperture mass computer
    ac = aperture_mass_computer(
        npix, 1., fieldsize, use_polynomial_filter=use_polynomial_filter)

    # Distribute shears in grid
    shears, norm = ac.normalize_shear(Xs, Ys, shear_catalogue)

    # Do the extraction of <Map2> for the shear grid
    result = extract_Map2_of_field(shears, npix, thetas, fieldsize, norm=norm, ac=ac, 
                                   save_map=save_map, same_fieldsize_for_all_theta=same_fieldsize_for_all_theta, use_polynomial_filter=use_polynomial_filter)
    return result



def extract_Map2_of_field(shears, npix, thetas, fieldsize, norm=None, ac=None, 
                          save_map=None, same_fieldsize_for_all_theta=False, use_polynomial_filter=False):
    """ Extracts <Map²> from a shear grid. 
    Note that <Map²> contains all information for theta1=theta2, so only equal aperture radii are considered


    Args:
        shears  (np.array, 1 x npix^2 (complex values)): Complex shears on a grid
        npix (int): number of pixels (along one side)
        thetas (np.array of floats): aperture radii [arcmin]
        fieldsize (float): side length of field [arcmin]
        norm: if None, assumes that the gamma_arr is a field with <\gamma_t> as pixel values
              if Scalar, uses the Estimator of Bartelmann & Schneider (2001) with n as the scalar
              if array, computes the mean number density within an aperture and uses this for n in
                                the Bartelmann & Schneider (2001) Estimator
        ac (aperture_mass_computer, optional): Computer for the calculation. Defaults to None, in which case a new one is initialized.
        save_map (str, optional): Filename to which aperture mass maps should be saved. Defaults to None, then nothing is saved.
        same_fieldsize_for_all_theta (bool, optional): If the same amount of "border" is cut for all aperture radii. Defaults to False.
        use_polynomial_filter (bool, optional): Whether polynomial filter should be used. Defaults to False, then exponential filter is used.

    Returns:
        np.array: <Map²> for all thetas
    """
    n_thetas = len(thetas) #Number of aperture radii
    maxtheta = np.max(thetas) # Maximum aperture radius

    # Create a new aperture mass computer if none is given to the function
    if ac is None:
        ac = aperture_mass_computer(
            npix, 1., fieldsize, use_polynomial_filter=use_polynomial_filter)

    # Initialize results array
    result = np.zeros(n_thetas)


    # Calculate the aperture mass maps for all aperture radii
    aperture_mass_fields = np.zeros((npix, npix, n_thetas))

    for x, theta in enumerate(thetas): # Go through all aperture radii
        ac.change_theta_ap(theta)
        # Calculate aperture mass maps
        map = ac.Map_fft(shears, norm=norm, return_mcross=False, periodic_boundary=False)

        aperture_mass_fields[:, :, x] = map

    # Save aperture mass maps if wanted
    if(save_map is not None):
        np.save(save_map, aperture_mass_fields)


    # Calculate the <Map²> for all aperture radii
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


def extract_Map2_of_kappa_field(kappas, npix, thetas, fieldsize, norm=None, ac=None,
                          save_map=None, same_fieldsize_for_all_theta=True, use_polynomial_filter=False):
    n_thetas = len(thetas) #Number of aperture radii
    maxtheta = np.max(thetas) # Maximum aperture radius

    # Create a new aperture mass computer if none is given to the function
    if ac is None:
        ac = aperture_mass_computer(
            npix, 1., fieldsize, use_polynomial_filter=use_polynomial_filter)

    # Initialize results array
    result = np.zeros(n_thetas)


    # Calculate the aperture mass maps for all aperture radii
    aperture_mass_fields = np.zeros((npix, npix, n_thetas))

    for x, theta in enumerate(thetas): # Go through all aperture radii
        ac.change_theta_ap(theta)
        # Calculate aperture mass maps
        map = ac.Map_fft_from_kappa(kappas)

        aperture_mass_fields[:, :, x] = map

    # Save aperture mass maps if wanted
    if(save_map is not None):
        np.save(save_map, aperture_mass_fields)


    # Calculate the <Map²> for all aperture radii
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

def extract_Map2_Map3(Xs, Ys, shear_catalogue, npix, thetas, fieldsize, save_map=None,
                 same_fieldsize_for_all_theta=False, use_polynomial_filter=False):

    """ Extracts <Map²> and <Map³> from a shear catalogue, using CIC distribution of galaxies on a grid and FFT
    Note that <Map²> contains all information for theta1=theta2, so only equal aperture radii are considered

    Args:
        Xs (np.array, 1 x Ngal): X-positions [arcmin]
        Ys (np.array, 1 x Ngal): Y-positions [arcmin]
        shear_catalogue (np.array, 1 x Ngal (complex values)): Complex shears
        npix (int): number of pixels (along one side)
        thetas (np.array of floats): aperture radii [arcmin]
        fieldsize (float): side length of field [arcmin]
        save_map (str, optional): Filename to which aperture mass maps should be saved. Defaults to None, then nothing is saved.
        same_fieldsize_for_all_theta (bool, optional): If the same amount of "border" is cut for all aperture radii. Defaults to False.
        use_polynomial_filter (bool, optional): Whether polynomial filter should be used. Defaults to False, then exponential filter is used.

    Returns:
        np.array: <Map²>  and <Map³> for all thetas
    """
        # Create aperture mass computer
    ac = aperture_mass_computer(
        npix, 1., fieldsize, use_polynomial_filter=use_polynomial_filter)

    # Distribute shears in grid
    shears, norm = ac.normalize_shear(Xs, Ys, shear_catalogue)

    # Do the extraction of <Map2> for the shear grid
    map2 = extract_Map2_of_field(shears, npix, thetas, fieldsize, norm=None, ac=ac, 
                          save_map=None, same_fieldsize_for_all_theta=same_fieldsize_for_all_theta, use_polynomial_filter=use_polynomial_filter)

        # Do the extraction of <Map3> for the shear grid
    map3 = extract_Map3_of_field(shears, npix, thetas, fieldsize, norm=norm, ac=ac, 
                                   save_map=save_map, same_fieldsize_for_all_theta=same_fieldsize_for_all_theta, use_polynomial_filter=use_polynomial_filter)


    return map2, map3




def extract_Map2_Map3_of_field(shears, npix, thetas, fieldsize, norm=None, ac=None, 
                          save_map=None, same_fieldsize_for_all_theta=False, use_polynomial_filter=False):
    """ Extracts <Map²>  and <Map³> from a shear grid. 
    Note that <Map²> contains all information for theta1=theta2, so only equal aperture radii are considered
    To Do: Currently calculated Aperture mass maps two times ==> This can be improved!


    Args:
        shears  (np.array, 1 x npix^2 (complex values)): Complex shears on a grid
        npix (int): number of pixels (along one side)
        thetas (np.array of floats): aperture radii [arcmin]
        fieldsize (float): side length of field [arcmin]
        norm: if None, assumes that the gamma_arr is a field with <\gamma_t> as pixel values
              if Scalar, uses the Estimator of Bartelmann & Schneider (2001) with n as the scalar
              if array, computes the mean number density within an aperture and uses this for n in
                                the Bartelmann & Schneider (2001) Estimator
        ac (aperture_mass_computer, optional): Computer for the calculation. Defaults to None, in which case a new one is initialized.
        save_map (str, optional): Filename to which aperture mass maps should be saved. Defaults to None, then nothing is saved.
        same_fieldsize_for_all_theta (bool, optional): If the same amount of "border" is cut for all aperture radii. Defaults to False.
        use_polynomial_filter (bool, optional): Whether polynomial filter should be used. Defaults to False, then exponential filter is used.
    """
    # Do the extraction of <Map2> for the shear grid
    map2 = extract_Map2_of_field(shears, npix, thetas, fieldsize, norm=norm, ac=ac, 
                                   save_map=save_map, same_fieldsize_for_all_theta=same_fieldsize_for_all_theta, use_polynomial_filter=use_polynomial_filter)

        # Do the extraction of <Map3> for the shear grid
    map3 = extract_Map3_of_field(shears, npix, thetas, fieldsize, norm=norm, ac=ac, 
                                   save_map=save_map, same_fieldsize_for_all_theta=same_fieldsize_for_all_theta, use_polynomial_filter=use_polynomial_filter)
    return map2, map3


class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

def Map3_Gaussian_Random_Field(power_spectrum, thetas, npix=4096, fieldsize=240, random_seed=None, cutOutFromBiggerField=False, subtract_mean=False):
    """ Creates one Gaussian Random Field and extracts Map³ from it

    Args:
        power_spectrum (function): Powerspectrum P(ell)
        thetas (np.array of floats): aperture radii [arcmin]
        npix (int, optional): Number of pixels along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [arcmin]. Defaults to 240.
        random_seed (int, optional): Random Seed. Defaults to None, then random seed is taken.
        cutOutFromBiggerField (bool, optional): Whether Gaussian Field should be cut out as center of a bigger field. Defaults to False.
        subtract_mean (bool, optional): Whether mean of kappa field should be subtracted. Defaults to False.

    Returns:
        Map3 for all thetas and a single Random field
    """

    fieldsize_rad=fieldsize/60*np.pi/180
    if cutOutFromBiggerField:
        kappa_field = create_gaussian_random_field(power_spectrum,n_pix=5*npix,fieldsize=5*fieldsize_rad,random_seed=random_seed)
        kappa_field = kappa_field[2*npix:3*npix, 2*npix:3*npix]
    else:
        kappa_field = create_gaussian_random_field(power_spectrum,n_pix=npix,fieldsize=fieldsize_rad,random_seed=random_seed)

    if subtract_mean:
        kappa_field=kappa_field-np.mean(kappa_field)


    #shears = create_gamma_field(kappa_field)
    
    #result = extract_Map3_of_field(shears, npix, thetas, fieldsize)

    result=extract_Map3_of_kappa_field(kappa_field, npix, thetas, fieldsize)
    return result



def Map3_Gaussian_Random_Field_kernel(kwargs):
    """ Kernel function for Map3_Gaussian_Random_Field

    Args:
        kwargs (list): all arguments for Map3_Gaussian_Random_Field
    """
    final_results, power_spectrum,thetas, npix, fieldsize, random_seed, cutOutFromBiggerField, realisation = kwargs
    result = Map3_Gaussian_Random_Field(power_spectrum,thetas, npix, fieldsize, random_seed, cutOutFromBiggerField)
    final_results[:,realisation] = result

def Map3_Gaussian_Random_Field_parallelised(power_spectrum, thetas, npix, fieldsize, n_realisations=256, n_processes=64, cutOutFromBiggerField=False):
    """ Parallelised calculation of all Map3 for the Gaussian Random Fields

    Args:
        power_spectrum (function): Powerspectrum P(ell)
        thetas (np.array of floats): aperture radii [arcmin]
        npix (int, optional): Number of pixels along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [arcmin]. Defaults to 240.
        n_realisations (int, optional): Number of random realisations. Defaults to 256.
        n_processes (int, optional): Number of parallel processes. Defaults to 64.
        cutOutFromBiggerField (bool, optional): Whether Gaussian Field should be cut out as center of a bigger field. Defaults to False.

    Returns:
        Map3 for all thetas and all realisations
    """
    m=MyManager()
    m.start()
    n_thetas = len(thetas)
  
    final_results= m.np_zeros((n_thetas*(n_thetas+1)*(n_thetas+2)//6,n_realisations))


    with Pool(processes=n_processes) as p:
        args = [[final_results, power_spectrum, thetas, npix, fieldsize,(i**3+250*i)%2**32, cutOutFromBiggerField,i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(Map3_Gaussian_Random_Field_kernel,args),total=n_realisations):
            pass

    return final_results


def Map3_Lognormal_Random_Field(power_spectrum, alpha, thetas, npix=4096, fieldsize=240, random_seed=None, cutOutFromBiggerField=False, subtract_mean=False):
    """ Creates one Lognormal Random Field and extracts Map³ from it

    Args:
        power_spectrum (function): Powerspectrum P(ell)
        alpha: Non-Gaussianity parameter (see Hilbert+ 2012 for a definition)
        thetas (np.array of floats): aperture radii [arcmin]
        npix (int, optional): Number of pixels along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [arcmin]. Defaults to 240.
        random_seed (int, optional): Random Seed. Defaults to None, then random seed is taken.
        cutOutFromBiggerField (bool, optional): Whether Gaussian Field should be cut out as center of a bigger field. Defaults to False.
        subtract_mean (bool, optional): Whether mean of kappa field should be subtracted. Defaults to False.

    Returns:
        Map3 for all thetas and a single Random field
    """

    fieldsize_rad=fieldsize/60*np.pi/180
    if cutOutFromBiggerField:
        kappa_field = create_lognormal_random_field(power_spectrum, alpha,n_pix=5*npix,fieldsize=5*fieldsize_rad,random_seed=random_seed)
        kappa_field = kappa_field[2*npix:3*npix, 2*npix:3*npix]
    else:
        kappa_field = create_lognormal_random_field(power_spectrum, alpha, n_pix=npix,fieldsize=fieldsize_rad,random_seed=random_seed)

    if subtract_mean:
        kappa_field=kappa_field-np.mean(kappa_field)


    shears = create_gamma_field(kappa_field)

    result = extract_Map3_of_field(shears, npix, thetas, fieldsize)

    return result



def Map3_Lognormal_Random_Field_kernel(kwargs):
    """ Kernel function for Map3_Lognormal_Random_Field

    Args:
        kwargs (list): all arguments for Map3_Lognormal_Random_Field
    """
    final_results, power_spectrum, alpha, thetas, npix, fieldsize, random_seed, cutOutFromBiggerField, realisation = kwargs
    result = Map3_Lognormal_Random_Field(power_spectrum, alpha, thetas, npix, fieldsize, random_seed, cutOutFromBiggerField)
    final_results[:,realisation] = result



def Map3_Lognormal_Random_Field_parallelised(power_spectrum, alpha, thetas, npix, fieldsize, n_realisations=256, n_processes=64, cutOutFromBiggerField=False):
    """ Parallelised calculation of all Map3 for the Gaussian Random Fields

    Args:
        power_spectrum (function): Powerspectrum P(ell)
        alpha: Non-Gaussianity parameter (see Hilbert+ 2012 for a definition)
        thetas (np.array of floats): aperture radii [arcmin]
        npix (int, optional): Number of pixels along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [arcmin]. Defaults to 240.
        n_realisations (int, optional): Number of random realisations. Defaults to 256.
        n_processes (int, optional): Number of parallel processes. Defaults to 64.
        cutOutFromBiggerField (bool, optional): Whether Gaussian Field should be cut out as center of a bigger field. Defaults to False.

    Returns:
        Map3 for all thetas and all realisations
    """
    m=MyManager()
    m.start()
    n_thetas = len(thetas)
  
    final_results= m.np_zeros((n_thetas*(n_thetas+1)*(n_thetas+2)//6,n_realisations))


    with Pool(processes=n_processes) as p:
        args = [[final_results, power_spectrum, alpha, thetas, npix, fieldsize,(i**3+250*i)%2**32, cutOutFromBiggerField,i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(Map3_Lognormal_Random_Field_kernel,args),total=n_realisations):
            pass

    return final_results




def Map3_MS(los, thetas, shapenoise=None, numberdensity=None):
    """Extracts Map3 for a single LOS of the MS

    Args:
        los (int): LOS number
        thetas (np.array of floats): aperture radii [arcmin]
        shapenoise (float, optional): Shapenoise that is to be added. Defaults to None.
        numberdensity (float, optional): Source galaxy number density to be used. Defaults to None, then whole pixel grid is used.

    Returns:
        Map3 for all thetas and a single LOS
    """
    fieldsize=4*60
    npix=4096

    if (shapenoise==None) and (numberdensity==None):
        field=get_gamma_millennium(los)
        result = extract_Map3_of_field(field, npix, thetas, fieldsize)
    elif numberdensity==None:
        field=get_gamma_millennium_shapenoise(los, shapenoise)
        result = extract_Map3_of_field(field, npix, thetas, fieldsize)
    else:
        Ngal_subsample=numberdensity*fieldsize*fieldsize # Number of galaxies to be considered
        Xs, Ys, shears1, shears2 = get_millennium_downsampled_shapenoise(los, Ngal_subsample, shapenoise)
        shear=shears1+1.0j*shears2
        result = extract_Map3(Xs, Ys, shear, npix, thetas, fieldsize=fieldsize)

    return result    


def Map3_MS_kernel(kwargs):
    """ Kernel function for Map3_MS

    Args:
        kwargs (list): all arguments for Map3_MS
    """
    results, los, thetas, shapenoise, numberdensity, realisation = kwargs
    map3=Map3_MS(los, thetas, shapenoise=shapenoise, numberdensity=numberdensity)
    results[:,realisation]=map3


def Map3_MS_parallelised(all_los=range(64), thetas=[2,4,8,16], shapenoise=None, numberdensity=None, n_processes=64):
    """ Parallelised calculation of all Map3 for the MS

    Args:
        all_los (list): List of LOS numbers to be considered
        thetas (np.array of floats): aperture radii [arcmin]
        shapenoise (float, optional): Shapenoise that is to be added. Defaults to None.
        numberdensity (float, optional): Source galaxy number density to be used. Defaults to None, then whole pixel grid is used.
        n_processes (int, optional): Number of parallel processes. Defaults to 64.

    Returns:
        Map3 for all thetas and all realisations
    """
    m=MyManager()
    m.start()
    n_thetas = len(thetas)
    n_realisations=len(all_los)
  
    final_results= m.np_zeros((n_thetas*(n_thetas+1)*(n_thetas+2)//6,n_realisations))


    with Pool(processes=n_processes) as p:
        args = [[final_results, all_los[i], thetas, shapenoise, numberdensity, i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(Map3_MS_kernel,args),total=n_realisations):
            pass

    return final_results



def Map3_SLICS(los, thetas):
    """Extracts Map3 for a single LOS of the SLICS

    Args:
        los (int): LOS number
        thetas (np.array of floats): aperture radii [arcmin]

    Returns:
        Map3 for all thetas and a single LOS
    """
    fieldsize=10*60
    npix=1024

    #Xs, Ys, shears1, shears2 = get_slics(los)
    #shear=shears1+1.0j*shears2
    #result = extract_Map3(Xs, Ys, shear, npix, thetas, fieldsize=fieldsize)
    
    kappa=get_kappa_slics(los)
    result=extract_Map3_of_kappa_field(kappa, npix, thetas, fieldsize, same_fieldsize_for_all_theta=True)

    return result    


def Map3_SLICS_kernel(kwargs):
    """ Kernel function for Map3_MS

    Args:
        kwargs (list): all arguments for Map3_MS
    """
    results, los, thetas, realisation = kwargs
    map3=Map3_SLICS(los, thetas)
    results[:,realisation]=map3


def Map3_SLICS_parallelised(all_los, thetas=[2,4,8,16], n_processes=64):
    """ Parallelised calculation of all Map3 for the SLICS

    Args:
        all_los (list): List of LOS numbers to be considered
        thetas (np.array of floats): aperture radii [arcmin]
        n_processes (int, optional): Number of parallel processes. Defaults to 64.

    Returns:
        Map3 for all thetas and all realisations
    """
    m=MyManager()
    m.start()
    n_thetas = len(thetas)
    n_realisations=len(all_los)
  
    final_results= m.np_zeros((n_thetas*(n_thetas+1)*(n_thetas+2)//6,n_realisations))


    with Pool(processes=n_processes) as p:
        args = [[final_results, all_los[i], thetas,  i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(Map3_SLICS_kernel,args),total=n_realisations):
            pass

    return final_results





def Map2_Gaussian_Random_Field(power_spectrum, thetas, npix=4096, fieldsize=240, random_seed=None, cutOutFromBiggerField=False, subtract_mean=False):
    """ Creates one Gaussian Random Field and extracts Map² from it

    Args:
        power_spectrum (function): Powerspectrum P(ell)
        thetas (np.array of floats): aperture radii [arcmin]
        npix (int, optional): Number of pixels along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [arcmin]. Defaults to 240.
        random_seed (int, optional): Random Seed. Defaults to None, then random seed is taken.
        cutOutFromBiggerField (bool, optional): Whether Gaussian Field should be cut out as center of a bigger field. Defaults to False.
        subtract_mean (bool, optional): Whether mean of kappa field should be subtracted. Defaults to False.

    Returns:
        Map² for all thetas and a single Random field
    """
    fieldsize_rad=fieldsize/60*np.pi/180

    if cutOutFromBiggerField:
        kappa_field = create_gaussian_random_field(power_spectrum,n_pix=5*npix,fieldsize=5*fieldsize_rad,random_seed=random_seed)
        kappa_field = kappa_field[2*npix:3*npix, 2*npix:3*npix]
    else:
        kappa_field = create_gaussian_random_field(power_spectrum,n_pix=npix,fieldsize=fieldsize_rad,random_seed=random_seed)

    if subtract_mean:
        kappa_field=kappa_field-np.mean(kappa_field)


    shears = create_gamma_field(kappa_field)

    result = extract_Map2_of_field(shears, npix, thetas, fieldsize)

    return result



def Map2_Gaussian_Random_Field_kernel(kwargs):
    """ Kernel function for Map3_Gaussian_Random_Field

    Args:
        kwargs (list): all arguments for Map2_Gaussian_Random_Field
    """
    final_results, power_spectrum,thetas, npix, fieldsize, random_seed, cutOutFromBiggerField, realisation = kwargs
    result = Map2_Gaussian_Random_Field(power_spectrum,thetas, npix, fieldsize, random_seed, cutOutFromBiggerField)
    final_results[:,realisation] = result

def Map2_Gaussian_Random_Field_parallelised(power_spectrum, thetas, npix, fieldsize, n_realisations=256, n_processes=64, cutOutFromBiggerField=False):
    """ Parallelised calculation of all Map2 for the Gaussian Random Fields

    Args:
        power_spectrum (function): Powerspectrum P(ell)
        thetas (np.array of floats): aperture radii [arcmin]
        npix (int, optional): Number of pixels along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [arcmin]. Defaults to 240.
        n_realisations (int, optional): Number of random realisations. Defaults to 256.
        n_processes (int, optional): Number of parallel processes. Defaults to 64.
        cutOutFromBiggerField (bool, optional): Whether Gaussian Field should be cut out as center of a bigger field. Defaults to False.

    Returns:
        Map2 for all thetas and all realisations
    """
    m=MyManager()
    m.start()
    n_theta = len(thetas)
  
    final_results= m.np_zeros((n_theta,n_realisations))


    with Pool(processes=n_processes) as p:
        args = [[final_results, power_spectrum, thetas, npix, fieldsize,(i**3+250*i)%2**32, cutOutFromBiggerField,i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(Map2_Gaussian_Random_Field_kernel,args),total=n_realisations):
            pass

    return final_results


def Map2_Lognormal_Random_Field(power_spectrum, alpha, thetas, npix=4096, fieldsize=4*np.pi/180, random_seed=None, cutOutFromBiggerField=False, subtract_mean=False):
    """ Creates one Lognormal Random Field and extracts Map² from it

    Args:
        power_spectrum (function): Powerspectrum P(ell)
        alpha: Non-Gaussianity parameter (see Hilbert+ 2012 for a definition)
        thetas (np.array of floats): aperture radii [arcmin]
        npix (int, optional): Number of pixels along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [arcmin]. Defaults to 240.
        random_seed (int, optional): Random Seed. Defaults to None, then random seed is taken.
        cutOutFromBiggerField (bool, optional): Whether Gaussian Field should be cut out as center of a bigger field. Defaults to False.
        subtract_mean (bool, optional): Whether mean of kappa field should be subtracted. Defaults to False.

    Returns:
        Map² for all thetas and a single Random field
    """

    fieldsize_rad=fieldsize/60*np.pi/180
    if cutOutFromBiggerField:
        kappa_field = create_lognormal_random_field(power_spectrum, alpha,n_pix=5*npix,fieldsize=5*fieldsize_rad,random_seed=random_seed)
        kappa_field = kappa_field[2*npix:3*npix, 2*npix:3*npix]
    else:
        kappa_field = create_lognormal_random_field(power_spectrum, alpha, n_pix=npix,fieldsize=fieldsize_rad,random_seed=random_seed)

    if subtract_mean:
        kappa_field=kappa_field-np.mean(kappa_field)


    shears = create_gamma_field(kappa_field)

    result = extract_Map2_of_field(shears, npix, thetas, fieldsize)

    return result



def Map2_Lognormal_Random_Field_kernel(kwargs):
    """ Kernel function for Map2_Lognormal_Random_Field

    Args:
        kwargs (list): all arguments for Map2_Lognormal_Random_Field
    """
    final_results, power_spectrum, alpha, thetas, npix, fieldsize, random_seed, cutOutFromBiggerField, realisation = kwargs
    result = Map2_Lognormal_Random_Field(power_spectrum, alpha, thetas, npix, fieldsize, random_seed, cutOutFromBiggerField)
    final_results[:,realisation] = result



def Map2_Lognormal_Random_Field_parallelised(power_spectrum, alpha, thetas, npix, fieldsize, n_realisations=256, n_processes=64, cutOutFromBiggerField=False):
    """ Parallelised calculation of all Map2 for the Gaussian Random Fields

    Args:
        power_spectrum (function): Powerspectrum P(ell)
        alpha: Non-Gaussianity parameter (see Hilbert+ 2012 for a definition)
        thetas (np.array of floats): aperture radii [arcmin]
        npix (int, optional): Number of pixels along one side. Defaults to 4096.
        fieldsize (float, optional): Sidelength [arcmin]. Defaults to 240.
        n_realisations (int, optional): Number of random realisations. Defaults to 256.
        n_processes (int, optional): Number of parallel processes. Defaults to 64.
        cutOutFromBiggerField (bool, optional): Whether Gaussian Field should be cut out as center of a bigger field. Defaults to False.

    Returns:
        Map2 for all thetas and all realisations
    """
    m=MyManager()
    m.start()
    n_theta = len(thetas)
  
    final_results= m.np_zeros((n_theta,n_realisations))


    with Pool(processes=n_processes) as p:
        args = [[final_results, power_spectrum, alpha, thetas, npix, fieldsize,(i**3+250*i)%2**32, cutOutFromBiggerField,i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(Map2_Lognormal_Random_Field_kernel,args),total=n_realisations):
            pass

    return final_results




def Map2_MS(los, thetas, shapenoise=None, numberdensity=None):
    """Extracts Map2 for a single LOS of the MS

    Args:
        los (int): LOS number
        thetas (np.array of floats): aperture radii [arcmin]
        shapenoise (float, optional): Shapenoise that is to be added. Defaults to None.
        numberdensity (float, optional): Source galaxy number density to be used. Defaults to None, then whole pixel grid is used.

    Returns:
        Map2 for all thetas and a single LOS
    """
    fieldsize=4*60
    npix=4096

    if (shapenoise==None) and (numberdensity==None):
        # field=get_gamma_millennium(los)
        # result = extract_Map2_of_field(field, npix, thetas, fieldsize)
        kappa = get_kappa_millennium(los)
        result = extract_Map2_of_kappa_field(kappa, npix, thetas, fieldsize, same_fieldsize_for_all_theta=True)
    elif numberdensity==None:
        field=get_gamma_millennium_shapenoise(los, shapenoise)
        result = extract_Map2_of_field(field, npix, thetas, fieldsize)
    else:
        Ngal_subsample=numberdensity*fieldsize*fieldsize
        Xs, Ys, shears1, shears2 = get_millennium_downsampled_shapenoise(los, Ngal_subsample, shapenoise)
        shear=shears1+1.0j*shears2
        result = extract_Map2(Xs, Ys, shear, npix, thetas, fieldsize=fieldsize)

    

    return result    


def Map2_MS_kernel(kwargs):
    """ Kernel function for Map2_MS

    Args:
        kwargs (list): all arguments for Map2_MS
    """
    results, los, thetas, shapenoise, numberdensity, realisation = kwargs
    Map2=Map2_MS(los, thetas, shapenoise=shapenoise, numberdensity=numberdensity)
    results[:,realisation]=Map2


def Map2_MS_parallelised(all_los=range(64), thetas=[2,4,8,16], shapenoise=None, numberdensity=None, n_processes=64):
    """ Parallelised calculation of all Map2 for the MS

    Args:
        all_los (list): List of LOS numbers to be considered
        thetas (np.array of floats): aperture radii [arcmin]
        shapenoise (float, optional): Shapenoise that is to be added. Defaults to None.
        numberdensity (float, optional): Source galaxy number density to be used. Defaults to None, then whole pixel grid is used.
        n_processes (int, optional): Number of parallel processes. Defaults to 64.

    Returns:
        Map2 for all thetas and all realisations
    """
    m=MyManager()
    m.start()
    n_theta = len(thetas)
    n_realisations=len(all_los)
  
    final_results= m.np_zeros((n_theta,n_realisations))


    with Pool(processes=n_processes) as p:
        args = [[final_results, all_los[i], thetas, shapenoise, numberdensity, i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(Map2_MS_kernel,args),total=n_realisations):
            pass

    return final_results



def Map2_SLICS(los, thetas):
    """Extracts Map2 for a single LOS of the SLICS

    Args:
        los (int): LOS number
        thetas (np.array of floats): aperture radii [arcmin]

    Returns:
        Map2 for all thetas and a single LOS
    """
    fieldsize=10*60
    npix=1024

        
    kappa=get_kappa_slics(los)
    result=extract_Map2_of_kappa_field(kappa, npix, thetas, fieldsize, same_fieldsize_for_all_theta=True)


    return result    


def Map2_SLICS_kernel(kwargs):
    """ Kernel function for Map2_MS

    Args:
        kwargs (list): all arguments for Map2_MS
    """
    results, los, thetas, realisation = kwargs
    Map2=Map2_SLICS(los, thetas)
    results[:,realisation]=Map2


def Map2_SLICS_parallelised(all_los, thetas=[2,4,8,16], n_processes=64):
    """ Parallelised calculation of all Map2 for the SLICS

    Args:
        all_los (list): List of LOS numbers to be considered
        thetas (np.array of floats): aperture radii [arcmin]
        n_processes (int, optional): Number of parallel processes. Defaults to 64.

    Returns:
        Map2 for all thetas and all realisations
    """
    m=MyManager()
    m.start()
    n_theta = len(thetas)
    n_realisations=len(all_los)
  
    final_results= m.np_zeros((n_theta,n_realisations))


    with Pool(processes=n_processes) as p:
        args = [[final_results, all_los[i], thetas,  i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(Map2_SLICS_kernel,args),total=n_realisations):
            pass

    return final_results








def Map2Map3_MS(los, thetas, shapenoise=None, numberdensity=None):
    fieldsize=4*60
    npix=4096

    if (shapenoise==None) and (numberdensity==None):
        field=get_gamma_millennium(los)
        Map2, Map3 = extract_Map2_Map3_of_field(field, npix, thetas, fieldsize)
    elif numberdensity==None:
        field=get_gamma_millennium_shapenoise(los, shapenoise)
        Map2, Map3 = extract_Map2_Map3_of_field(field, npix, thetas, fieldsize)
    else:
        Ngal_subsample=numberdensity*fieldsize*fieldsize
        Xs, Ys, shears1, shears2 = get_millennium_downsampled_shapenoise(los, Ngal_subsample, shapenoise)
        shear=shears1+1.0j*shears2
        Map2, Map3 = extract_Map2_Map3(Xs, Ys, shear, npix, thetas, fieldsize=fieldsize)

    return Map2, Map3  


def Map2Map3_MS_kernel(kwargs):
    resultsMap2, resultsMap3, los, thetas, shapenoise, numberdensity, realisation = kwargs
    Map2, Map3=Map2Map3_MS(los, thetas, shapenoise=shapenoise, numberdensity=numberdensity)
    resultsMap2[:,realisation]=Map2
    resultsMap3[:,realisation]=Map3


def Map2Map3_MS_parallelised(all_los=range(64), thetas=[2,4,8,16], shapenoise=None, numberdensity=None, n_processes=64):
    m=MyManager()
    m.start()
    n_thetas = len(thetas)
    n_realisations=len(all_los)
  
    final_results_Map2= m.np_zeros((n_thetas,n_realisations))
    final_results_Map3= m.np_zeros((n_thetas*(n_thetas+1)*(n_thetas+2)//6, n_realisations))



    with Pool(processes=n_processes) as p:
        args = [[final_results_Map2, final_results_Map3, all_los[i], thetas, shapenoise, numberdensity, i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(Map2Map3_MS_kernel,args),total=n_realisations):
            pass

    return final_results_Map2, final_results_Map3






