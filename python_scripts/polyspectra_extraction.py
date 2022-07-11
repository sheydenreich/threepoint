"""Functions for the extraction of Polyspectra
    """

import numpy as np
from scipy import stats


class bispectrum_extractor:
    """Class for the extraction of a bispectrum from a scalar field
    """

    def __init__(self, field, fieldsize=(4*np.pi/180)):
        """Initialization

        Args:
            field (numpy array): scalar field (as 2D map)
            fieldsize (float, optional): Sidelength [rad]. Defaults to (4*np.pi/180).
        """
        # Get pixel distances
        idx, idy = np.indices(field.shape)
        idx = idx - idx.shape[0]/2
        idy = idy - idy.shape[1]/2
        dist = np.sqrt(idx**2+idy**2)*2*np.pi/fieldsize
        self.dist = dist
        # Set Field
        self.field = np.copy(field)

        # Set FFT of field
        self.fftfield = np.fft.fftshift(np.fft.fft2(field))
        self.fieldshape = field.shape
        self.prefactor = fieldsize**4/field.shape[0]**6

    def new_field(self, field):
        """Load a new field and calculates FFT.

        Args:
            field (np. array): new field
        """
        self.field = np.copy(field)
        self.fftfield = np.fft.fftshift(np.fft.fft2(field))

    def extract_bispectrum(self, k1, k2, k3, delta_k1=0.13, delta_k2=0.13, delta_k3=0.13):
        """Extract bispectrum of currently loaded field

        Args:
            k1 (_type_): _description_
            k2 (_type_): _description_
            k3 (_type_): _description_
            delta_k1 (float, optional): binsize. Defaults to 0.13.
            delta_k2 (float, optional): binsize. Defaults to 0.13.
            delta_k3 (float, optional): binsize. Defaults to 0.13.

        Returns:
           Bispectrum (k1, k2, k3)
        """
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

        fftfield_k1 = np.zeros(self.fieldshape, dtype=complex)
        fftfield_k2 = np.zeros(self.fieldshape, dtype=complex)
        fftfield_k3 = np.zeros(self.fieldshape, dtype=complex)

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


def extract_power_spectrum(field, fieldsize,
                           bins=10, linlog='log', lmin=200, lmax=10**4):
    """ Extract power spectrum P(ell) of a field

    Args:
        field (np.array): field as 2D map
        fieldsize (float): sidelength [rad]
        bins (int, optional): number of ell-bins. Defaults to 10.
        linlog (str, optional): Linear or logarithmic binning. Defaults to 'log'.
        lmin (int, optional): minimal ell. Defaults to 200.
        lmax (_type_, optional): maximal ell. Defaults to 10**4.

    Returns:
        np.array: P(ell)
    """
    n_pix = field.shape[0]
    pixel_size = (fieldsize/n_pix)**2
    fourier_image = np.fft.fftn(field)
    fourier_amplitudes = np.abs(fourier_image)**2*pixel_size
    kfreq = np.fft.fftfreq(n_pix)*2*np.pi*n_pix/fieldsize
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    if not hasattr(bins, "__len__"):
        if(linlog == 'lin'):
            kbins = np.linspace(lmin, lmax, bins+1)
            kvals = 0.5 * (kbins[1:] + kbins[:-1])

        if(linlog == 'log'):
            kbins = np.geomspace(lmin, lmax, bins+1)
            kvals = np.exp(0.5 * (np.log(kbins[1:]) + np.log(kbins[:-1])))

    else:
        kbins = bins
        kvals = 0.5 * (kbins[1:] + kbins[:-1])

    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)
    return Abins*pixel_size/fieldsize**2, kvals
