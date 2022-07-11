""" Useful functions
"""

import numpy as np


def D(npix = 4096,pixsize = 1.):
    """ Calculates D function in Kaiser-Squires relation for a grid

    Args:
        npix (int, optional): Number of pixels in one direction. Defaults to 4096.
        pixsize (float, optional): Length of a pixel. Defaults to 1..

    Returns:
        np.array: Grid of D-values
    """
    xs1,xs2 = np.indices((npix,npix))
    xs1 = (xs1 - npix/2)*pixsize
    xs2 = (xs2 - npix/2)*pixsize
    a = (-xs1**2+xs2**2-xs1*xs2*2.j)/((xs1**2+xs2**2)**2)
    a[(xs1**2+xs2**2==0)] = 0
    return a

def Dhat_func(npix = 4096,pixsize = 1.):
    """ Calculates D_hat function in Kaiser-Squires relation for a grid

    Args:
        npix (int, optional): Number of pixels in one direction. Defaults to 4096.
        pixsize (float, optional): Length of a pixel. Defaults to 1..

    Returns:
        np.array: Grid of D-values
    """
    xs1,xs2 = np.indices((npix,npix))
    xs1 = (xs1 - npix/2)*pixsize
    xs2 = (xs2 - npix/2)*pixsize
    with np.errstate(divide="ignore",invalid="ignore"):
        a = (xs1**2-xs2**2+2.j*xs1*xs2)/(xs1**2+xs2**2)
    a[(xs1**2+xs2**2==0)] = 0
    return a


def create_gamma_field(kappa_field,Dhat=None):
    """ Calculates Gamma Field from Kappa

    Args:
        kappa_field (np.array): Kappa grid
        Dhat (np.array, optional): Precomputed Dhat. Defaults to None, then Dhat is calculated new.

    Returns:
        np.array: Gamma grid
    """
    if Dhat is None: #Calculate Dhat if not available
        Dhat = Dhat_func(npix=kappa_field.shape[0])
    # Calculate kappa hat
    fieldhat = np.fft.fftshift(np.fft.fft2(kappa_field))
    # Calculate gamma hat
    gammahat = fieldhat*Dhat
    #Calculate Gamma
    gamma = np.fft.ifft2(np.fft.ifftshift(gammahat))
    return gamma


def is_triangle(l1,l2,l3):
    """ Check if l1, l2, and l3 form a closed triangle

    Args:
        l1 (float): sidelength
        l2 (float): sidelength
        l3 (float): sidelength

    Returns:
        bool: True if l1, l2, l3 form triangle, False otherwise
    """
    if(np.abs(l1-l2)>l3 or l1+l2<l3):
        return False
    if(np.abs(l2-l3)>l1 or l2+l3<l1):
        return False
    if(np.abs(l3-l1)>l2 or l3+l1<l2):
        return False
    return True

def create_triangle(r1,r2,r3,offset = [0,0],yscale = 1.):
    x1 = np.array([0,0])
    x2 = np.array([r1,0])
    y = (r2**2+r1**2-r3**2)/(2*r1)
    x = np.sqrt(r2**2-y**2)
    x3 = np.array([y,x])
    offset = np.array(offset)
    x1 = x1 + offset
    x2 = x2 + offset
    x3 = x3 + offset
    result = np.array([x1,x2,x3])
    result[:,1]*=yscale
    return result
