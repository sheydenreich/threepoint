"""
Functions useful for the plotting of Covariance Matrices
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def plotCov(ax, cov_values, theta1, theta2, theta3, label, color='k', ls='-'):
    """
    Adds the plot of a single covariance to ax

    Parameters
    ----------
    ax : axis object from Matplotlib
        Plot to be shown
    cov_values : np array (dimensions: Ntheta^6 x 1)
        Values of the covariance matrix, unravelled!
    theta1: float
        First aperture radius [arcmin]
    theta2: float
        Second aperture radius [arcmin]
    theta3: float
        Third aperture radius [arcmin]
    label : string
        Label for this covariance
    color : string (optional)
        Color of plot, usual matplotlib alternatives
    ls : string (optional)
        Linestyle of plot, usual matplotlib alternatives
    """

    # Set x and y axis
    ax.set_yscale('log')
    ax.set_ylabel(r'$\mathrm{Cov}(\theta_1, \theta_2, \theta_3, \theta_4, theta_5, \theta_6)$')
    ax.set_xlabel(r'$(\theta_4, \theta_5, \theta_6)$ Bins')

    # Do Plot
    N=len(cov_values)
    ax.plot(range(N), cov_values, label=label, color=color, ls=ls)

    # Add label for theta1-theta2-theta3 bin
    at=AnchoredText(r'$\theta_1=$'+f'{theta1}'+r',$\theta_2=$'+f'{theta2}'r',$\theta_3=$'+f'{theta3}', loc='lower right')
    ax.add_artist(at)



def getCovFromMap3(fn, Nlos, Nthetas):
    """
    Reads in Map3 from np-file and returns covariance in correct form for plotCov

    Parameters
    ----------
    fn : string
        Filename of np-file with Covariance
    Nlos : int
        Number of LOS
    Nthetas : int
        Number of theta-bins
    """

    # Read in <Map3>
    mapmapmaps=np.load(fn)[:,:,:,0,:Nlos]

    # Calculate Covariance
    mapmapmaps=mapmapmaps.reshape((Nthetas*Nthetas*Nthetas, Nlos))

    mapmapmaps_mean=np.mean(mapmapmaps, axis=1, keepdims=True)
    diff=mapmapmaps-mapmapmaps_mean
    
    covariance=diff.dot(diff.T)/(Nlos-1)

    return covariance.ravel()


def readCovFromFile(fn):
    """
    Reads in Covariance from ASCII file

    Parameters
    ----------
    fn : string
        Filename of ASCII file
    """

    covariance=np.loadtxt(fn)[:,6]
    return covariance


