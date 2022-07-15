from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from file_loader import get_kappa_millennium
import treecorr
import multiprocessing.managers
from multiprocessing import Pool
from utility import create_gaussian_random_field_array,create_gamma_field
import os


from time import time

import numpy as np
from scipy.signal import fftconvolve,correlate
from scipy.ndimage import mean as ndmean
from scipy.signal import correlate2d as ndcorrelate

from tqdm import trange,tqdm

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


class correlationfunction:
    def __init__(self,npix,fieldsize):
        self.npix = npix
        self.fieldsize = fieldsize
        

        # X, Y = np.ogrid[0:npix, 0:npix]

        # self.dist = np.hypot(X - npix/2, Y - npix/2)*fieldsize/npix
        
        # self.azimuthalAngle = np.arctan2((X-npix/2),(Y-npix/2))
    
    def normalize_shear(self,Xs,Ys,shears,weights=None,CIC=True):
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
            norm = np.histogram2d(Xs,Ys,bins=np.arange(npix+1)/npix*fieldsize,weights=weights)[0]



        else:
            cell_size = fieldsize/(npix+1)


            index_x = np.floor(Xs/cell_size)
            index_y = np.floor(Ys/cell_size)
        
            difference_x = (Xs/cell_size-index_x)
            difference_y = (Ys/cell_size-index_y)
            
            hist_bins = np.arange(npix+3)/(npix+1)*(fieldsize)        

            # lower left
            shears_grid_real = np.histogram2d(Xs,Ys,bins=hist_bins,
                                              weights=shears.real*(1-difference_x)*(1-difference_y))[0]
            shears_grid_imag = np.histogram2d(Xs,Ys,bins=hist_bins,
                                              weights=shears.imag*(1-difference_x)*(1-difference_y))[0]
            if weights is None:
                norm = np.histogram2d(Xs,Ys,bins=hist_bins,
                                    weights=(1-difference_x)*(1-difference_y))[0]
            else:
                norm = np.histogram2d(Xs,Ys,bins=hist_bins,
                    weights=(1-difference_x)*(1-difference_y)*weights)[0]

            # lower right
            shears_grid_real += np.histogram2d(Xs+cell_size,Ys,bins=hist_bins,
                                              weights=shears.real*(difference_x)*(1-difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs+cell_size,Ys,bins=hist_bins,
                                              weights=shears.imag*(difference_x)*(1-difference_y))[0]
            if weights is None:
                norm += np.histogram2d(Xs+cell_size,Ys,bins=hist_bins,
                                    weights=(difference_x)*(1-difference_y))[0]
            else:
                norm += np.histogram2d(Xs+cell_size,Ys,bins=hist_bins,
                                  weights=(difference_x)*(1-difference_y)*weights)[0]


            # upper left
            shears_grid_real += np.histogram2d(Xs,Ys+cell_size,bins=hist_bins,
                                              weights=shears.real*(1-difference_x)*(difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs,Ys+cell_size,bins=hist_bins,
                                              weights=shears.imag*(1-difference_x)*(difference_y))[0]
            if weights is None:
                norm += np.histogram2d(Xs,Ys+cell_size,bins=hist_bins,
                                    weights=(1-difference_x)*(difference_y))[0]
            else:
                norm += np.histogram2d(Xs,Ys+cell_size,bins=hist_bins,
                                    weights=(1-difference_x)*(difference_y)*weights)[0]

            # upper right
            shears_grid_real += np.histogram2d(Xs+cell_size,Ys+cell_size,bins=hist_bins,
                                              weights=shears.real*(difference_x)*(difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs+cell_size,Ys+cell_size,bins=hist_bins,
                                              weights=shears.imag*(difference_x)*(difference_y))[0]
            if weights is None:
                norm += np.histogram2d(Xs+cell_size,Ys+cell_size,bins=hist_bins,
                                    weights=(difference_x)*(difference_y))[0]
            else:
                norm += np.histogram2d(Xs+cell_size,Ys+cell_size,bins=hist_bins,
                                    weights=(difference_x)*(difference_y)*weights)[0]




        result = (shears_grid_real + 1.0j*shears_grid_imag)
#         return result,norm
        return result[1:-1,1:-1],norm[1:-1,1:-1]
    ####################################################################
    # Functions for 2pcf FFT estimator. Not relevant right now. ########
    ####################################################################
    def azimuthalAverage(self,f,mask,n_bins,rmin,rmax,linlog,map_fieldsize,rotate = False):
        sx, sy = f.shape
        X, Y = np.ogrid[0:sx, 0:sy]
        
#         map_fieldsize is double the value due to full correlation
        r = np.hypot(X - sx/2, Y - sy/2)*map_fieldsize*2/sx
        
#         r = self.full_dist
        azimuthalAngle = np.arctan2(X-sx/2, Y-sy/2)
    
        if(rotate):
            f = np.copy(f)*np.exp(-4.0j*azimuthalAngle)

        if(linlog=='lin'):
            rbin = (n_bins* (r-rmin)/(rmax-rmin)).astype(int)
            bins = np.linspace(rmin,rmax,n_bins)
        elif(linlog=='log'):
            lrmin = np.log(rmin)
            lrmax = np.log(rmax)
            lr = np.log(r)
            rbin = (n_bins*(lr-lrmin)/(lrmax-lrmin)).astype(int)
            bins = np.geomspace(rmin,rmax,n_bins)
            
        else:
            raise ValueError('Invalid value for linlog!')
        
        rbin[mask] = -1
        radial_mean_real = ndmean(np.real(f), labels=rbin, index=np.arange(n_bins))
        return bins,radial_mean_real

    def calculate_2d_correlationfunction(self,field_1,norm,n_bins,rmin,rmax,field_2=None,linlog='log'):
        if field_2 is None:
            field_2 = field_1
        gammagammacorr = correlate(field_1,field_2,'full','fft')
        gammagammastarcorr = correlate(field_1,np.conj(field_2),'full','fft')
        
        normcorr = correlate(norm,norm,'full','fft')
        mask = (normcorr==0)
            
        bins,xi_p = self.azimuthalAverage(gammagammacorr,mask,n_bins,rmin,rmax,linlog,self.fieldsize)
        _,xi_m = self.azimuthalAverage(gammagammastarcorr,mask,n_bins,rmin,rmax,linlog,self.fieldsize,rotate=True)
        _,weight = self.azimuthalAverage(normcorr,mask,n_bins,rmin,rmax,linlog,self.fieldsize)
        return bins,np.real(xi_p)/weight,np.real(xi_m)/weight

    def calculate_shear_correlation(self,Xs,Ys,shears,n_bins,rmin,rmax,weights=None,CIC=True):
        shear_grid,norm_grid = self.normalize_shear(Xs,Ys,shears,weights,CIC=CIC)
        return self.calculate_2d_correlationfunction(shear_grid,norm_grid,n_bins,rmin,rmax)

powerspectrum = np.loadtxt("/users/sven/Documents/code/results/powerspectrum_SLICS.dat")
# create Gaussian random field

def calculate_shear_GRF_kernel(args):
    xip_array,xim_array,n_pix,fieldsize,n_bins,subsample,i = args
    tpc = correlationfunction(n_pix,fieldsize)
    kappa_field = create_gaussian_random_field_array(powerspectrum[:,0],powerspectrum[:,1],n_pix,fieldsize,random_seed=i)
    field = (create_gamma_field(kappa_field))
    norm = np.ones_like(field)
    
    if subsample is not None:
        mask = np.zeros(n_pix**2,dtype=bool)
        mask[np.random.choice(n_pix**2,size=subsample,replace=False)] = 1
        mask = mask.reshape(n_pix,n_pix)
        field[~mask] = 0
        norm[~mask] = 0
    bins,xip,xim = tpc.calculate_2d_correlationfunction(field,norm,n_bins,0.5,120)
    xip_array[i] = xip
    xim_array[i] = xim


def calculate_shear_GRF(n_bins,n_runs,n_processes,n_pix,fieldsize = 10.*np.pi/180,subsample = 1000000):
    m = MyManager()
    m.start()
    
    final_xip = np.zeros((n_runs,n_bins))
    final_xim = np.zeros((n_runs,n_bins))
    final_xip_subsample = np.zeros((n_runs,n_bins))
    final_xim_subsample = np.zeros((n_runs,n_bins))
        
    arglist = []
    arglist_subsample = []
    
    for i in range(n_runs):
        arglist.append([final_xip,final_xim,n_pix,fieldsize,n_bins,None,i])
        arglist_subsample.append([final_xip_subsample,final_xim_subsample,n_pix,fieldsize,n_bins,subsample,i])
        
    with Pool(processes=n_processes) as p:
        for i in tqdm(p.imap_unordered(calculate_shear_GRF_kernel, arglist),total=len(arglist)):
            pass
        
    with Pool(processes=n_processes) as p:
        for i in tqdm(p.imap_unordered(calculate_shear_GRF_kernel, arglist_subsample),total=len(arglist)):
            pass
        
    return final_xip,final_xim,final_xip_subsample,final_xim_subsample

result = calculate_shear_GRF(30,128,64,4096)
np.save("/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/subsampling_effects/subsampling_xipm",
        result)