#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:36:01 2019

@author: pierre
"""
import numpy as np
import sys
from astropy.io import fits
from astropy.table import Table
import healpy as hp
import os



#os.environ['OMP_NUM_THREADS'] = '6'
#os.system('export OMP_NUM_THREADS=100')
#os.system('sudo swapoff -a; sudo swapon -a')

nside=2**1
npix =hp.nside2npix(nside)
patch_field=np.zeros(npix)
#patch_indices = np.array([0,2,6,7,10,11,13,17,22,23,26,27,29,33,38,39,42,43,44,46])
patch_indices = np.arange(0,npix)
patch_field[patch_indices]=patch_indices+1
patch_field_highres = hp.ud_grade(patch_field,nside_out=2**12)
patch_pixel = []
for i in patch_indices:
    patch_pixel.append(np.where(patch_field_highres==i+1)[0])
patch_pixel = np.array(patch_pixel)

nside=2**12
npix =hp.nside2npix(nside)
pixel_area_deg = hp.pixelfunc.nside2pixarea(nside=nside,degrees=True)
pixel_area_arcmin = hp.pixelfunc.nside2pixarea(nside=nside)*(180*60/np.pi)**2
pixel_area_inrad = hp.pixelfunc.nside2pixarea(nside=nside)

sigma_epsilon = 0.265 # Note we do not include sqrt(2)
ngal_per_arcmin = 6.17 


def Ufunc(theta,theta_ap):
    """
    The U filter function for the aperture mass calculation from Schneider et al. (2002)
    input: theta: aperture radius in arcmin
    """
    xsq_half = (theta/theta_ap)**2/2
    small_ufunc = np.exp(-xsq_half)*(1.-xsq_half)/(2*np.pi)
    return small_ufunc/theta_ap**2

seed = 0
for los in np.arange(0,108):
    kappa_noise_free = hp.read_map('/vol/euclidraid4/data/pburger/Takahashi/lensing_maps/kappa_KiDS1000_'+str(los)+'.fits')
    print('los: ',los)
    for seed in range(1):
        print('seed: ',seed)
        
        np.random.seed(seed*108+los)
        gaussian_noise = np.random.normal(0,sigma_epsilon/np.sqrt(pixel_area_arcmin*ngal_per_arcmin),kappa_noise_free.shape[0])
        kappa = kappa_noise_free + gaussian_noise

        maps_full = {}

        for theta_ap in [2,4,8,16,32,64]:
            print(theta_ap)
            b = np.linspace(0,np.radians(500),100000)
            bw = Ufunc(theta=b,theta_ap=np.radians(theta_ap/60))
            beam_U = hp.beam2bl(bw, b, nside*3)
            Map = hp.smoothing(kappa, beam_window=beam_U, verbose=False)
            for i in range(len(patch_indices)):
                maps_full[str(theta_ap)+'_patch'+str(patch_indices[i])]=Map[patch_pixel[i]]

 
        all_theta_ap = ['2','4','8','16','32','64']
        n_theta = len(all_theta_ap)

        for i in range(len(patch_indices)):

            all_mapmapmaps = np.zeros(n_theta*(n_theta+1)*(n_theta+2)//6+6)

            counter = 0
            for x1 in range(n_theta):
                theta_1 = all_theta_ap[x1]+'_patch'+str(patch_indices[i])
                mapmapmap = np.mean(maps_full[theta_1]**2)
                all_mapmapmaps[counter] = mapmapmap
                counter += 1

            counter = 0
            for x1 in range(n_theta):
                theta_1 = all_theta_ap[x1]+'_patch'+str(patch_indices[i])
                for x2 in range(x1,n_theta):
                    theta_2 = all_theta_ap[x2]+'_patch'+str(patch_indices[i])
                    for x3 in range(x2,n_theta):
                        theta_3 = all_theta_ap[x3]+'_patch'+str(patch_indices[i])
                        mapmapmap = np.mean(maps_full[theta_1]*maps_full[theta_2]*maps_full[theta_3])
                        all_mapmapmaps[counter+6] = mapmapmap
                        counter += 1


            np.save('outputs/Map_moments_KiDS1000/Map_moment_KiDS1000_patch'+str(patch_indices[i])+'_los'+str(los)+'_seed'+str(seed),all_mapmapmaps)
        




