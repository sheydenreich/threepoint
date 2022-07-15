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
        

        X, Y = np.ogrid[0:npix, 0:npix]

        self.dist = np.hypot(X - npix/2, Y - npix/2)*fieldsize/npix
        
        self.azimuthalAngle = np.arctan2((X-npix/2),(Y-npix/2))
    
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
#     def azimuthalAverage(self,f,mask,n_bins,rmin,rmax,linlog,map_fieldsize,rotate = False):
#         sx, sy = f.shape
#         X, Y = np.ogrid[0:sx, 0:sy]
        
# #         map_fieldsize is double the value due to full correlation
#         r = np.hypot(X - sx/2, Y - sy/2)*map_fieldsize*2/sx
        
# #         r = self.full_dist
#         azimuthalAngle = np.arctan2(X-sx/2, Y-sy/2)
    
#         if(rotate):
#             f = np.copy(f)*np.exp(-4.0j*azimuthalAngle)

#         if(linlog=='lin'):
#             rbin = (n_bins* (r-rmin)/(rmax-rmin)).astype(int)
#             bins = np.linspace(rmin,rmax,n_bins)
#         elif(linlog=='log'):
#             lrmin = np.log(rmin)
#             lrmax = np.log(rmax)
#             lr = np.log(r)
#             rbin = (n_bins*(lr-lrmin)/(lrmax-lrmin)).astype(int)
#             bins = np.geomspace(rmin,rmax,n_bins)
            
#         else:
#             raise ValueError('Invalid value for linlog!')
        
#         rbin[mask] = -1
#         radial_mean_real = ndmean(np.real(f), labels=rbin, index=np.arange(n_bins))
#         return bins,radial_mean_real

    # def calculate_2d_correlationfunction(self,field_1,norm,n_bins,rmin,rmax,field_2=None,linlog='log'):
    #     if field_2 is None:
    #         field_2 = field_1
    #     gammagammacorr = correlate(field_1,field_2,'full','fft')
    #     gammagammastarcorr = correlate(field_1,np.conj(field_2),'full','fft')
        
    #     normcorr = correlate(norm,norm,'full','fft')
    #     mask = (normcorr==0)
            
    #     bins,xi_p = self.azimuthalAverage(gammagammacorr,mask,n_bins,rmin,rmax,linlog,self.fieldsize)
    #     _,xi_m = self.azimuthalAverage(gammagammastarcorr,mask,n_bins,rmin,rmax,linlog,self.fieldsize,rotate=True)
    #     _,weight = self.azimuthalAverage(normcorr,mask,n_bins,rmin,rmax,linlog,self.fieldsize)
    #     return bins,np.real(xi_p)/weight,np.real(xi_m)/weight

    # def calculate_shear_correlation(self,Xs,Ys,shears,n_bins,rmin,rmax,weights=None,CIC=True):
    #     shear_grid,norm_grid = self.normalize_shear(Xs,Ys,shears,weights,CIC=CIC)
    #     return self.calculate_2d_correlationfunction(shear_grid,norm_grid,n_bins,rmin,rmax)
        
    # compute 3pcf of shear with FFT method
    def prepare_3pcf(self,field,normfield,rmin,rmax,n_bins,n_multipoles=10):
        m = MyManager()
        m.start()

        bin_edges = np.geomspace(rmin,rmax,n_bins+1)
        gamma0_multipoles = m.np_zeros((n_bins,n_bins,n_multipoles*2+1),dtype=complex)
        gamma0_norm = m.np_zeros((n_bins,n_bins,n_multipoles*2+1),dtype=complex)

        bin_centers = np.zeros(n_bins)

        arglist = []
        for i in trange(n_bins):
            # calculate centers of bins
            bin_centers[i] = np.sum(self.get_gn(bin_edges,i,0)*norm*self.dist)/np.sum(self.get_gn(bin_edges,i,0)*norm)
            # gn = self.get_gn(bin_edges,i,1)
            for j in range(n_bins):
                for n in range(-n_multipoles,n_multipoles+1):
                    if(i==0):
                        print(n,end="\t")
                    # create arguments
                    arglist.append([i,j,n,n_multipoles,field,normfield,bin_edges,gamma0_multipoles,gamma0_norm,self.dist,self.npix,self.azimuthalAngle])
        with Pool(processes=128) as p:
            # call function on 128 processes with progress bar
            for i in tqdm(p.imap_unordered(calculate_one_bin, arglist),total=len(arglist)):
                pass
                
        return gamma0_multipoles,gamma0_norm,bin_centers

    # compute 3pcf of convergence with FFT method
    def prepare_3pcf_scalar(self,field,normfield,rmin,rmax,n_bins,n_multipoles=10):
        m = MyManager()
        m.start()

        bin_edges = np.geomspace(rmin,rmax,n_bins+1)
        final_multipoles = m.np_zeros((n_bins,n_bins,n_multipoles),dtype=complex)
        final_norm = m.np_zeros((n_bins,n_bins,n_multipoles),dtype=complex)

        bin_centers = np.zeros(n_bins)

        arglist = []
        for i in trange(n_bins):
            bin_centers[i] = np.sum(self.get_gn(bin_edges,i,0)*norm*self.dist)/np.sum(self.get_gn(bin_edges,i,0)*norm)
            for j in range(n_bins):
                # gn = self.get_gn(bin_edges,i,1)
                for n in range(n_multipoles):
                    arglist.append([i,j,n,field,normfield,bin_edges,final_multipoles,final_norm,self.dist,self.npix,self.azimuthalAngle])
        with Pool(processes=128) as p:
            for i in tqdm(p.imap_unordered(calculate_one_bin_scalar, arglist),total=len(arglist)):
                pass
                
        return final_multipoles,final_norm,bin_centers


    # def prepare_scalar_3pcf(self,field,norm,rmin,rmax,n_bins,n_multipoles=30):
    #     for i in trange(n_bins):
    #         for n in trange(n_multipoles):
    #             part1 = correlate(self.get_gn(bin_edges,i,n),field,'same','fft')
    #             part2 = np.conjugate(part1)
    #             multipoles = 
    
    
    def get_gn(self,bin_edges,i,n):       
        gn = np.zeros((self.npix,self.npix),dtype=complex)
        gn[np.logical_and(self.dist<bin_edges[i+1],self.dist>=bin_edges[i])] = 1/(np.pi*(bin_edges[i+1]**2-bin_edges[i]**2))
    #         print(n)
        gn = gn * np.exp(1.0j*n*self.azimuthalAngle)
    #         print(gn.shape)
        return gn

# functions defined outside of class since class-functions can not be pickled for multiprocessing
def get_gn(bin_edges,i,n,dist,npix,azimuthalAngle):       
    gn = np.zeros((npix,npix),dtype=complex)
    gn[np.logical_and(dist<bin_edges[i+1],dist>=bin_edges[i])] = 1/(np.pi*(bin_edges[i+1]**2-bin_edges[i]**2))
#         print(n)
    gn = gn * np.exp(1.0j*n*azimuthalAngle)
#         print(gn.shape)
    return gn

def calculate_one_bin_scalar(args):
    i,j,n,field,normfield,bin_edges,gamma0_multipoles,gamma0_norm,dist,npix,azimuthalAngle = args
    part1 = correlatewrap(field,(get_gn(bin_edges,i,n,dist,npix,azimuthalAngle)))
    part2 = correlatewrap(field,(get_gn(bin_edges,j,-n,dist,npix,azimuthalAngle)))
    gamma0_multipoles[i,j,n] = np.mean(field*part1*part2)
    
    norm1 = correlatewrap(normfield,(get_gn(bin_edges,i,n,dist,npix,azimuthalAngle)))
    norm2 = correlatewrap(normfield,(get_gn(bin_edges,j,-n,dist,npix,azimuthalAngle)))
    gamma0_norm[i,j,n] = np.mean(normfield*norm1*norm2)

        
# def calculate_one_bin(args):
#     i,n,field,normfield,bin_edges,gamma0_multipoles,gamma0_norm,dist,npix,azimuthalAngle = args
#     part1 = correlate(field,np.conj(get_gn(bin_edges,i,n-3,dist,npix,azimuthalAngle)),'same','fft')

#     part2 = correlate(field,np.conj(get_gn(bin_edges,i,-n-3,dist,npix,azimuthalAngle)),'same','fft')
#     gamma0_multipoles[i,n] = np.mean(field*part1*part2)
    
#     norm1 = correlate(normfield,np.conj(get_gn(bin_edges,i,n,dist,npix,azimuthalAngle)),'same','fft')
#     norm2 = correlate(normfield,np.conj(get_gn(bin_edges,i,-n,dist,npix,azimuthalAngle)),'same','fft')
#     gamma0_norm[i,n] = np.mean(normfield*norm1*norm2)

def calculate_one_bin(args):
    # import warnings
    # warnings.warn("Periodic field assumed")
    i,j,n,n_multipoles,field,normfield,bin_edges,gamma0_multipoles,gamma0_norm,dist,npix,azimuthalAngle = args
    # import warnings
    # warnings.warn("Complex conjugating field")
    # field = np.conj(_field)
    part1 = correlatewrap(field,(get_gn(bin_edges,i,n-3,dist,npix,azimuthalAngle)))

    part2 = correlatewrap(field,(get_gn(bin_edges,j,-n-3,dist,npix,azimuthalAngle)))
    gamma0_multipoles[i,j,n+n_multipoles] = np.mean(field*part1*part2)
    
    norm1 = correlatewrap(normfield,(get_gn(bin_edges,i,n,dist,npix,azimuthalAngle)))
    norm2 = correlatewrap(normfield,(get_gn(bin_edges,j,-n,dist,npix,azimuthalAngle)))
    gamma0_norm[i,j,n+n_multipoles] = np.mean(normfield*norm1*norm2)


def correlatewrap(x1,x2):
    # either periodic boundary conditions (1st), or zero-padding (2nd)
    # return ((np.fft.ifftshift(np.fft.ifft2((np.fft.fft2(x1))*(np.fft.fft2(x2[::-1,::-1]))))))
    return correlate(x1,np.conj(x2),'same','fft')

# powerspectrum = np.loadtxt("/users/sven/Documents/code/results/powerspectrum_SLICS.dat")
# create Gaussian random field
# field = create_gaussian_random_field_array(powerspectrum[:,0],powerspectrum[:,1],1024,4.*np.pi/180,random_seed=5)
# very crude transformation to lognormal random field
# field = np.exp(field)
# field = (field-np.mean(field))/np.std(field)
# KS-inversion to create shear field
# shear_field = (create_gamma_field(field))




tpc = correlationfunction(4096,240.)

field = get_kappa_millennium(0)
shear_field = (create_gamma_field(field))

# create "galaxy catalogue" for treecorr
idx,idy = np.indices(field.shape)
idx = idx*240./4096
idy = idy*240./4096

# randomly select 100000 pixel
np.random.seed(5)
norm = np.zeros(4096**2)
norm[np.random.choice(4096**2,1000000)] = 1
norm = norm.reshape(4096,4096)
field = field*norm
shear_field = shear_field*norm



# compute kappa 3pcf with FFT estimator
kappa_multi = tpc.prepare_3pcf_scalar(field,norm,5,120,10)
np.save("/users/sven/kappa0_az_correct_corr_select_100000",kappa_multi[0])
np.save("/users/sven/kappa0_norm_az_correct_corr_select_100000",kappa_multi[1])
np.save("/users/sven/kappa0_bin_centers",kappa_multi[2])

# compute gamma 3pcf with FFT estimator
gamma_multi = tpc.prepare_3pcf(field,norm,5,120,10)
np.save("/users/sven/gamma0_az_correct_corr_select_100000_conj",gamma_multi[0])
np.save("/users/sven/gamma0_norm_az_correct_corr_select_100000_conj",gamma_multi[1])
np.save("/users/sven/gamma0_bin_centers_conj",gamma_multi[2])



import treecorr
from time import time

cat = treecorr.Catalog(x=idx.ravel(),y=idy.ravel(),x_units='arcmin',y_units='arcmin',
k=field.ravel(),g1 = np.real(shear_field).ravel(), g2 = np.imag(shear_field).ravel(), w=norm.ravel())

# # compute kappa 3pcf with treecorr

kkk = treecorr.KKKCorrelation(nbins=10,min_sep=5.,max_sep=120.,sep_units='arcmin',
        nubins=10,min_u=0,max_u=1,nvbins=10,min_v=0,max_v=1,verbose=0,num_threads=256)

startt = time()
print("Calculating  on {} cores.".format(256))
kkk.process(cat)
print("done in {:.1f} h on {} cores. \n Saving as".format((time()-startt)/3600,256),"gamma0_treecorr")
kkk.write("/users/sven/kappa0_treecorr_lognormal_select_100000.dat")

# # compute gamma 3pcf with treecorr

ggg = treecorr.GGGCorrelation(nbins=10,min_sep=5.,max_sep=120.,sep_units='arcmin',
        nubins=10,min_u=0,max_u=1,nvbins=10,min_v=0,max_v=1,verbose=0,num_threads=256)

startt = time()
print("Calculating  on {} cores.".format(256))
ggg.process(cat)
print("done in {:.1f} h on {} cores. \n Saving as".format((time()-startt)/3600,256),"gamma0_treecorr")
ggg.write("/users/sven/gamma0_treecorr_lognormal_select_100000.dat")

# field = fits.open("/users/sven/Downloads/GalCatalog_LOS1_new.fits")
# field_data = field[1].data
# print(field[1].columns)
# Xs = field_data['x_arcmin']
# Ys = field_data['y_arcmin']
# shears = field_data['shear1']+1.0j*field_data['shear2']

# tpc = correlationfunction(2048,10.*60)
# shear_field,norm = tpc.normalize_shear(Xs,Ys,shears)

# gamma_multi = tpc.prepare_3pcf(shear_field,norm,5,120,10)

# np.save("/users/sven/gamma0_az_correct_corr_other_angle",gamma_multi[0])
# np.save("/users/sven/gamma0_norm_az_correct_corr_other_angle",gamma_multi[1])
# np.save("/users/sven/gamma0_bin_centers",gamma_multi[2])


# import treecorr
# from time import time

# cat = treecorr.Catalog(x=Xs,y=Ys,x_units='arcmin',y_units='arcmin',
#                     g1=np.real(shears),g2=np.imag(shears))
# ggg = treecorr.GGGCorrelation(nbins=10,min_sep=5.,max_sep=120.,sep_units='arcmin',
#         nubins=10,min_u=0,max_u=1,nvbins=10,min_v=0,max_v=1,verbose=0,num_threads=256)
# startt = time()
# print("Calculating  on {} cores.".format(256))
# ggg.process(cat)
# print("done in {:.1f} h on {} cores. \n Saving as".format((time()-startt)/3600,256),"gamma0_treecorr")
# ggg.write("/users/sven/gamma0_treecorr.dat")

