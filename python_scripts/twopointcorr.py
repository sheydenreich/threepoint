import numpy as np
from scipy.signal import fftconvolve,correlate
from scipy.ndimage import mean as ndmean

class correlationfunction:
    def __init__(self,npix,fieldsize):
        self.npix = npix
        self.fieldsize = fieldsize
    
    
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

        return result,norm
    
    def azimuthalAverage(self,f,mask,n_bins,rmin,rmax,linlog,map_fieldsize,rotate = False):
        sx, sy = f.shape
        X, Y = np.ogrid[0:sx, 0:sy]

        r = np.hypot(X - sx/2, Y - sy/2)*map_fieldsize/sx
        
        if(rotate):
            azimuthalAngle = np.arctan((Y-sy/2)/(X-sx/2))
            f = np.copy(f)*np.exp(-4.0j*np.pi*azimuthalAngle)

        if(linlog=='lin'):
            rbin = (n_bins* (r-rmin)/(rmax-rmin)).astype(np.int)
            bins = np.linspace(rmin,rmax,n_bins)
        elif(linlog=='log'):
            lrmin = np.log(rmin)
            lrmax = np.log(rmax)
            lr = np.log(r)
            rbin = (n_bins*(lr-lrmin)/(lrmax-lrmin)).astype(np.int)
            bins = np.geomspace(rmin,rmax,n_bins)
            
        else:
            raise ValueError('Invalid value for linlog!')
        
        rbin[mask] = -1
        radial_mean_real = ndmean(np.real(f), labels=rbin, index=np.arange(n_bins))
        # radial_mean_imag = ndmean(np.imag(f), labels=rbin, index=np.arange(n_bins))
        
        return bins,radial_mean_real

    def calculate_2d_correlationfunction(self,field_1,norm,n_bins,rmin,rmax,field_2=None,linlog='log'):
        if field_2 is None:
            field_2 = field_1
        gammagammacorr = correlate(field_1,field_2,'full','fft')
        gammagammastarcorr = correlate(field_1,np.conj(field_2),'full','fft')
        
        normcorr = correlate(norm,norm,'full','fft')
        mask = (normcorr==0)
        with np.errstate(invalid='ignore',divide='ignore'):
            gammagammacorr
            gammagammastarcorr
            
        bins,xi_p = self.azimuthalAverage(gammagammacorr,mask,n_bins,rmin,rmax,linlog,self.fieldsize)
        _,xi_m = self.azimuthalAverage(gammagammastarcorr,mask,n_bins,rmin,rmax,linlog,self.fieldsize,rotate=True)
        _,weight = self.azimuthalAverage(normcorr,mask,n_bins,rmin,rmax,linlog,self.fieldsize)
        return bins,np.real(xi_p)/weight,np.real(xi_m)/weight

    def calculate_shear_correlation(self,Xs,Ys,shears,n_bins,rmin,rmax,weights=None):
        shear_grid,norm_grid = self.normalize_shear(Xs,Ys,shears,weights)
        return self.calculate_2d_correlationfunction(shear_grid,norm_grid,n_bins,rmin,rmax)
        
        