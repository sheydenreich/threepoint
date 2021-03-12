import numpy as np
import warnings
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter
from tqdm import trange
from multiprocessing import Process, Pool
import sys
from astropy.io import fits

"""
positions of files:
millennium simulations: 
- /vol/aibn1113/data1/sven/millenium/ (can be read by using the function get_millennium)
- ascii-files, can best be accessed with np.loadtxt
- 4096x4096 grid of gamma1, gamma2 and kappa values at z=1 for 64 lines of sight spanning 4x4 deg
- can be brought into that form by np.reshape(get_millennium(los),(4096,4096,5)), then
    - 1st column: x-position
    - 2nd column: y-position
    - 3rd column: gamma1
    - 4th column: gamma2
    - 5th column: kappa

SLICS without masks(euclid-like number density and redshift distribution):
- /vol/euclid7/euclid7_2/sven/slics_lsst
- .fits files, can best be accessed with fits from astropy.io
- example: get_slics function
- dictionary galaxy catalogue, relevant entries are x_arcmin,y_arcmin,shear1 and shear2
- in total 958 lines-of-sight spanning 10x10 degree

cosmo-SLICS without masks (euclid-like number density and redshift distribution):
- /vol/euclid7/euclid7_2/sven/cosmoslics_lsst/$cosmo_$lett
- file format same as SLICS
- 26 cosmologies, 2 N-body simulations each (denoted by _a/ and _f/), and 10 semi-independent lines-of-sight per N-body simulation

SLICS and cosmo-SLICS tiled to the DES_Y1 data are on /vol/euclid7/euclid7_1, but they have a different file format, if you need them just ask :)
"""

def compute_map_millennium(los):
    ac = aperture_mass_computer(4096,1,4*60)
    data = get_millennium(los)
    Xs = data[:,0]
    Ys = data[:,1]
    gamma_1 = data[:,2]
    gamma_2 = data[:,3]
    shears = ac.normalize_shear(Xs,Ys,gamma_1,gamma_2)

    for theta_ap in [0.5,1,2,4,8,16,32]:
        ac.change_theta_ap(theta_ap)
        result = ac.Map_fft(shears,return_mcross=True)
        np.save("/vol/euclid6/euclid6_1/sven/maps_millennium/theta_"+str(theta_ap)+"_los_"+str(los),result)

def compute_map_slics(los):
    ac = aperture_mass_computer(4096,1,10*60)
    Xs,Ys,gamma_1,gamma_2 = get_slics(los)
    shears = ac.normalize_shear(Xs,Ys,gamma_1,gamma_2)

    for theta_ap in [1,2,4,8,16,32]:
        ac.change_theta_ap(theta_ap)
        result = ac.Map_fft(shears,return_mcross=True)
        np.save("/vol/euclid6/euclid6_1/sven/maps_slics/theta_"+str(theta_ap)+"_los_"+str(los),result)
    


def get_millennium(los):
    """
    returns (4096^2 , 5) array
    [:,0]: x-pos [arcmin]
    [:,1]: y-pos [arcmin]
    [:,2]: gamma1
    [:,3]: gamma2
    [:,4]: kappa
    """
    los_no1 = los//8
    los_no2 = los%8
    result = np.loadtxt("/vol/aibn1113/data1/sven/millenium/41_los_8_"+ str(los_no1) +"_"+ str(los_no2) +".ascii")
    return result

def get_slics(los):
    hdul = fits.open('/vol/euclid7/euclid7_2/sven/slics_lsst/GalCatalog_LOS_cone'+str(los)+'.fits')
    data = hdul[1].data

    Xs = data['x_arcmin']
    Ys = data['y_arcmin']
    shears1 = data['shear1']
    shears2 = data['shear2']

    return Xs,Ys,shears1,shears2

def calculate_multiprocessed(function,processnum,arglist,verbose=False,name=""):
    """
    computes function with arglist as arguments, using processnum processes
    """                    
    length = len(arglist)
    exitflag = True
    counter = 0
    while(exitflag):
        if(verbose):
            progressBar(name,counter,length)
        for i in range(processnum):
            if(counter+i<length):
                if hasattr(arglist[counter+i],'__len__'):
                    job = Process(target = function, args = arglist[counter+i])
                else:
                    job = Process(target = function, args = [arglist[counter+i]])

                job.start()
            else:
                exitflag = False
        for i in range(processnum):
            if(counter+i<length):
                job.join()
                job.terminate()
        counter = counter+processnum


class aperture_mass_computer:
    """
    a class handling the computation of aperture masses.
    initialization:
        npix: number of pixel of desired aperture mass map
        theta_ap: aperture radius of desired aperture mass map (in arcmin)
        fieldsize: fieldsize of desired aperture mass map (in arcmin)
    """
    def __init__(self,npix,theta_ap,fieldsize):
        self.theta_ap = theta_ap
        self.npix = npix
        self.fieldsize = fieldsize

        # compute distances to the center in arcmin
        idx,idy = np.indices([self.npix,self.npix])
        idx = idx - (self.npix)/2
        idy = idy - (self.npix)/2

        self.idc = idx + 1.0j*idy
        self.dist = np.abs(self.idc)*self.fieldsize/self.npix

        # compute the Q filter function on a grid
        self.q_arr = self.Qfunc_array()

    def change_theta_ap(self,theta_ap):
        self.theta_ap = theta_ap
        self.q_arr = self.Qfunc_array()

    def Qfunc(self,theta):
        """
        The Q filter function for the aperture mass calculation from Schneider et al. (2002)

        input: theta: aperture radius in arcmin
        """
        thsq = (theta/self.theta_ap)**2
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

    def Map_fft(self,gamma_arr,return_mcross=False):
        """
        Computes the signal-to-noise of an aperture mass map
        input:
            gamma_arr: npix^2 grid with sum of ellipticities of galaxies as (complex) pixel values
            return_mcross: bool -- if true, also computes the cross-aperture map and returns it

        output:
            result: resulting aperture mass map and, if return_mcross, the cross aperture map


        this uses Map(theta) = - int d^2 theta' gamma(theta) Q(|theta'-theta|)conj(theta'-theta)^2/abs(theta'-theta)^2
        """
        yr = gamma_arr.real
        yi = gamma_arr.imag
        qr = self.q_arr.real
        qi = self.q_arr.imag
        rr = fftconvolve(yr,qr,'same')
        ii = fftconvolve(yi,qi,'same')
        result = (ii-rr)*self.fieldsize**2/self.npix**2
        if(np.any(np.isnan(result))):
            print("ERROR! NAN in aperture mass computation!")
        if(return_mcross):
            ri = fftconvolve(yr,qi,'same')
            ir = fftconvolve(yi,qr,'same')
            mcross = (-ri -ir)*self.fieldsize**2/self.npix**2
            return result,mcross
        else:
            return result

    def normalize_shear(self,Xs,Ys,gamma_1,gamma_2):
        """
        distributes a galaxy catalogue on a pixel grid

        input:
            Xs: x-positions (arcmin)
            Ys: y-positions (arcmin)
            gamma_1: measured shear_1
            gamma_2: measured shear_2

        output:
            zahler_arr: npix^2 grid of sum of galaxy ellipticities
        """
        shears_arr = np.zeros((self.npix,self.npix),dtype=complex)
        
        # Xs = Xs-np.min(Xs)
        # Ys = Ys-np.min(Ys)
        shears = gamma_1 + 1.0j*gamma_2

        for i in range(len(Xs)):
            idx = int(self.npix*Xs[i]/self.fieldsize)
            idy = int(self.npix*Ys[i]/self.fieldsize)
            try:
                shears_arr[idx,idy]+=shears[i]

            except Exception as inst:
                add_galaxy = True
                if(idx<0):
                    if(idx>=-1): #account for boundary/rounding errors
                        idx=0
                    else:
                        print("Error! Galaxy out of grid!",idx,idy)
                        add_galaxy = False
                if(idx>=self.npix):
                    if(idx<self.npix+1): 
                        idx = self.npix-1
                    else:
                        print("Error! Galaxy out of grid!",idx,idy)
                        add_galaxy = False

                if(idy<0):
                    if(idy>=-1): #account for boundary/rounding errors
                        idy=0
                    else:
                        print("Error! Galaxy out of grid!",idx,idy)
                        add_galaxy = False
                if(idy>=self.npix):
                    if(idy<self.npix+1): 
                        idy = self.npix-1
                    else:
                        print("Error! Galaxy out of grid!",idx,idy)
                        add_galaxy = False

                if(add_galaxy):
                    shears_arr[idx,idy]+=shears[i]

        return shears_arr

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

def progressBar(name, value, endvalue, bar_length = 25, width = 20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent*bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r{0: <{1}} : [{2}]{3}%".format(name, width, arrow + spaces, int(round(percent*100))))
    sys.stdout.flush()
    if value == endvalue:        
         sys.stdout.write('\n\n')


if __name__ == "__main__":
    #compute map for slics
    arglist = np.arange(74,74+512)
    calculate_multiprocessed(compute_map_slics,1,arglist,True,"Computing Map for SLICS")

    
    #compute map for MS
    arglist = np.arange(64)
    calculate_multiprocessed(compute_map_millennium,64,arglist)


