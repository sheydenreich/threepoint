from utility import create_gaussian_random_field, create_gamma_field, extract_power_spectrum, bispectrum_extractor, is_triangle
import numpy as np
import sys
from tqdm import tqdm
import multiprocessing.managers
from multiprocessing import Pool

global_fieldsize_deg = 10.
n_pix = 4096

if(len(sys.argv)<2):
    print("Usage: python3 compute_aperture_mass_correlations_of_random_fields.py [powerspectrum_type] [npix] [fieldsize].")
    print("powerspectrum_type: 0: constant, 1: (x/1e4)^2*exp(-(x/1e4)^2), 2:(x/1e4)*exp(-(x/1e4))")
    print("npix: default 4096")
    print("fieldsize [deg]: default 10")
    sys.exit()

CONSTANT_POWERSPECTRUM = False
ANALYTICAL_POWERSPECTRUM = False
ANALYTICAL_POWERSPECTRUM_V2 = False

if(int(sys.argv[1])==0):
    print("Using constant powerspectrum")
    CONSTANT_POWERSPECTRUM = True

if(int(sys.argv[1])==1):
    print("Using (x/1e4)^2*exp(-(x/1e4)^2) powerspectrum")
    ANALYTICAL_POWERSPECTRUM = True

if(int(sys.argv[1])==2):
    print("Using (x/1e4)*exp(-(x/1e4)) powerspectrum")
    ANALYTICAL_POWERSPECTRUM_V2 = True

if(len(sys.argv)>2):
    n_pix = int(sys.argv[2])

if(len(sys.argv)>3):
    global_fieldsize_deg = float(sys.argv(3))

global_fieldsize_rad = global_fieldsize_deg*np.pi/180
global_fieldsize_arcmin = global_fieldsize_deg*60.

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

def spectra_extraction_kernel(kwargs):
    bispectrum_array,power_spectrum,ell_array,los = kwargs
    kappa = create_gaussian_random_field(power_spectrum,n_pix=n_pix,fieldsize=global_fieldsize_rad,random_seed=los)

    n_ell = len(ell_array)
    a = bispectrum_extractor(kappa,fieldsize=global_fieldsize_rad)
    

    for i in range(n_ell):
        ell1 = ell_array[i]
        for j in range(i,n_ell):
            ell2 = ell_array[j]
            for k in range(j,n_ell):
                ell3 = ell_array[k]
                if(is_triangle(ell1,ell2,ell3)):
                    _temp_data = a.extract_bispectrum(ell1,ell2,ell3)
                    bispectrum_array[i,j,k,los] = _temp_data
                    bispectrum_array[i,k,j,los] = _temp_data
                    bispectrum_array[j,k,i,los] = _temp_data
                    bispectrum_array[j,i,k,los] = _temp_data
                    bispectrum_array[k,i,j,los] = _temp_data
                    bispectrum_array[k,j,i,los] = _temp_data
                    # print(i,j,k,_temp_data)
    # print(bispectrum_array[10,10,10,0])
    


def extract_spectra(ell_array,power_spectrum,n_processes=64,n_los=64):
    m = MyManager()
    m.start()
    n_ell = len(ell_array)
    bispectrum_array = m.np_zeros((n_ell,n_ell,n_ell,n_los))
    # spectra_extraction_kernel([bispectrum_array,power_spectrum,ell_array,0])
    # print(bispectrum_array[10,10,10,0])
    with Pool(processes=n_processes) as p:
        args = [[bispectrum_array,power_spectrum,ell_array,i] for i in range(n_los)]
        for i in tqdm(p.imap_unordered(spectra_extraction_kernel,args),total=n_los):
            pass

    return bispectrum_array

if(__name__=='__main__'):
    ell_array = np.logspace(2,5,30)
    if(CONSTANT_POWERSPECTRUM):
        def power_spectrum(x):
            return 0.3**2/(2.*n_pix**2/global_fieldsize_rad**2)*np.ones(x.shape)
        savepath = '/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/constant_powerspectrum/'
        
    elif(ANALYTICAL_POWERSPECTRUM):
        def power_spectrum(x):
            return (x/10000)**2*np.exp(-(x/10000)**2)
        savepath = '/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/analytical_powerspectrum/'

    elif(ANALYTICAL_POWERSPECTRUM_V2):
        def power_spectrum(x):
            return x/10000*np.exp(-x/10000)
        savepath = '/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/analytical_powerspectrum_v2/'

    bispec = extract_spectra(ell_array,power_spectrum,n_los=256)
    np.save(savepath+'bispectrum_measured',bispec)
    # np.savetxt('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/powerspec_MR_data.dat',power_spec)
