from utility import is_triangle
from polyspectra_extraction import extract_power_spectrum, bispectrum_extractor
import multiprocessing
from multiprocessing import Pool
import numpy as np
from file_loader import get_kappa_millennium,get_kappa_slics
from tqdm import tqdm
from lenstools import ConvergenceMap
import astropy.units as u
import os

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def MS_spectra_extraction_kernel(kwargs):
    bispectrum_array,power_spectrum_array,ell_array,los2,method = kwargs
    los = los2-74
    try:
        kappa = get_kappa_slics(los2)[0]
    except Exception as e:
        print("Error in los ",los,": ",e)
        return 0
    savepath = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_SLICS/single_bispectra/bispec_slics_los_"+str(los)
    savepath_PS = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_SLICS/single_bispectra/powerspec_slics_los_"+str(los)

    if os.path.exists(savepath+'.npy'):
        print(savepath,"already exists.")
        bispectrum_array[:,:,:,los] = np.load(savepath+'.npy')
        power_spectrum_array[:,los] = np.load(savepath_PS+'.npy')

        return 0
    if(method=='myEstimator'):
        n_ell = len(ell_array)
        a = bispectrum_extractor(kappa,fieldsize=10.*np.pi/180)
        
        if power_spectrum_array is not None:
            power_spectrum_array[:,los] = extract_power_spectrum(kappa,10.*np.pi/180,bins = ell_array)[0]
            np.save(savepath_PS,power_spectrum_array[:,los])


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
        np.save(savepath,bispectrum_array[:,:,:,los])


    if(method=='lenstools'):
        convmap = ConvergenceMap(kappa,angle=10.*u.deg)
        bispectrum_array[:,los] = convmap.bispectrum(ell_array)[1]
        power_spectrum_array[:,los] = convmap.powerSpectrum(ell_array)[1]


def extract_SLICS_spectra(ell_array,n_processes=64,los_array = range(74,1101),method='myEstimator'):
    m = MyManager()
    m.start()
    n_ell = len(ell_array)
    n_los = len(los_array)
    if(method=='myEstimator'):
        bispectrum_array = m.np_zeros((n_ell,n_ell,n_ell,n_los))
        power_spectrum_array = m.np_zeros((n_ell-1,n_los))
    if(method=='lenstools'):
        bispectrum_array = m.np_zeros((n_ell-1,n_los))
        power_spectrum_array = m.np_zeros((n_ell-1,n_los))

    with Pool(processes=n_processes) as p:
        args = [[bispectrum_array,power_spectrum_array,ell_array,i,method] for i in los_array]
        for i in tqdm(p.imap_unordered(MS_spectra_extraction_kernel,args),total=n_los):
            pass

    return bispectrum_array,power_spectrum_array



if(__name__=='__main__'):
    ell_array = np.logspace(1.5,4,30)
    # bispec,power_spec = extract_MS_spectra(ell_array,64)
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/bispec_MR_data.dat',bispec)
    # np.savetxt('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/powerspec_MR_data.dat',power_spec)
    bispec,power_spec = extract_SLICS_spectra(ell_array,128)
    np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_SLICS/bispec_SLICS_data',bispec)
    np.savetxt('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_SLICS/powerspec_SLICS_data.dat',power_spec)
