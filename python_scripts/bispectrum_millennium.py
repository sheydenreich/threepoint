from utility import extract_power_spectrum,bispectrum_extractor,is_triangle
import multiprocessing
from multiprocessing import Pool
import numpy as np
from file_loader import get_kappa_millennium
from tqdm import tqdm

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def MS_spectra_extraction_kernel(kwargs):
    bispectrum_array,power_spectrum_array,ell_array,los = kwargs
    n_ell = len(ell_array)
    kappa = get_kappa_millennium(los)
    ell1_array,ell2_array,ell3_array = np.meshgrid(ell_array,ell_array,ell_array)
    a = bispectrum_extractor(kappa,fieldsize=4.*np.pi/180)
    
    if power_spectrum_array is not None:
        power_spectrum_array[:,los] = extract_power_spectrum(kappa,4.*np.pi/180,bins = ell_array)[0]


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

def extract_MS_spectra(ell_array,n_processes=64,n_los=64):
    m = MyManager()
    m.start()
    n_ell = len(ell_array)
    bispectrum_array = m.np_zeros((n_ell,n_ell,n_ell,n_los))
    power_spectrum_array = m.np_zeros((n_ell-1,n_los))

    with Pool(processes=n_processes) as p:
        args = [[bispectrum_array,power_spectrum_array,ell_array,i] for i in range(n_los)]
        for i in tqdm(p.imap_unordered(MS_spectra_extraction_kernel,args),total=n_los):
            pass

    return bispectrum_array,power_spectrum_array



if(__name__=='__main__'):
    ell_array = np.logspace(2,5,30)
    bispec,power_spec = extract_MS_spectra(ell_array,64)
    np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/bispec_MR_data.dat',bispec)
    np.savetxt('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/powerspec_MR_data.dat',power_spec)
