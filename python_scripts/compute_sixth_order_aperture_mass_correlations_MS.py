from file_loader import get_gamma_millennium
from utility import extract_sixth_order_aperture_masses_of_field
import numpy as np
import sys
from tqdm import tqdm
import multiprocessing.managers
from multiprocessing import Pool
from astropy.io import fits
import os

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)




def compute_aperture_masses_of_field(los,theta_ap_array,save_map=None,use_polynomial_filter=False):
    fieldsize = 4.*60
    npix = 4096
    field = get_gamma_millennium(los)

    result = extract_sixth_order_aperture_masses_of_field(field,npix,theta_ap_array,fieldsize,save_map=save_map,use_polynomial_filter=use_polynomial_filter, same_fieldsize_for_all_theta=True)

    return result

def compute_all_aperture_masses(all_los,savepath,aperture_masses = [1.17,2.34,4.69,9.37],n_processes = 64,use_polynomial_filter=False):
    n_files = len(all_los)
    with Pool(processes=n_processes) as p:
        # print('test')
        result = [p.apply_async(compute_aperture_masses_of_field, args=(all_los[i],aperture_masses,None,use_polynomial_filter,)) for i in range(n_files)]
        data = [p.get() for p in result]
        datavec = np.array([data[i] for i in range(len(data))])
        np.savetxt(savepath+'Map4_fft',datavec)

if(__name__=='__main__'):
    # print("Computing test aperture mass maps:")
    # path_kappa_dustgrain = "/vol/euclid7/euclid7_2/llinke/HOWLS/convergence_maps/DUSTGRAIN_COSMO_128/kappa_noise_0_LCDM_Om02_ks_nomask_shear.fits"


    all_los = range(64)
    # if not 'SLICS' in dirpath:
        # dir_end_path = dirpath.split('/')[-1]
    savepath = "/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/results_MR/"
    print('Writing summary statistics to ',savepath)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    compute_all_aperture_masses(all_los,savepath+'/',n_processes=1,aperture_masses = [2,4])

