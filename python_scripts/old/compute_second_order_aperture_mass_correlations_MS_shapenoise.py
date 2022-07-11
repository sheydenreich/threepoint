from matplotlib import use
from file_loader import get_gamma_millennium_shapenoise
from utility import aperture_mass_computer,extract_second_order_aperture_masses_of_field
import numpy as np
import sys
from tqdm import tqdm
import multiprocessing.managers
from multiprocessing import Pool
from astropy.io import fits
import os

shapenoise=1

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

process_parallel=False

startpath = '/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/MS/' # '/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/MS/'

def compute_aperture_masses_of_field(los,theta_ap_array,save_map=None,use_polynomial_filter=False):
    fieldsize = 4.*60
    npix = 4096
    field = get_gamma_millennium_shapenoise(los, shapenoise)

    result = extract_second_order_aperture_masses_of_field(field,npix,theta_ap_array,fieldsize,compute_mcross=False,save_map=save_map,use_polynomial_filter=use_polynomial_filter)

    return result


def compute_aperture_masses_of_field_kernel(kwargs):
    result, los, theta_ap_array, save_map, use_polynomial_filter, realisation = kwargs
    map2=compute_aperture_masses_of_field(los, theta_ap_array, save_map=save_map, use_polynomial_filter=use_polynomial_filter)
    result[:,realisation]=map2

def compute_all_aperture_masses(all_los,savepath,aperture_masses = [1.17,2.34,4.69,9.37],n_processes = 64,use_polynomial_filter=False):
    n_files = len(all_los)
    n_thetas=len(aperture_masses)
    if(process_parallel):
        m = MyManager()
        m.start()
        results=m.np_zeros((n_thetas, n_files))

        with Pool(processes=n_processes) as p:
            args=[[results, all_los[i], aperture_masses, None, use_polynomial_filter, i] for i in range(n_files)]
            for i in tqdm(p.imap_unordered(compute_aperture_masses_of_field_kernel, args), total=n_files):
                pass
        np.savetxt(savepath+f'map_squared_sigma_{shapenoise}',results)
    else:
        for los in all_los:
            print(f"Processing {los}")
            map2=compute_aperture_masses_of_field(los, aperture_masses, save_map=None, use_polynomial_filter=use_polynomial_filter)
            np.savetxt(savepath+f"map_squared_{los}_sigma_{shapenoise}.dat", map2)

if(__name__=='__main__'):
    # print("Computing test aperture mass maps:")
    # path_kappa_dustgrain = "/vol/euclid7/euclid7_2/llinke/HOWLS/convergence_maps/DUSTGRAIN_COSMO_128/kappa_noise_0_LCDM_Om02_ks_nomask_shear.fits"


    all_los = range(64)
    # if not 'SLICS' in dirpath:
        # dir_end_path = dirpath.split('/')[-1]
    savepath = startpath + 'map_squared_our_thetas'
    print('Writing summary statistics to ',savepath)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    compute_all_aperture_masses(all_los,savepath+'/',n_processes=10,aperture_masses = [2,4,8,16])

    # for (dirpath,_,_filenames) in os.walk(startpath+"shear_catalogues/"):
    #     if(len(_filenames)>2):
    #         filenames = np.sort([filename for filename in _filenames if '.fits' in filename])
    #         # dir_end_path = dirpath.split('/')[-1]
    #         savepath = dirpath.split('shear_catalogues')[0]+'map_cubed_our_thetas'+dirpath.split('shear_catalogues')[1]
    #         print('Reading shear catalogues from ',dirpath)
    #         print('Writing summary statistics to ',savepath)
    #         if not os.path.exists(savepath):
    #             os.makedirs(savepath)

            # compute_all_aperture_masses(dirpath+'/',filenames,savepath+'/',aperture_masses = [0.5,1,2,4,8,16,32],n_processes=32)

    # for (dirpath,_,_filenames) in os.walk(startpath+"shear_catalogues/"):
    #     if(len(_filenames)>2):
    #         filenames = np.sort([filename for filename in _filenames if '.fits' in filename])
    #         # dir_end_path = dirpath.split('/')[-1]
    #         savepath = dirpath.split('shear_catalogues')[0]+'map_cubed_1_to_8_arcmin'+dirpath.split('shear_catalogues')[1]
    #         print('Reading shear catalogues from ',dirpath)
    #         print('Writing summary statistics to ',savepath)
    #         if not os.path.exists(savepath):
    #             os.makedirs(savepath)

    #         compute_all_aperture_masses(dirpath+'/',filenames,savepath+'/',aperture_masses = [1,2,4,8],n_processes=16)

    # for (dirpath,_,_filenames) in os.walk(startpath+"shear_catalogues/"):
    #     if(len(_filenames)>2):
    #         filenames = [filename for filename in _filenames if '.fits' in filename]
    #         # dir_end_path = dirpath.split('/')[-1]
    #         savepath = dirpath.split('shear_catalogues')[0]+'map_cubed_lower_resolution_intermediate_thetas'+dirpath.split('shear_catalogues')[1]
    #         print('Reading shear catalogues from ',dirpath)
    #         print('Writing summary statistics to ',savepath)
    #         if not os.path.exists(savepath):
    #             os.makedirs(savepath)

    #         compute_all_aperture_masses(dirpath+'/',filenames,savepath+'/',aperture_masses = [1.085,1.085*2,1.085*4,1.085*8],n_processes=32)


