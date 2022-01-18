from file_loader import get_gamma_millennium
from utility import aperture_mass_computer,extract_aperture_masses_of_field,extract_second_order_aperture_masses_of_field
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


startpath = '/vol/euclid2/euclid2_raid2/sven/MS/'

def compute_aperture_masses_of_field(los,theta_ap_array,save_map=None,use_polynomial_filter=False):
    fieldsize = 4.*60
    npix = 4096
    field = get_gamma_millennium(los)

    result_3pt = extract_aperture_masses_of_field(field,npix,theta_ap_array,fieldsize,
    save_map=save_map,use_polynomial_filter=use_polynomial_filter)
    result_2pt = extract_second_order_aperture_masses_of_field(field,npix,theta_ap_array,fieldsize,
    save_map=save_map,use_polynomial_filter=use_polynomial_filter)

    return result_2pt,result_3pt

def compute_all_aperture_masses(all_los,savepath,aperture_masses = [1.17,2.34,4.69,9.37],n_processes = 64,use_polynomial_filter=False):
    n_files = len(all_los)
    with Pool(processes=n_processes) as p:
        # print('test')
        result = [p.apply_async(compute_aperture_masses_of_field, args=(all_los[i],aperture_masses,None,use_polynomial_filter,)) for i in range(n_files)]
        data_2pt = [p.get()[0] for p in result]
        data_3pt = [p.get()[1] for p in result]

        datavec_2pt = np.array([data_2pt[i] for i in range(len(data_2pt))])
        datavec_3pt = np.array([data_3pt[i] for i in range(len(data_3pt))])

        np.savetxt(savepath+'map_squared',datavec_2pt)
        np.savetxt(savepath+'map_cubed',datavec_3pt)

if(__name__=='__main__'):
    # print("Computing test aperture mass maps:")
    # path_kappa_dustgrain = "/vol/euclid7/euclid7_2/llinke/HOWLS/convergence_maps/DUSTGRAIN_COSMO_128/kappa_noise_0_LCDM_Om02_ks_nomask_shear.fits"


    all_los = range(64)
    # if not 'SLICS' in dirpath:
        # dir_end_path = dirpath.split('/')[-1]
    savepath = startpath + 'map2_MS_2_to_16'
    print('Writing summary statistics to ',savepath)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    compute_all_aperture_masses(all_los,savepath+'/',n_processes=64,aperture_masses = [2,4,8,16])

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


