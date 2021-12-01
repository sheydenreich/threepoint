from utility import aperture_mass_computer,extract_aperture_masses
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


startpath = '/vol/euclid2/euclid2_raid2/sven/HOWLS/'


def compute_aperture_masses_of_field(filepath,theta_ap_array,save_map=None,use_polynomial_filter=False):
    slics = ('SLICS' in filepath)
    if(slics):
        fieldsize = 600.
        npix = 1024
    else:
        fieldsize = 5*60.
        npix = 512

    field = fits.open(filepath)
    data = field[1].data

    # if(slics):
    #     X_pos = data['x_arcmin']
    #     Y_pos = data['y_arcmin']
    #     shear = data['shear1']+1.0j*data['shear2']
    #     noise = data['e1_intr']+1.0j*data['e2_intr']
    #     shear_noise = (shear+noise)/(1+shear*np.conj(noise))
    # else:
    X_pos = data['ra_gal']*60.
    Y_pos = data['dec_gal']*60.
    shear_noise = -data['gamma1_noise']+1.0j*data['gamma2_noise']
    # print("Flipping e2!")
    # shear_noise = -data['gamma1_noise']-1.0j*data['gamma2_noise']

    result = extract_aperture_masses(X_pos,Y_pos,shear_noise,npix,theta_ap_array,fieldsize,compute_mcross=False,save_map=save_map,use_polynomial_filter=use_polynomial_filter)

    return result

def compute_all_aperture_masses(openpath,filenames,savepath,aperture_masses = [1.17,2.34,4.69,9.37],n_processes = 64,use_polynomial_filter=False):
    n_files = len(filenames)
    with Pool(processes=n_processes) as p:
        # print('test')
        result = [p.apply_async(compute_aperture_masses_of_field, args=(openpath+filenames[i],aperture_masses,None,use_polynomial_filter,)) for i in range(n_files)]
        data = [p.get() for p in result]
        datavec = np.array([data[i] for i in range(len(data))])
        np.savetxt(savepath+'map_cubed',datavec)


def cond_for_analysis(x):
    cond1 = '.fits' in x
    cond2 = 'cone1041' in x
    cond3 = 'cone1042' in x
    cond4 = 'cone1046' in x
    cond5 = 'cone1069' in x
    cond = cond1 and (cond2 or cond3 or cond4 or cond5)
    return cond

if(__name__=='__main__'):
    # print("Computing test aperture mass maps:")
    # path_kappa_dustgrain = "/vol/euclid7/euclid7_2/llinke/HOWLS/convergence_maps/DUSTGRAIN_COSMO_128/kappa_noise_0_LCDM_Om02_ks_nomask_shear.fits"

    for (dirpath,_,_filenames) in os.walk(startpath+"shear_catalogues/"):
        if(len(_filenames)>2 and 'SLICS' in dirpath):
            filenames = np.sort([filename for filename in _filenames if cond_for_analysis(filename)])
            # if not 'SLICS' in dirpath:
            	# dir_end_path = dirpath.split('/')[-1]
            savepath = dirpath.split('shear_catalogues')[0]+'map_cubed_flip_e2'+dirpath.split('shear_catalogues')[1]
            print('Reading shear catalogues from ',dirpath)
            print('Writing summary statistics to ',savepath)
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            compute_all_aperture_masses(dirpath+'/',filenames,savepath+'/',n_processes=32)#,aperture_masses = [0.5,1,2,4,8,16,32])

    # for (dirpath,_,_filenames) in os.walk(startpath+"shear_catalogues/"):
    #     if(len(_filenames)>2 and 'SLICS' in dirpath):
    #         filenames = np.sort([filename for filename in _filenames if '.fits' in filename])
    #         # if not 'SLICS' in dirpath:
    #         	# dir_end_path = dirpath.split('/')[-1]
    #         savepath = dirpath.split('shear_catalogues')[0]+'map_cubed_same_cutoff'+dirpath.split('shear_catalogues')[1]
    #         print('Reading shear catalogues from ',dirpath)
    #         print('Writing summary statistics to ',savepath)
    #         if not os.path.exists(savepath):
    #             os.makedirs(savepath)

    #         compute_all_aperture_masses(dirpath+'/',filenames,savepath+'/',n_processes=64)#,aperture_masses = [0.5,1,2,4,8,16,32])

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


