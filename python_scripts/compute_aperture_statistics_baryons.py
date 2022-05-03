from utility import extract_both_aperture_masses
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


startpath = '/vol/euclid2/euclid2_raid2/sven/SLICS_hydrosims/'


def compute_aperture_masses_of_field(filepath,theta_ap_array,save_map=None,use_polynomial_filter=False):
    fieldsize = 600.
    npix = 4096


    data = np.loadtxt(filepath)
    # data = field[1].data

    X_pos = data[:,0]
    Y_pos = data[:,1]

    
    shear_noise = -data[:,3]+1.0j*data[:,4]
    # print("Flipping e2!")
    # shear_noise = -data['gamma1_noise']-1.0j*data['gamma2_noise']

    result = extract_both_aperture_masses(X_pos,Y_pos,shear_noise,npix,theta_ap_array,fieldsize,save_map=save_map)

    return result

def compute_all_aperture_masses(openpath,filenames,savepath,aperture_masses = [2,4,8,16],n_processes = 64,use_polynomial_filter=False):
    n_files = len(filenames)
    with Pool(processes=n_processes) as p:
        # print('test')
        result = [p.apply_async(compute_aperture_masses_of_field, args=(openpath+filenames[i],aperture_masses,None,use_polynomial_filter,)) for i in range(n_files)]
        data_2pt = [p.get()[0] for p in result]
        data_3pt = [p.get()[1] for p in result]

        datavec_2pt = np.array([data_2pt[i] for i in range(len(data_2pt))])
        datavec_3pt = np.array([data_3pt[i] for i in range(len(data_3pt))])

        np.savetxt(savepath+'map_squared',datavec_2pt)
        np.savetxt(savepath+'map_cubed',datavec_3pt)


if(__name__=='__main__'):
    # print("Computing test aperture mass maps:")
    # path_kappa_dustgrain = "/vol/euclid7/euclid7_2/llinke/HOWLS/convergence_maps/DUSTGRAIN_COSMO_128/kappa_noise_0_LCDM_Om02_ks_nomask_shear.fits"
    
    
    # compute for DM
    savepath = startpath+'mapmapmap_2_to_16_dm/'
    _filenames = os.listdir(startpath)
    filenames = np.sort([filename for filename in _filenames if "dm.cat" in filename])
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    readnames = []
    savenames = []
    for i in range(len(filenames)):
        readnames.append(startpath+filenames[i])
        savenames.append(savepath+filenames[i].split(".")[0]+(filenames[i].split(".")[-2])[1:]+".dat")
    print("Computing {} correlation functions".format(len(readnames)))

    compute_all_aperture_masses(startpath,filenames,savepath,n_processes=20)#,aperture_masses = [0.5,1,2,4,8,16,32])


    # compute for baryons
    savepath = startpath+'mapmapmap_2_to_16_bao/'
    _filenames = os.listdir(startpath)
    filenames = np.sort([filename for filename in _filenames if "bao.cat" in filename])
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    readnames = []
    savenames = []
    for i in range(len(filenames)):
        readnames.append(startpath+filenames[i])
        savenames.append(savepath+filenames[i].split(".")[0]+(filenames[i].split(".")[-2])[1:]+".dat")
    print("Computing {} correlation functions".format(len(readnames)))

    compute_all_aperture_masses(startpath,filenames,savepath,n_processes=20)#,aperture_masses = [0.5,1,2,4,8,16,32])
