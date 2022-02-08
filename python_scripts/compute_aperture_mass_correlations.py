import numpy as np
# from compute_aperture_mass import progressBar
from tqdm import trange,tqdm
from utility import extract_aperture_masses
import numpy as np
import sys
import multiprocessing.managers
from multiprocessing import Pool
from astropy.io import fits
import os
from file_loader import get_slics

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)



def compute_aperture_masses_of_field(los,theta_ap_array,save_map=None,use_polynomial_filter=False):
    savepath = "/vol/euclid2/euclid2_raid2/sven/map_cubed_slics_euclid/map_cubed_same_fieldsize/map_cubed_los_"+str(los)
    if os.path.exists(savepath+".npy"):
        print(savepath," already exists, moving on")
        return np.load(savepath+".npy")
    fieldsize = 10.*60
    npix = 4096
    try:
        Xs,Ys,shears1,shears2 = get_slics(los)
    except Exception as e:
        print("Error in line of sight ",los,": ",e)
        return None
    shears = shears1 + 1.0j*shears2
    result = extract_aperture_masses(Xs,Ys,shears,npix,theta_ap_array,fieldsize,
    compute_mcross=False,save_map=save_map,use_polynomial_filter=use_polynomial_filter,
    same_fieldsize_for_all_theta=True)
    np.save(savepath,result)
    return result

def compute_all_aperture_masses(all_los,savepath,aperture_masses = [1.17,2.34,4.69,9.37],n_processes = 64,use_polynomial_filter=False):
    n_files = len(all_los)
    with Pool(processes=n_processes) as p:
        # print('test')
        result = [p.apply_async(compute_aperture_masses_of_field, args=(all_los[i],aperture_masses,None,use_polynomial_filter,)) for i in range(n_files)]
        data = [p.get() for p in result]
        datavec = np.array([data[i] for i in range(len(data)) if data[i] is not None])
        np.savetxt(savepath+'map_cubed',datavec)

if(__name__=='__main__'):
    all_los = range(74,1100)
    n_pix = 4096
    startpath = "/vol/euclid2/euclid2_raid2/sven/map_cubed_slics_euclid/"
    savepath = startpath + 'map_cubed_same_fieldsize'
    print('Writing summary statistics to ',savepath)
    if not os.path.exists(savepath):
            os.makedirs(savepath)

    compute_all_aperture_masses(all_los,savepath+'/',n_processes=64)#,aperture_masses = [0.5,1,2,4,8,16,32])
