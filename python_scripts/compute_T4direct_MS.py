from file_loader import get_gamma_millennium
from utility import extract_M3_of_field
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


startpath = '/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/MS' #'/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/MS/'

def compute_M3_of_field(los,theta_ap_array, rs, save_map=None,use_polynomial_filter=False):
    fieldsize = 4.*60
    npix = 4096
    print("Reading in LOS")
    field = get_gamma_millennium(los)
    print("Finished reading los", los)
    result, weight = extract_M3_of_field(field,npix,theta_ap_array,fieldsize, rs,save_map=save_map,use_polynomial_filter=use_polynomial_filter)

    return result/weight

def compute_all_M3(all_los,savepath,rs, aperture_masses = [1.17,2.34,4.69,9.37],n_processes = 64,use_polynomial_filter=False):
    n_files = len(all_los)
    with Pool(processes=n_processes) as p:
        # print('test')
        result = [p.apply_async(compute_M3_of_field, args=(all_los[i],aperture_masses,rs, None,use_polynomial_filter,)) for i in range(n_files)]
        data = [p.get() for p in result]
        datavec = np.array([data[i] for i in range(len(data))])
        np.save(savepath+'M3',datavec)

if(__name__=='__main__'):
    # print("Computing test aperture mass maps:")
    # path_kappa_dustgrain = "/vol/euclid7/euclid7_2/llinke/HOWLS/convergence_maps/DUSTGRAIN_COSMO_128/kappa_noise_0_LCDM_Om02_ks_nomask_shear.fits"

    fieldsize = 4.*60
    all_los = range(64)
    thetas=np.array([2,4,8,16])
    rs=np.linspace(0, fieldsize, num=100)
    n_thetas=len(thetas)
    n_rs=len(rs)
    delta_r=np.max(rs)/n_rs
    # if not 'SLICS' in dirpath:
        # dir_end_path = dirpath.split('/')[-1]
    savepath = startpath# + 'map_cubed_our_thetas'
    print('Writing M3 statistics to ',savepath)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    parallelized=True

    if parallelized:
        compute_all_M3(all_los,savepath+'/',n_processes=64,aperture_masses = thetas, rs=rs)
        M3=np.load(savepath+'/M3')
    else:
        M3=[]
        weights=[]
        for los in all_los:
            print("Computing los", los)
            result=compute_M3_of_field(los, thetas, rs)
            M3.append(result)
        M3=np.array(M3)
        np.save(savepath+'/M3', M3)

   
    M3=np.load(savepath+'M3.npy')

    M3_mean=np.nanmean(M3, axis=3)

    T4=np.zeros((n_thetas, n_thetas, n_thetas, n_thetas, n_thetas, n_thetas))

    for i, theta1 in enumerate(thetas):
        for j, theta2 in enumerate(thetas):
            for k, theta3 in enumerate(thetas):
                for l, theta4 in enumerate(thetas):
                    for m, theta5 in enumerate(thetas):
                        for n, theta6 in enumerate(thetas):
                            tmp=np.sum(M3_mean[i,j,k]*M3_mean[l,m,n]*rs, axis=3)
                            tmp*=delta_r/fieldsize/fieldsize
                            T4[i,j,k,l,m,n]=tmp

                            print(theta1, theta2, theta3, theta4, theta5, theta6, tmp)

    np.save(savepath+"/T4", T4)