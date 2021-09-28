from utility import aperture_mass_computer
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

def extract_aperture_masses(Xs,Ys,shear_catalogue,npix,thetas,fieldsize,compute_mcross=False):
    n_thetas = len(thetas)
    maxtheta = np.max(thetas)

    ac = aperture_mass_computer(npix,1.,fieldsize)
    shears,norm = ac.normalize_shear(Xs,Ys,shear_catalogue)

 
    if(compute_mcross):
        # result = np.zeros((n_thetas,n_thetas,n_thetas,8))
        return
    else:
        result = np.zeros(n_thetas*(n_thetas+1)*(n_thetas+2)//6)

    aperture_mass_fields = np.zeros((npix,npix,n_thetas))
    if(compute_mcross):
        cross_aperture_fields = np.zeros((npix,npix,n_thetas))

    for x,theta in enumerate(thetas):
        ac.change_theta_ap(theta)
        if(compute_mcross):
            map,mx = ac.Map_fft(shears,norm=None,return_mcross=True,normalize_weighted=False)
            cross_aperture_fields[:,:,x] = mx
        else:
            map = ac.Map_fft(shears,norm=norm,return_mcross=False)

        aperture_mass_fields[:,:,x] = map

    counter = 0
    for i in range(n_thetas):
        field1 = aperture_mass_fields[:,:,i]
        if(compute_mcross):
            error1 = cross_aperture_fields[:,:,i]
        for j in range(i,n_thetas):
            field2 = aperture_mass_fields[:,:,j]
            if(compute_mcross):
                error2 = cross_aperture_fields[:,:,j]
            for k in range(j,n_thetas):                     
                field3 = aperture_mass_fields[:,:,k]
                if(compute_mcross):
                    error3 = cross_aperture_fields[:,:,k]

                index_maxtheta = int(maxtheta/(fieldsize)*npix)*2 #take double the aperture radius and cut it off
                field1_cut = field1[index_maxtheta:(npix-index_maxtheta),index_maxtheta:(npix-index_maxtheta)]
                field2_cut = field2[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                field3_cut = field3[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                if(compute_mcross):
                    error1_cut = error1[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                    error2_cut = error2[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                    error3_cut = error3[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]



                if(compute_mcross):
                    result[i,j,k,0] = np.mean(field1_cut*field2_cut*field3_cut)
                    result[i,j,k,1] = np.mean(field1_cut*field2_cut*error3_cut)
                    result[i,j,k,2] = np.mean(field1_cut*error2_cut*field3_cut)
                    result[i,j,k,3] = np.mean(error1_cut*field2_cut*field3_cut)
                    result[i,j,k,4] = np.mean(error1_cut*error2_cut*field3_cut)
                    result[i,j,k,5] = np.mean(error1_cut*field2_cut*error3_cut)
                    result[i,j,k,6] = np.mean(field1_cut*error2_cut*error3_cut)
                    result[i,j,k,7] = np.mean(error1_cut*error2_cut*error3_cut)
                else:
                    result[counter] = np.mean(field1_cut*field2_cut*field3_cut)
                counter += 1

    # for i in range(n_thetas):
    #         for j in range(n_thetas):
    #                 for k in range(n_thetas):
    #                         i_new,j_new,k_new = np.sort([i,j,k])
    #                         result[i,j,k] = result[i_new,j_new,k_new]
    return result

def compute_aperture_masses_of_field(filepath,theta_ap_array):
    slics = ('SLICS' in filepath)
    if(slics):
        fieldsize = 600.
        npix = 4096
    else:
        fieldsize = 5*60.
        npix = 2048

    field = fits.open(filepath)
    data = field[1].data

    if(slics):
        X_pos = data['x_arcmin']
        Y_pos = data['y_arcmin']
        shear = data['shear1']+1.0j*data['shear2']
        noise = data['e1_intr']+1.0j*data['e2_intr']
        shear_noise = (shear+noise)/(1+shear*np.conj(noise))
    else:
        X_pos = data['ra_gal']*60.
        Y_pos = data['dec_gal']*60.
        shear_noise = -data['gamma1_noise']+1.0j*data['gamma2_noise']

    result = extract_aperture_masses(X_pos,Y_pos,shear_noise,npix,theta_ap_array,fieldsize,compute_mcross=False)

    return result

def compute_all_aperture_masses(openpath,filenames,savepath,aperture_masses = [1.17,2.34,4.69,9.37],n_processes = 64):
    n_files = len(filenames)
    with Pool(processes=n_processes) as p:
        # print('test')
        result = [p.apply_async(compute_aperture_masses_of_field, args=(openpath+filenames[i],aperture_masses,)) for i in range(n_files)]
        data = [p.get() for p in result]
        datavec = np.array([data[i] for i in range(len(data))])
        np.savetxt(savepath+'map_cubed',datavec)

if(__name__=='__main__'):
    for (dirpath,_,_filenames) in os.walk(startpath+"shear_catalogues/"):
        if(len(_filenames)>2):
            filenames = [filename for filename in _filenames if '.fits' in filename]
            if not 'SLICS' in dirpath:
            	# dir_end_path = dirpath.split('/')[-1]
                savepath = dirpath.split('shear_catalogues')[0]+'map_cubed'+dirpath.split('shear_catalogues')[1]
                print('Reading shear catalogues from ',dirpath)
                print('Writing summary statistics to ',savepath)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

                compute_all_aperture_masses(dirpath+'/',filenames,savepath+'/')#,aperture_masses = [0.5,1,2,4,8,16,32])

    for (dirpath,_,_filenames) in os.walk(startpath+"shear_catalogues/"):
        if(len(_filenames)>2):
            filenames = [filename for filename in _filenames if '.fits' in filename]
            # dir_end_path = dirpath.split('/')[-1]
            savepath = dirpath.split('shear_catalogues')[0]+'map_cubed_our_thetas'+dirpath.split('shear_catalogues')[1]
            print('Reading shear catalogues from ',dirpath)
            print('Writing summary statistics to ',savepath)
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            compute_all_aperture_masses(dirpath+'/',filenames,savepath+'/',aperture_masses = [0.5,1,2,4,8,16,32])
