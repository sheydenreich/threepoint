from utility_public import extract_power_spectrum
import numpy as np
from tqdm import tqdm
import multiprocessing.managers
from multiprocessing import Pool
from astropy.io import fits
import os

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


startpath = '/vol/euclid7/euclid7_2/llinke/HOWLS/'
# '/vol/euclid7/euclid7_2/llinke/HOWLS/'


def compute_power_spectrum_of_field(filepath,savename):
    slics = ('SLICS' in filepath)
    if(slics):
        fieldsize = 600.
    else:
        fieldsize = 5*60.

    fieldsize *= (np.pi/60./180.) #convert arcmin -> rad

    field = fits.open(filepath)
    data = (field[0].data)[0]
    result,bins = extract_power_spectrum(data, fieldsize, lmin=10, lmax=5e4, bins=128)
    np.savetxt(savename,result)
    return result

def compute_all_power_spectra(openpath,filenames,savepath,n_processes = 64):
    n_files = len(filenames)
    with Pool(processes=n_processes) as p:
        # print('test')
        result = [p.apply_async(compute_power_spectrum_of_field, args=(openpath+filenames[i],savepath+"powerspectrum_"+filenames[i].split(".fits")[0]+".dat")) for i in range(n_files)]
        data = [p.get() for p in result]
        datavec = np.array([data[i] for i in range(len(data))])
        np.savetxt(savepath+'powerspectra',datavec)


if(__name__=='__main__'):

    for (dirpath,_,_filenames) in os.walk(startpath+"convergence_maps/"):
        if(len(_filenames)>2):
            filenames = np.sort([filename for filename in _filenames if filename.endswith(".fits")])
            # if not 'SLICS' in dirpath:
            	# dir_end_path = dirpath.split('/')[-1]
            savepath = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/GaussianRandomFields_slicslike/"
#"/vol/euclid2/euclid2_raid2/sven/HOWLS/power_spectra"+dirpath.split('convergence_maps')[1]
            print('Reading convergence maps from ',dirpath)
            print('Writing power spectra to ',savepath)
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            compute_all_power_spectra(dirpath+'/',filenames,savepath+'/',n_processes=64)
