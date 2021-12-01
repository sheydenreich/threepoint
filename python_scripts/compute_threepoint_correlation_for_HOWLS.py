import numpy as np
import sys
from tqdm import tqdm
import multiprocessing.managers
from multiprocessing import Pool
from astropy.io import fits
import os
import treecorr
from time import time

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

offset = int(sys.argv[1])
n_proc = int(sys.argv[2])

startpath = '/vol/euclid2/euclid2_raid2/sven/HOWLS/'


def compute_3pcf_of_field(fname,savepath,
        bins_r = 10, bins_u = 10, bins_v = 10, nprocs = 8, min_sep = 0.1, max_sep = 100.):
    if(os.path.exists(savepath)):
        print(savepath," already exists. Moving on.")
        return 0

    field = fits.open(fname)
    data = field[1].data

    X_pos = data['ra_gal']*60.
    Y_pos = data['dec_gal']*60.

    cat = treecorr.Catalog(x=X_pos,y=Y_pos,x_units='arcmin',y_units='arcmin',
    g1=-data['gamma1_noise'],g2=data['gamma2_noise'])
    ggg = treecorr.GGGCorrelation(nbins=bins_r,min_sep=min_sep,max_sep=max_sep,sep_units='arcmin',
            nubins=bins_u,min_u=0,max_u=1,nvbins=bins_v,min_v=0,max_v=1,verbose=0,num_threads=nprocs)
    startt = time()
    ggg.process(cat)
    print(fname,"done in {:.1f} h on {} cores. \n Saving as".format((time()-startt)/3600,nprocs),savepath)
    ggg.write(savepath)
    return ggg

def compute_all_3pcfs(filenames,savenames,n_processes = 8):
    n_files = len(filenames)
    with Pool(processes=n_processes) as p:
        # print('test')
        result = [p.apply_async(compute_3pcf_of_field, args=(filenames[i],savenames[i],)) for i in range(n_files)]
        temp = [p.get() for p in result]

if(__name__=='__main__'):
    for (dirpath,_,_filenames) in os.walk(startpath+"shear_catalogues/"):
        if(len(_filenames)>2 and 'SLICS' in dirpath):
            filenames = np.sort([filename for filename in _filenames if ".fits" in filename])[offset:]
            savepath = dirpath.split('shear_catalogues')[0]+'3pcfs'+dirpath.split('shear_catalogues')[1]
            print('Reading shear catalogues from ',dirpath)
            print('Writing summary statistics to ',savepath)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            readnames = []
            savenames = []
            for i in range(len(filenames)):
                readnames.append(dirpath+os.sep+filenames[i])
                savenames.append(dirpath+os.sep+filenames[i].split(".")[0]+".dat")
            compute_all_3pcfs(readnames,savenames,n_processes=n_proc)
