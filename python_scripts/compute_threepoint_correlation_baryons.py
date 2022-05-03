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

startpath = '/vol/euclid2/euclid2_raid2/sven/SLICS_hydrosims/'


def compute_3pcf_of_field(fname,savepath,
        bins_r = 10, bins_u = 10, bins_v = 10, nprocs = 32, min_sep = 0.5, max_sep = 120.):
    if(os.path.exists(savepath)):
        print(savepath," already exists. Moving on.")
        return 0

    data = np.loadtxt(fname)
    # data = field[1].data

    X_pos = data[:,0]
    Y_pos = data[:,1]

    cat = treecorr.Catalog(x=X_pos,y=Y_pos,x_units='arcmin',y_units='arcmin',
    g1=-1.*data[:,3],g2=data[:,4])
    
    # gg = treecorr.GGCorrelation(nbins=bins_r,min_sep=min_sep,max_sep=max_sep,sep_units='arcmin',
    #                             verbose=0,num_threads=nprocs)
    
    ggg = treecorr.GGGCorrelation(nbins=bins_r,min_sep=min_sep,max_sep=max_sep,sep_units='arcmin',
            nubins=bins_u,min_u=0,max_u=1,nvbins=bins_v,min_v=0,max_v=1,verbose=0,num_threads=nprocs)
    startt = time()
    print("Calculating ",fname," on {} cores.".format(nprocs))
    ggg.process(cat)
    print(fname,"done in {:.1f} h on {} cores. \n Saving as".format((time()-startt)/3600,nprocs),savepath)
    ggg.write(savepath)
    return ggg

def compute_all_3pcfs(filenames,savenames,n_processes = 4):
    n_files = len(filenames)
    with Pool(processes=n_processes) as p:
        # print('test')
        result = [p.apply_async(compute_3pcf_of_field, args=(filenames[i],savenames[i],)) for i in range(n_files)]
        temp = [p.get() for p in result]

if(__name__=='__main__'):
    savepath = startpath+'3pcfs_0p5_to_120_10_bins/'
    print('Reading shear catalogues from ',startpath)
    print('Writing summary statistics to ',savepath)

    _filenames = os.listdir(startpath)
    filenames = np.sort([filename for filename in _filenames if ".cat" in filename])[offset:]
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    readnames = []
    savenames = []
    for i in range(len(filenames)):
        readnames.append(startpath+filenames[i])
        savenames.append(savepath+filenames[i].split(".")[0]+(filenames[i].split(".")[-2])[1:]+".dat")
    print("Computing {} correlation functions".format(len(readnames)))
    compute_all_3pcfs(readnames,savenames,n_processes=n_proc)
