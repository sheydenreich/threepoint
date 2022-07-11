from utility import aperture_mass_computer
from astropy.io import fits
import numpy as np
from multiprocessing import Pool
import multiprocessing.managers
import tqdm
import matplotlib.pyplot as plt
from numpy.random import default_rng

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)
import numpy as np

def read_gamma_SLICS(filename, shapenoise=0, seed=42):
    data = fits.open(filename)[1].data
    
    X_pos = data['ra_gal']*60.
    Y_pos = data['dec_gal']*60.
    shear_noise = -data['gamma1_noise']+1.0j*data['gamma2_noise']

    if(shapenoise>0):
        rng=default_rng(seed=seed)
        sn=rng.normal(size=(len(shear_noise), 2), scale=shapenoise)
        noise=sn[:,0]+1.0j*sn[:,1]
        shear_noise=(shear_noise+noise)



    return X_pos, Y_pos, shear_noise


def read_filenames_from_file(filename):
    with open(filename) as file:
        names=[line.rstrip() for line in file]

    return names


def make_Maps_one_field(shears_normed, norm, thetas, Npix, fieldsize):

    Nthetas=len(thetas)
    Maps=np.zeros((Nthetas, Npix, Npix))

    for i,theta in enumerate(thetas):
        ac=aperture_mass_computer(Npix, theta, fieldsize)
        Maps[i]=ac.Map_fft(shears_normed, norm=norm)
        # plt.imshow(Maps[i])
        # plt.show()

    return Maps



def make_Map3s_one_field(Maps, Nthetas, Npix, Ncut):

    Nind=int(Nthetas*(Nthetas+1)*(Nthetas+2)/6)

    Map3s=np.zeros(Nind)

    counter=0
    for i in range(Nthetas):
        for j in range(i, Nthetas):
            for k in range(j, Nthetas):
                Map3s[counter]=np.mean(Maps[i, Ncut:Npix-Ncut, Ncut:Npix-Ncut ]*Maps[j, Ncut:Npix-Ncut, Ncut:Npix-Ncut]*Maps[k, Ncut:Npix-Ncut, Ncut:Npix-Ncut])
                counter+=1
    
    return Map3s


def process_one_field(kwargs):


    filename, thetas, Npix, fieldsize, results, i, shapenoise, seed=kwargs

    Ncut=int(4*np.max(thetas)*Npix/fieldsize)

    X_pos, Y_pos, shears=read_gamma_SLICS(filename, shapenoise, seed)

    ac=aperture_mass_computer(Npix, 1, fieldsize)
    shears_normed, norm=ac.normalize_shear(X_pos, Y_pos, shears)


    Maps=make_Maps_one_field(shears_normed, norm, thetas, Npix, fieldsize)

    Map3s=make_Map3s_one_field(Maps, len(thetas), Npix, Ncut)
    print(Map3s)
    results[i]=Map3s
    print(results[i])





if(__name__=='__main__'):

    Npix=1024
    fieldsize=600
    thetas=np.array([2, 4, 8, 16])
    shapenoise=0.5

    filenames=read_filenames_from_file('SLICS_gamma_fields_filenames.dat')
    n_proc=10
    outfn="/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS_sn0_5/map_cubed"
    Nthetas=len(thetas)
    Nind=int(Nthetas*(Nthetas+1)*(Nthetas+2)/6)
    Nfields=len(filenames)

    m = MyManager()
    m.start()
    results=m.np_zeros((Nfields, Nind))

    with Pool(n_proc) as pool:
        args=[[filenames[i], thetas, Npix, fieldsize, results, i, shapenoise/np.sqrt(2), i] for i in range(Nfields)]
        for _ in tqdm.tqdm(pool.imap_unordered(process_one_field,args), total=Nfields):
            pass

    # args=[[filenames[i], thetas, Npix, fieldsize, results, i] for i in range(Nfields)]
    # for i in range(Nfields):
    #     process_one_field(args[i])
    print(results)
    np.savetxt(outfn, results)