from utility import aperture_mass_computer
from astropy.io import fits
import numpy as np
from multiprocessing import Pool
import tqdm
import matplotlib.pyplot as plt

def read_convergence_SLICS(filename):
    kappa = fits.open(filename)[0].data[0]
    #plt.imshow(kappa)
    #plt.show()
    return kappa


def read_filenames_from_file(filename):
    with open(filename) as file:
        names=[line.rstrip() for line in file]

    return names


def make_Maps_one_field(kappa, thetas, Npix, fieldsize):

    Nthetas=len(thetas)
    Maps=np.zeros((Nthetas, Npix, Npix))

    for i,theta in enumerate(thetas):
        ac=aperture_mass_computer(Npix, theta, fieldsize)
        Maps[i]=ac.Map_fft_from_kappa(kappa)
        #plt.imshow(Maps[i])
        #plt.show()

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
    
    print(Map3s)
    return Map3s


def process_one_field(kwargs):


    filename, thetas, Npix, fieldsize, results, i=kwargs

    Ncut=int(4*np.max(thetas)*Npix/fieldsize)

    kappa=read_convergence_SLICS(filename)

    Maps=make_Maps_one_field(kappa, thetas, Npix, fieldsize)

    Map3s=make_Map3s_one_field(Maps, len(thetas), Npix, Ncut)

    results[i]=Map3s





if(__name__=='__main__'):

    Npix=1024
    fieldsize=600
    thetas=np.array([2, 4, 8, 16])

    filenames=read_filenames_from_file('SLICS_kappa_fields_filenames.dat')
    n_proc=12
    outfn="/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS/map_cubed_from_convergence.dat"


    Nthetas=len(thetas)
    Nind=int(Nthetas*(Nthetas+1)*(Nthetas+2)/6)
    Nfields=len(filenames)

    results=np.zeros((Nfields, Nind))

    # with Pool(n_proc) as pool:
    #     args=[[filenames[i], thetas, Npix, fieldsize, results, i] for i in range(Nfields)]
    #     for _ in tqdm.tqdm(pool.imap_unordered(process_one_field,args), total=Nfields):
    #         pass

    args=[[filenames[i], thetas, Npix, fieldsize, results, i] for i in range(Nfields)]
    for i in range(Nfields):
        process_one_field(args[i])

    np.savetxt(outfn, results)