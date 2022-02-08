from utility_public import create_gaussian_random_field_array, create_gamma_field, extract_power_spectrum
import numpy as np
from tqdm import tqdm
import multiprocessing.managers
from multiprocessing import Pool
import os
import argparse

parser = argparse.ArgumentParser(
    description='Script for computing third-order aperture mass correlations of random fields.')

parser.add_argument(
    '--npix', default=512, metavar='INT', type=int,
    help='Number of pixels in the aperture mass map. default: %(default)s'
)

parser.add_argument(
    '--fieldsize', default=5, metavar='FLOAT', type=float,
    help='Sidelength of the field in degrees. default: %(default)s'
)


parser.add_argument(
    '--processes', default=64, metavar='INT', type=int,
    help='Number of processes for parallel computation. default: %(default)s'
)

parser.add_argument(
    '--realisations', default=1024, metavar='INT', type=int,
    help='Number of realisations computed. default: %(default)s'
)

args = parser.parse_args()

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

import numpy as np


global_fieldsize_deg = args.fieldsize
n_pix = args.npix

global_fieldsize_rad = global_fieldsize_deg*np.pi/180
global_fieldsize_arcmin = global_fieldsize_deg*60.

def power_spectrum_gaussian_random_field(ell_array,power_spectrum_array,npix,random_seed):
    kappa_field = create_gaussian_random_field_array(ell_array,power_spectrum_array,n_pix=npix,fieldsize=global_fieldsize_rad,random_seed=random_seed)
    if(np.any(np.isnan(kappa_field))):
        raise ValueError("Error! NAN in kappa!")

    shears = create_gamma_field(kappa_field)
    result_gamma,kvals = extract_power_spectrum(shears,global_fieldsize_rad)
    result_kappa,_ = extract_power_spectrum(kappa_field,global_fieldsize_rad)

    return result_gamma,result_kappa,kvals

def power_spectrum_gaussian_random_field_kernel(kwargs):
    ell_array,power_spectrum_array,final_results_gamma,final_results_kappa,npix,random_seed,realisation = kwargs
    result_gamma,result_kappa,kvals = power_spectrum_gaussian_random_field(ell_array,power_spectrum_array,npix,random_seed)
    final_results_gamma[realisation,:] = result_gamma
    final_results_kappa[realisation,:] = result_kappa
    # if(realisation==args.realisations-1):
    #     print("Kvals: ")
    #     print(kvals)
    #     print("\n")

def compute_power_spectra_gaussian_random_field(fname,npix,n_realisations,n_processes=64):
    m = MyManager()
    m.start()
    final_results_gamma = m.np_zeros((n_realisations,10))
    final_results_kappa = m.np_zeros((n_realisations,10))
    data = np.loadtxt("/vol/euclid2/euclid2_raid2/sven/HOWLS/powerspectrum_{}.dat".format(fname))
    ell_array = data[:,0]
    power_spectrum_array = data[:,1]
    with Pool(processes=n_processes) as p:
        args = [[ell_array,power_spectrum_array,final_results_gamma,final_results_kappa,npix,np.random.randint(0,2**32-1),i] for i in range(n_realisations)]
        # args = [[ell_array,power_spectrum_array,final_results_gamma,final_results_kappa,npix,i**2*143+i+39485,i] for i in range(n_realisations)]

        for i in tqdm(p.imap_unordered(power_spectrum_gaussian_random_field_kernel,args),total=n_realisations,
                    desc="Computing {}".format(fname)):
            pass

    return final_results_gamma,final_results_kappa

if(__name__=='__main__'):
    savepath = "/vol/euclid2/euclid2_raid2/sven/HOWLS/power_spectra/gaussian_random_fields/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    cosmology_dict = {}
    cosmology_dict["Om"] = np.array(['02','0300912','0325988','04'])
    cosmology_dict["s8"] = np.array(['0707210','0808240','0875594','0976624'])
    cosmology_dict["w_"] = np.array(['-1.16','-1.04','-0.96','-0.84'])
    cosmology_dict["h"] = np.array(['05','06','07','08'])

    for cosmo in ["h","Om","s8","w_"]:
        for i in range(4):
            fname = "dustgrain_fine_"+cosmo+cosmology_dict[cosmo][i]
            realisations = args.realisations
            res_gamma,res_kappa = compute_power_spectra_gaussian_random_field(fname,n_pix,realisations,n_processes=args.processes)
            np.savetxt(savepath+'powerspectrum_gamma_{}.dat'.format(fname.replace("_fine","")),res_gamma)
            np.savetxt(savepath+'powerspectrum_kappa_{}.dat'.format(fname.replace("_fine","")),res_kappa)

    realisations = 2048
    fname = "SLICS_fine"
    res_gamma,res_kappa = compute_power_spectra_gaussian_random_field(fname,n_pix,realisations,n_processes=args.processes)
    np.savetxt(savepath+'powerspectrum_gamma_{}.dat'.format(fname.replace("_fine","")),res_gamma)
    np.savetxt(savepath+'powerspectrum_kappa_{}.dat'.format(fname.replace("_fine","")),res_kappa)


    # res = compute_random_shear_power_spectra(4096,global_fieldsize_rad,1024,100)
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/power_spectra_only_shapenoise',res)
    # kbins = random_shear_power_spectrum(4096,1,100,global_fieldsize_rad)[1]
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/power_spectra_only_shapenoise_bins',kbins)

    #all_theta_MS = np.zeros((5,5,5,1,32))
    #for los in range(32):
    #    print(los)
    #    los_no1 = los//8
    #    los_no2 = los%8
    #    ms_field = np.loadtxt("/vol/euclid7/euclid7_2/sven/millennium_maps/41_los_8_"+ str(los_no1) +"_"+ str(los_no2) +".ascii")
        # kappa_field = ms_field[:,4].reshape(4096,4096)
    #    shear_field = ms_field[:,2] + 1.0j*ms_field[:,3]
        # PS,bins = extract_power_spectrum(kappa_field,global_fieldsize_rad,100,4096/2)
        # all_ps_MS[:,los] = PS
    #    all_theta_MS[:,:,:,:,los] = extract_aperture_masses(shear_field,4096,[1,2,4,8,16],compute_mcross=False)
    #np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/map_cubed_direct',all_theta_MS)
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/power_spectra_MS',all_ps_MS)
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/power_spectra_MS_bins',bins)
