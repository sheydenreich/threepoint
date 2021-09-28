from utility import aperture_mass_computer, create_gaussian_random_field, create_gamma_field
import numpy as np
import sys
from tqdm import tqdm
import multiprocessing.managers
from multiprocessing import Pool
from utility import extract_power_spectrum,create_gaussian_random_field
from file_loader import get_millennium
import os
import sys

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

import numpy as np


global_fieldsize_deg = 10.
n_pix = 4096

if(len(sys.argv)<2):
    print("Usage: python3 compute_aperture_mass_correlations_of_random_fields.py [powerspectrum_type] [npix] [fieldsize].")
    print("powerspectrum_type: 0: constant, 1: (x/1e4)^2*exp(-(x/1e4)^2), 2:(x/1e4)*exp(-(x/1e4))")
    print("npix: default 4096")
    print("fieldsize [deg]: default 10")
    sys.exit()

CONSTANT_POWERSPECTRUM = False
ANALYTICAL_POWERSPECTRUM = False
ANALYTICAL_POWERSPECTRUM_V2 = False

if(int(sys.argv[1])==0):
    print("Using constant powerspectrum")
    CONSTANT_POWERSPECTRUM = True

if(int(sys.argv[1])==1):
    print("Using (x/1e4)^2*exp(-(x/1e4)^2) powerspectrum")
    ANALYTICAL_POWERSPECTRUM = True

if(int(sys.argv[1])==2):
    print("Using (x/1e4)*exp(-(x/1e4)) powerspectrum")
    ANALYTICAL_POWERSPECTRUM_V2 = True

if(len(sys.argv)>2):
    print("Setting npix to ",int(sys.argv[2]))
    n_pix = int(sys.argv[2])

if(len(sys.argv)>3):
    print("Setting fieldsize to ",int(sys.argv[3])," degree")
    global_fieldsize_deg = float(sys.argv(3))

global_fieldsize_rad = global_fieldsize_deg*np.pi/180
global_fieldsize_arcmin = global_fieldsize_deg*60.



def random_shear_power_spectrum(npix,random_seed,n_bins,fieldsize,galaxy_density=None,shapenoise = 0.3):
    shapenoise_1d = shapenoise/np.sqrt(2)
    np.random.seed(random_seed)
    if(galaxy_density is None):
        shears = np.random.normal(0,shapenoise_1d,size=(npix,npix)) + 1.0j * np.random.normal(0,shapenoise_1d,size=(npix,npix))
    else:
        print("Galaxy number densities other than npix^2/fieldsize^2 not yet implemented.")
        sys.exit()
    return extract_power_spectrum(shears,fieldsize,n_bins,npix/2)

def compute_random_shear_power_spectrum_kernel(kwargs):
    result,npix,fieldsize,random_seed,n_bins,realisation = kwargs
    power_spectrum,_ = random_shear_power_spectrum(npix,random_seed,n_bins,fieldsize)
    result[:,realisation] = power_spectrum

def compute_random_shear_power_spectra(npix,fieldsize,n_realisations,n_bins,n_processes=64):
    m = MyManager()
    m.start()
    final_results = m.np_zeros((n_bins,n_realisations))

    with Pool(processes=n_processes) as p:
        args = [[final_results,npix,fieldsize,i**3+250*i,n_bins,i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(compute_random_shear_power_spectrum_kernel,args),total=n_realisations):
            pass
    return final_results


def random_aperture_mass_correlation(npix,thetas,random_seed,galaxy_density=None,shapenoise = 0.3):
    n_thetas = len(thetas)
    maxtheta = np.max(thetas)
    shapenoise_1d = shapenoise/np.sqrt(2)
    np.random.seed(random_seed)
    if(galaxy_density is None):
        shears = np.random.normal(0,shapenoise_1d,size=(npix,npix)) + 1.0j * np.random.normal(0,shapenoise_1d,size=(npix,npix))
    else:
        print("Galaxy number densities other than npix^2/fieldsize^2 not yet implemented.")
        sys.exit()
    result = extract_aperture_masses(shears,npix,thetas,compute_mcross=False)
    return result

def random_aperture_mass_computation_kernel(kwargs):
    final_result,npix,thetas,random_seed,realisation = kwargs
    result = random_aperture_mass_correlation(npix,thetas,random_seed)
    final_result[:,:,:,:,realisation] = result

def compute_random_aperture_mass_correlations(npix,thetas,n_realisations,n_processes=64):
    m = MyManager()
    m.start()
    n_theta = len(thetas)
    final_results = m.np_zeros((n_theta,n_theta,n_theta,8,n_realisations))

    with Pool(processes=n_processes) as p:
        args = [[final_results,npix,thetas,(i**3+250*i)%2**32,i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(random_aperture_mass_computation_kernel,args),total=n_realisations):
            pass
    return final_results


def aperture_mass_correlation_gaussian_random_field(power_spectrum,npix,thetas,random_seed,compute_gamma,compute_kappa,galaxy_density=None,shapenoise = 0.3):
    if(galaxy_density is None):
        kappa_field = create_gaussian_random_field(power_spectrum,npix,random_seed=random_seed)
        if(np.any(np.isnan(kappa_field))):
            print("Error! NAN in kappa!")
            sys.exit()
        if(compute_gamma):
            shears = create_gamma_field(kappa_field)
    else:
        print("Galaxy number densities other than npix^2/fieldsize^2 not yet implemented.")
        sys.exit()
    if(compute_gamma):
        result_gamma = extract_aperture_masses(shears,npix,thetas,compute_mcross=False)
    else:
        result_gamma = None

    if(compute_kappa):
        result_kappa = extract_aperture_masses(kappa_field,npix,thetas,compute_mcross=False,kappa_field=True)
    else:
        result_gamma = None

    return result_gamma,result_kappa

def aperture_mass_correlation_gaussian_random_field_kernel(kwargs):
    power_spectrum,final_results_gamma,final_results_kappa,npix,thetas,random_seed,realisation,compute_gamma,compute_kappa = kwargs
    result_gamma,result_kappa = aperture_mass_correlation_gaussian_random_field(power_spectrum,npix,thetas,random_seed,compute_gamma,compute_kappa)
    if(compute_gamma):
        final_results_gamma[:,:,:,:,realisation] = result_gamma
    if(compute_kappa):
        final_results_kappa[:,:,:,:,realisation] = result_kappa

def compute_aperture_mass_correlations_of_gaussian_random_fields(power_spectrum,npix,thetas,n_realisations,n_processes=64,compute_gamma=True,compute_kappa=False):
    m = MyManager()
    m.start()
    n_theta = len(thetas)
    if(compute_gamma):
        final_results_gamma = m.np_zeros((n_theta,n_theta,n_theta,1,n_realisations))
    else:
        final_results_gamma = None

    if(compute_kappa):
        final_results_kappa = m.np_zeros((n_theta,n_theta,n_theta,1,n_realisations))
    else:
        final_results_kappa = None

    with Pool(processes=n_processes) as p:
        args = [[power_spectrum,final_results_gamma,final_results_kappa,npix,thetas,(i**3+250*i)%2**32,i,compute_gamma,compute_kappa] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(aperture_mass_correlation_gaussian_random_field_kernel,args),total=n_realisations):
            pass

    return final_results_gamma,final_results_kappa


def extract_aperture_masses(shears,npix,thetas,compute_mcross=False,kappa_field=False):
    n_thetas = len(thetas)
    maxtheta = np.max(thetas)

    ac = aperture_mass_computer(npix,1.,global_fieldsize_arcmin)
 
    if(compute_mcross):
        result = np.zeros((n_thetas,n_thetas,n_thetas,8))
    else:
        result = np.zeros((n_thetas,n_thetas,n_thetas,1))

    aperture_mass_fields = np.zeros((npix,npix,n_thetas))
    if(compute_mcross):
        cross_aperture_fields = np.zeros((npix,npix,n_thetas))
    for x,theta in enumerate(thetas):
        ac.change_theta_ap(theta)
        if(kappa_field):
            if(compute_mcross):
                print("Error! Mcross can not be computed from kappa fields")
                sys.exit()
            map = ac.Map_fft_from_kappa(shears)
        else:
            if(compute_mcross):
                map,mx = ac.Map_fft(shears,norm=None,return_mcross=True,normalize_weighted=False)
                cross_aperture_fields[:,:,x] = mx
            else:
                map = ac.Map_fft(shears,norm=None,return_mcross=False,normalize_weighted=False)

        aperture_mass_fields[:,:,x] = map
        if(np.any(np.isnan(map))):
            print("Error! NAN in map!")
            sys.exit()

    index_maxtheta = int(maxtheta/(global_fieldsize_arcmin)*npix)*2 #take double the aperture radius and cut it off
    # print(index_maxtheta,npix,maxtheta,global_fieldsize_arcmin)

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

                field1_cut = field1[index_maxtheta:(npix-index_maxtheta),index_maxtheta:(npix-index_maxtheta)]
                field2_cut = field2[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                field3_cut = field3[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                if(compute_mcross):
                    error1_cut = error1[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                    error2_cut = error2[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]
                    error3_cut = error3[index_maxtheta:npix-index_maxtheta,index_maxtheta:npix-index_maxtheta]



                result[i,j,k,0] = np.mean(field1_cut*field2_cut*field3_cut)
                if(compute_mcross):
                    result[i,j,k,1] = np.mean(field1_cut*field2_cut*error3_cut)
                    result[i,j,k,2] = np.mean(field1_cut*error2_cut*field3_cut)
                    result[i,j,k,3] = np.mean(error1_cut*field2_cut*field3_cut)
                    result[i,j,k,4] = np.mean(error1_cut*error2_cut*field3_cut)
                    result[i,j,k,5] = np.mean(error1_cut*field2_cut*error3_cut)
                    result[i,j,k,6] = np.mean(field1_cut*error2_cut*error3_cut)
                    result[i,j,k,7] = np.mean(error1_cut*error2_cut*error3_cut)


    for i in range(n_thetas):
            for j in range(n_thetas):
                    for k in range(n_thetas):
                            i_new,j_new,k_new = np.sort([i,j,k])
                            result[i,j,k] = result[i_new,j_new,k_new]
    if(np.any(np.isnan(result))):
        print("NAN in result!")
        sys.exit()
    return result




if(__name__=='__main__'):
    # only shapenoise
    # res = compute_random_aperture_mass_correlations(4096,[1,2,4,8,16],2048,n_processes=128)
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/map_cubed_only_shapenoise',res)

    if(CONSTANT_POWERSPECTRUM):
        def power_spectrum(x):
            return 0.3**2/(2.*n_pix**2/global_fieldsize_rad**2)*np.ones(x.shape)
        savepath = '/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/constant_powerspectrum/'
        
    elif(ANALYTICAL_POWERSPECTRUM):
        def power_spectrum(x):
            return (x/10000)**2*np.exp(-(x/10000)**2)
        savepath = '/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/analytical_powerspectrum/'

    elif(ANALYTICAL_POWERSPECTRUM_V2):
        def power_spectrum(x):
            return x/10000*np.exp(-x/10000)
        savepath = '/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/analytical_powerspectrum_v2/'

    res_gamma,res_kappa = compute_aperture_mass_correlations_of_gaussian_random_fields(power_spectrum,n_pix,[1,2,4,8,16],2048,n_processes=64,compute_kappa=True)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    np.save(savepath+'map_cubed_from_gamma_npix_'+str(n_pix)+'_fieldsize_'+str(np.int(np.round(global_fieldsize_deg))),res_gamma)
    np.save(savepath+'map_cubed_from_kappa_npix_'+str(n_pix)+'_fieldsize_'+str(np.int(np.round(global_fieldsize_deg))),res_kappa)



    # res = compute_random_shear_power_spectra(4096,global_fieldsize_rad,1024,100)
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/power_spectra_only_shapenoise',res)
    # kbins = random_shear_power_spectrum(4096,1,100,global_fieldsize_rad)[1]
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/power_spectra_only_shapenoise_bins',kbins)

    # all_theta_MS = np.zeros((5,5,5,1,64))
    # for los in range(64):
    #     print(los)
    #     ms_field = get_millennium(los)
    #     # kappa_field = ms_field[:,4].reshape(4096,4096)
    #     shear_field = ms_field[:,:,2] + 1.0j*ms_field[:,:,3]
    #     # PS,bins = extract_power_spectrum(kappa_field,global_fieldsize_rad,100,4096/2)
    #     # all_ps_MS[:,los] = PS
    #     all_theta_MS[:,:,:,:,los] = extract_aperture_masses(shear_field,4096,[1,2,4,8,16],compute_mcross=False)
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/map_cubed_direct',all_theta_MS)
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/power_spectra_MS',all_ps_MS)
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/power_spectra_MS_bins',bins)
