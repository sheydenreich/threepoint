from utility import aperture_mass_computer, create_gaussian_random_field, create_gamma_field
import numpy as np
import sys
from tqdm import tqdm
import multiprocessing.managers
from multiprocessing import Pool
from utility import extract_power_spectrum,create_gaussian_random_field

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

import numpy as np

global_fieldsize_arcmin = 10.*60.

global_fieldsize_deg = global_fieldsize_arcmin/60
global_fieldsize_rad = global_fieldsize_deg*np.pi/180



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


def random_aperture_mass_correlation(npix,thetas,random_seed,galaxy_density=None,shapenoise = 0.3, periodic_boundary=False):
    n_thetas = len(thetas)
    maxtheta = np.max(thetas)
    shapenoise_1d = shapenoise/np.sqrt(2)
    np.random.seed(random_seed)
    if(galaxy_density is None):
        shears = np.random.normal(0,shapenoise_1d,size=(npix,npix)) + 1.0j * np.random.normal(0,shapenoise_1d,size=(npix,npix))
    else:
        print("Galaxy number densities other than npix^2/fieldsize^2 not yet implemented.")
        sys.exit()
    result = extract_aperture_masses(shears,npix,thetas,compute_mcross=False, periodic_boundary=periodic_boundary)
    return result

def random_aperture_mass_computation_kernel(kwargs):
    final_result,npix,thetas,random_seed,realisation, periodic_boundary = kwargs
    result = random_aperture_mass_correlation(npix,thetas,random_seed, periodic_boundary=periodic_boundary)
    final_result[:,:,:,:,realisation] = result

def compute_random_aperture_mass_correlations(npix,thetas,n_realisations,n_processes=64, periodic_boundary=True):
    m = MyManager()
    m.start()
    n_theta = len(thetas)
    final_results = m.np_zeros((n_theta,n_theta,n_theta,8,n_realisations))

    with Pool(processes=n_processes) as p:
        args = [[final_results,npix,thetas,(i**3+250*i)%2**32,i, periodic_boundary] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(random_aperture_mass_computation_kernel,args),total=n_realisations):
            pass
    return final_results


def aperture_mass_correlation_gaussian_random_field(power_spectrum,npix,thetas,random_seed,galaxy_density=None,shapenoise = 0.3):
    if(galaxy_density is None):
        kappa_field = create_gaussian_random_field(power_spectrum,npix,random_seed=random_seed)
        shears = create_gamma_field(kappa_field)
    else:
        print("Galaxy number densities other than npix^2/fieldsize^2 not yet implemented.")
        sys.exit()

    result = extract_aperture_masses(shears,npix,thetas,compute_mcross=False)
    return result

def aperture_mass_correlation_gaussian_random_field_kernel(kwargs):
    power_spectrum,final_result,npix,thetas,random_seed,realisation = kwargs
    result = aperture_mass_correlation_gaussian_random_field(power_spectrum,npix,thetas,random_seed)
    final_result[:,:,:,:,realisation] = result

def compute_aperture_mass_correlations_of_gaussian_random_fields(power_spectrum,npix,thetas,n_realisations,n_processes=64):
    m = MyManager()
    m.start()
    n_theta = len(thetas)
    final_results = m.np_zeros((n_theta,n_theta,n_theta,1,n_realisations))

    with Pool(processes=n_processes) as p:
        args = [[power_spectrum,final_results,npix,thetas,(i**3+250*i)%2**32,i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(aperture_mass_correlation_gaussian_random_field_kernel,args),total=n_realisations):
            pass

    return final_results


def extract_aperture_masses(shears,npix,thetas,compute_mcross=False, periodic_boundary=False):
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
        if(compute_mcross):
            map,mx = ac.Map_fft(shears,norm=None,return_mcross=True,normalize_weighted=False, periodic_boundary=periodic_boundary)
            cross_aperture_fields[:,:,x] = mx
        else:
            map = ac.Map_fft(shears,norm=None,return_mcross=False,normalize_weighted=False, periodic_boundary=periodic_boundary)

        aperture_mass_fields[:,:,x] = map

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

                index_maxtheta = int(maxtheta/(global_fieldsize_arcmin)*npix)*2 #take double the aperture radius and cut it off
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
    return result




if(__name__=='__main__'):
    # only shapenoise
    # res = compute_random_aperture_mass_correlations(4096,[1,2,4,8,16],2048,n_processes=128)
    res = compute_random_aperture_mass_correlations(4096,[1,2,4,8,16],1,n_processes=12, periodic_boundary=True)
    # np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/map_cubed_only_shapenoise',res)
    np.save('test',res)

    # gaussian random field in kappa
    #def power_spectrum(x):
    #    return x/10000*np.exp(-x/10000)

    #res = compute_aperture_mass_correlations_of_gaussian_random_fields(power_spectrum,4096,[1,2,4,8,16],2048,n_processes=64)
    #np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/map_cubed_gaussian_random_field_x_exp_minus_x',res)


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
