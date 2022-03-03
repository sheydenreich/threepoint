from utility import aperture_mass_computer, create_lognormal_random_field, create_gamma_field
import numpy as np
import sys
from tqdm import tqdm
import multiprocessing.managers
from multiprocessing import Pool
import os
import sys
import argparse
from scipy.interpolate import interp1d
import scipy.integrate as integrate

parser = argparse.ArgumentParser(
    description='Script for computing third-order aperture mass correlations of lognormal fields.')

parser.add_argument(
    '--npix', default=4096, metavar='INT', type=int,
    help='Number of pixels in the aperture mass map. default: %(default)s'
)

parser.add_argument(
    '--fieldsize', default=10, metavar='FLOAT', type=float,
    help='Sidelength of the field in degrees. default: %(default)s'
)

parser.add_argument(
    '--compute_from_kappa', action='store_true',
    help='Also computes the aperture mass from the kappa-maps. default: %(default)s'
)

parser.add_argument(
    '--substract_mean', action='store_true',
    help='Substracts the mean value of the kappa-maps. default: %(default)s'
)

parser.add_argument(
    '--calculate_mcross', action='store_true',
    help='Also compute the cross-aperture statistics. default: %(default)s'
)


parser.add_argument(
    '--power_spectrum_filename',
    help='filename for the power spectrum (of the final lognormal fields)'
)

parser.add_argument(
    '--alpha', default=0.5, metavar='FLOAT', type=float,
    help='Parameter describing departure from Gaussianity. default: %(default)s'
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


print("Using power spectrum from input file: ",args.power_spectrum_filename)
 


global_fieldsize_rad = global_fieldsize_deg*np.pi/180
global_fieldsize_arcmin = global_fieldsize_deg*60.



def random_aperture_mass_correlation(npix,thetas,random_seed,galaxy_density=None,shapenoise = 0.3, periodic_boundary=True):
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


def aperture_mass_correlation_lognormal_field(power_spectrum, alpha, sigma, npix,thetas,random_seed,compute_gamma,compute_kappa,galaxy_density=None):
    if(galaxy_density is None):
        kappa_field=create_lognormal_random_field(power_spectrum,alpha,random_seed=random_seed,sigma=sigma)
        
        if(args.substract_mean):
            kappa_field = kappa_field - np.mean(kappa_field)
        if(np.any(np.isnan(kappa_field))):
            print("Error! NAN in kappa!")
            sys.exit()
        if(compute_gamma):
            shears = create_gamma_field(kappa_field)
    else:
        print("Galaxy number densities other than npix^2/fieldsize^2 not yet implemented.")
        sys.exit()
    if(compute_gamma):
        result_gamma = extract_aperture_masses(shears,npix,thetas,compute_mcross=args.calculate_mcross)
    else:
        result_gamma = None

    if(compute_kappa):
        result_kappa = extract_aperture_masses(kappa_field,npix,thetas,compute_mcross=False,kappa_field=True)
    else:
        result_kappa = None

    return result_gamma,result_kappa

def aperture_mass_correlation_lognormal_field_kernel(kwargs):
    power_spectrum, alpha, sigma, final_results_gamma,final_results_kappa,npix,thetas,random_seed,realisation,compute_gamma,compute_kappa = kwargs
    result_gamma,result_kappa = aperture_mass_correlation_lognormal_field(power_spectrum, alpha, sigma, npix,thetas,random_seed,compute_gamma,compute_kappa)
    if(compute_gamma):
        final_results_gamma[:,:,:,:,realisation] = result_gamma
    if(compute_kappa):
        final_results_kappa[:,:,:,:,realisation] = result_kappa

def compute_aperture_mass_correlations_of_lognormal_fields(power_spectrum,alpha, sigma,npix,thetas,n_realisations,n_processes=64,compute_gamma=True,compute_kappa=False):
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
        args = [[power_spectrum, alpha, sigma, final_results_gamma,final_results_kappa,npix,thetas,(i**3+250*i)%2**32,i,compute_gamma,compute_kappa] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(aperture_mass_correlation_lognormal_field_kernel,args),total=n_realisations):
            pass

    return final_results_gamma,final_results_kappa


def extract_aperture_masses(shears,npix,thetas,compute_mcross=False,kappa_field=False, periodic_boundary=False):
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
                map,mx = ac.Map_fft(shears,norm=None,return_mcross=True,normalize_weighted=False, periodic_boundary=periodic_boundary)
                cross_aperture_fields[:,:,x] = mx
            else:
                map = ac.Map_fft(shears,norm=None,return_mcross=False,normalize_weighted=False, periodic_boundary=periodic_boundary)

        aperture_mass_fields[:,:,x] = map
        if(np.any(np.isnan(map))):
            print("Error! NAN in map!")
            sys.exit()

    index_maxtheta = int(round(maxtheta/(global_fieldsize_arcmin)*npix*4)) #take 4* the aperture radius and cut it off
    if(periodic_boundary):
        index_maxtheta = 0

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
    if(np.sum(result)==0):
        print("Error! Result is zero!")
        sys.exit()
    return result




if(__name__=='__main__'):

    
    savepath = f"/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/LognormalFields_alpha_{args.alpha:.1f}/"

# Get sigma (standard-deviation of GRF = correlation function at zero)
    power_spectrum = np.loadtxt(args.power_spectrum_filename)
    ells = power_spectrum[:,0]
    vals = power_spectrum[:,1]

    power_spectrum_func = interp1d(ells,vals,fill_value=0,bounds_error=False)

    sigma = np.sqrt(integrate.quad(lambda ell: ell*power_spectrum_func(ell),10,10**5)[0]/(2.*np.pi))
    print(sigma)
# Do Calculation
    res_gamma,res_kappa = compute_aperture_mass_correlations_of_lognormal_fields(power_spectrum_func,args.alpha, sigma, n_pix,[2,4,8,16],args.realisations,n_processes=args.processes,compute_kappa=args.compute_from_kappa)
# Output
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savename = 'npix_'+str(n_pix)+'_fieldsize_'+str(np.int(np.round(global_fieldsize_deg)))
    if(args.substract_mean):
        savename += '_mean_substracted'
    np.save(savepath+'map_cubed_from_gamma_'+savename,res_gamma)
    if(args.compute_from_kappa):
        np.save(savepath+'map_cubed_from_kappa_'+savename,res_kappa)
