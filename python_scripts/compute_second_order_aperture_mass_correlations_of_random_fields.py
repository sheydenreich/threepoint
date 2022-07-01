from utility import aperture_mass_computer, create_gaussian_random_field_array, create_gamma_field, extract_power_spectrum, create_gaussian_random_field
import numpy as np
import sys
from tqdm import tqdm
import multiprocessing.managers
from multiprocessing import Pool
import os
import sys
import argparse

parser = argparse.ArgumentParser(
    description='Script for computing third-order aperture mass correlations of random fields.')

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
    '--power_spectrum', default=0, metavar='INT', type=int,
    help='Type of power spectrum used. \n 0:\t constant\n 1:\t (x/1e4)^2*exp(-(x/1e4)^2)\n 2:\t (x/1e4)*exp(-(x/1e4))\n default: %(default)s'
)

parser.add_argument(
    '--power_spectrum_filename',
    help='if power_spectrum=-1, filename for the power spectrum'
)


parser.add_argument(
    '--processes', default=64, metavar='INT', type=int,
    help='Number of processes for parallel computation. default: %(default)s'
)



parser.add_argument(
    '--realisations', default=1024, metavar='INT', type=int,
    help='Number of realisations computed. default: %(default)s'
)

parser.add_argument(
    '--savepath', default="", help="Outputpath"
)

parser.add_argument(
    '--cutOutFromBiggerField', action='store_true',
    help='Cut out the random fields from a random field with size 10x field_size.'
)

args = parser.parse_args()

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

import numpy as np


global_fieldsize_deg = args.fieldsize
n_pix = args.npix

CONSTANT_POWERSPECTRUM = False
ANALYTICAL_POWERSPECTRUM = False
ANALYTICAL_POWERSPECTRUM_V2 = False
INPUT_FILE_POWER_SPECTRUM = False


if(args.power_spectrum<0):
    print("Using power spectrum from input file: ",args.power_spectrum_filename)
    INPUT_FILE_POWER_SPECTRUM = True

if(args.power_spectrum==0):
    print("Using constant powerspectrum")
    CONSTANT_POWERSPECTRUM = True

if(args.power_spectrum==1):
    print("Using (x/1e4)^2*exp(-(x/1e4)^2) powerspectrum")
    ANALYTICAL_POWERSPECTRUM = True

if(args.power_spectrum==2):
    print("Using (x/1e4)*exp(-(x/1e4)) powerspectrum")
    ANALYTICAL_POWERSPECTRUM_V2 = True



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


def random_aperture_mass_correlation(npix,thetas,random_seed,galaxy_density=None,shapenoise = 0.3, periodic_boundary=True):
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
    final_result[:,realisation] = result

def compute_random_aperture_mass_correlations(npix,thetas,n_realisations,n_processes=64, periodic_boundary=True):
    m = MyManager()
    m.start()
    n_theta = len(thetas)
    final_results = m.np_zeros((n_theta,n_realisations))

    with Pool(processes=n_processes) as p:
        args = [[final_results,npix,thetas,(i**3+250*i)%2**32,i, periodic_boundary] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(random_aperture_mass_computation_kernel,args),total=n_realisations):
            pass
    return final_results


def aperture_mass_correlation_gaussian_random_field(power_spectrum,npix,thetas,random_seed,compute_gamma,compute_kappa,galaxy_density=None, cutOutFromBiggerField=False):
    if(galaxy_density is None):
        if(INPUT_FILE_POWER_SPECTRUM):
            ell = power_spectrum[:,0]
            pkappa_of_ell = power_spectrum[:,1]
            if cutOutFromBiggerField:
                kappa_field = create_gaussian_random_field_array(ell,pkappa_of_ell,n_pix=5*npix,fieldsize=5*global_fieldsize_rad,random_seed=random_seed)
            else:
                kappa_field = create_gaussian_random_field_array(ell, pkappa_of_ell, n_pix=n_pix, fieldsize=global_fieldsize_rad, random_seed=random_seed)
        else:
            if cutOutFromBiggerField:
                kappa_field = create_gaussian_random_field(power_spectrum,n_pix=5*npix,fieldsize=5*global_fieldsize_rad,random_seed=random_seed)
            else:
                kappa_field = create_gaussian_random_field(power_spectrum,n_pix=npix,fieldsize=global_fieldsize_rad,random_seed=random_seed)
        if(args.substract_mean):
            kappa_field = kappa_field - np.mean(kappa_field)
        if(np.any(np.isnan(kappa_field))):
            print("Error! NAN in kappa!")
            sys.exit()

        if cutOutFromBiggerField:
            kappa_field=kappa_field[2*n_pix:3*n_pix, 2*n_pix:3*n_pix]
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

def aperture_mass_correlation_gaussian_random_field_kernel(kwargs):
    power_spectrum,final_results_gamma,final_results_kappa,npix,thetas,random_seed,realisation,compute_gamma,compute_kappa = kwargs
    result_gamma,result_kappa = aperture_mass_correlation_gaussian_random_field(power_spectrum,npix,thetas,random_seed,compute_gamma,compute_kappa)
    if(compute_gamma):
        final_results_gamma[:,realisation] = result_gamma
    if(compute_kappa):
        final_results_kappa[:,realisation] = result_kappa

def compute_aperture_mass_correlations_of_gaussian_random_fields(power_spectrum,npix,thetas,n_realisations,n_processes=64,compute_gamma=True,compute_kappa=False):
    m = MyManager()
    m.start()
    n_theta = len(thetas)
    if(compute_gamma):
        final_results_gamma = m.np_zeros((n_theta,n_realisations))
    else:
        final_results_gamma = None

    if(compute_kappa):
        final_results_kappa = m.np_zeros((n_theta,n_realisations))
    else:
        final_results_kappa = None

    with Pool(processes=n_processes) as p:
        args = [[power_spectrum,final_results_gamma,final_results_kappa,npix,thetas,(i**3+250*i)%2**32,i,compute_gamma,compute_kappa] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(aperture_mass_correlation_gaussian_random_field_kernel,args),total=n_realisations):
            pass

    return final_results_gamma,final_results_kappa


def extract_aperture_masses(shears,npix,thetas,compute_mcross=False,kappa_field=False, periodic_boundary=False):
    n_thetas = len(thetas)
    maxtheta = np.max(thetas)

    ac = aperture_mass_computer(npix,1.,global_fieldsize_arcmin)
 
    if(compute_mcross):
        result = np.zeros((n_thetas,2))
    else:
        result = np.zeros(n_thetas)

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
    # print(index_maxtheta,npix,maxtheta,global_fieldsize_arcmin)
    if(periodic_boundary):
        index_maxtheta = 0

    for i in range(n_thetas):
        field1 = aperture_mass_fields[:,:,i]
        field1_cut = field1[index_maxtheta:(npix-index_maxtheta),index_maxtheta:(npix-index_maxtheta)]
        result[i] = np.mean(field1_cut**2)


    if(np.any(np.isnan(result))):
        print("NAN in result!")
        sys.exit()
    if(np.sum(result)==0):
        print("Error! Result is zero!")
        sys.exit()
    return result




if(__name__=='__main__'):
    # only shapenoise
#     res = compute_random_aperture_mass_correlations(4096,[1,2,4,8,16],2048,n_processes=64, periodic_boundary=True)
# #    np.save('map_cubed_only_shapenoise_without_zeropadding', res)
#     np.save('/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/map_cubed_only_shapenoise_without_zeropadding',res)

    
    #res = compute_random_aperture_mass_correlations(4096,[1,2,4,8,16],1,n_processes=12, periodic_boundary=True)
    #np.save('test',res)

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

    elif(INPUT_FILE_POWER_SPECTRUM):
        power_spectrum = np.loadtxt(args.power_spectrum_filename)
        filename = args.power_spectrum_filename.split("/")[-1]
        savepath = '/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/GaussianRandomFields_slicslike/'
        #'/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_cosmicShearShapenoise/'
        #'/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/GaussianRandomFields_cosmicShearShapenoise/'
#'/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/input_powerspectrum/'+filename.split(".")[0]
        

    res_gamma,res_kappa = compute_aperture_mass_correlations_of_gaussian_random_fields(power_spectrum,n_pix,[1.17,2.34,4.69,9.37],args.realisations,n_processes=args.processes,compute_kappa=args.compute_from_kappa)
    
    if(args.savepath!=""):
        savepath=args.savepath
    
    
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savename = 'npix_'+str(n_pix)+'_fieldsize_'+str(np.int(np.round(global_fieldsize_deg)))
    if(args.substract_mean):
        savename += '_mean_substracted'
    np.save(savepath+'map_squared_from_gamma_'+savename,res_gamma)
    if(args.compute_from_kappa):
        np.save(savepath+'map_squared_from_kappa_'+savename,res_kappa)



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
