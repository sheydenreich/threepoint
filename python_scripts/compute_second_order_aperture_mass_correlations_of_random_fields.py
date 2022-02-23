from utility import aperture_mass_computer, extract_both_aperture_masses_of_field, create_gaussian_random_field, create_gaussian_random_field_array, create_gamma_field, extract_power_spectrum, extract_aperture_masses, extract_second_order_aperture_masses
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
    help='Type of power spectrum used. \n -1: \t from file 0:\t constant\n 1:\t (x/1e4)^2*exp(-(x/1e4)^2)\n 2:\t (x/1e4)*exp(-(x/1e4))\n default: %(default)s'
)

parser.add_argument(
    '--power_spectrum_filename',default=None,
    help='Filename of input power spectrum, in case power_spectrum=-1'
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

CONSTANT_POWERSPECTRUM = False
ANALYTICAL_POWERSPECTRUM = False
ANALYTICAL_POWERSPECTRUM_V2 = False
power_spectrum_filename = None

if(args.power_spectrum==0):
    print("Using constant powerspectrum")
    CONSTANT_POWERSPECTRUM = True

if(args.power_spectrum==1):
    print("Using (x/1e4)^2*exp(-(x/1e4)^2) powerspectrum")
    ANALYTICAL_POWERSPECTRUM = True

if(args.power_spectrum==2):
    print("Using (x/1e4)*exp(-(x/1e4)) powerspectrum")
    ANALYTICAL_POWERSPECTRUM_V2 = True

if(args.power_spectrum<0):
    print("Using power spectrum from file: ",args.power_spectrum_filename)
    power_spectrum_filename = args.power_spectrum_filename



global_fieldsize_rad = global_fieldsize_deg*np.pi/180
global_fieldsize_arcmin = global_fieldsize_deg*60.

def aperture_mass_correlation_gaussian_random_field(power_spectrum,npix,thetas,random_seed,compute_gamma,compute_kappa,galaxy_density=None,shapenoise = 0.3):
    if(galaxy_density is None):
        if(power_spectrum_filename is None):
            kappa_field = create_gaussian_random_field(power_spectrum,n_pix=npix,fieldsize=global_fieldsize_rad,random_seed=random_seed)
        else:
            kappa_field = create_gaussian_random_field_array(power_spectrum[:,0],power_spectrum[:,1],
                                                        n_pix=npix,fieldsize=global_fieldsize_rad,random_seed=random_seed)
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

    result_gamma_second_order,result_gamma_third_order = extract_both_aperture_masses_of_field(shears,npix,thetas,global_fieldsize_arcmin) 


    measured_power_spectrum = extract_power_spectrum(shears,global_fieldsize_rad)
    return result_gamma_second_order,result_gamma_third_order,measured_power_spectrum

def aperture_mass_correlation_gaussian_random_field_kernel(kwargs):
    power_spectrum,final_result_gamma_second_order,final_result_gamma_third_order,npix,thetas,random_seed,realisation = kwargs
    result_gamma_second_order,result_gamma_third_order,measured_power_spectrum = aperture_mass_correlation_gaussian_random_field(power_spectrum,npix,thetas,random_seed,compute_gamma,compute_kappa)
    np.save("/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/powerspectrum_{}",measured_power_spectrum)

    final_result_gamma_second_order[:,realisation] = result_gamma_second_order
    final_result_gamma_third_order[:,realisation] = result_gamma_third_order

def compute_aperture_mass_correlations_of_gaussian_random_fields(power_spectrum,npix,thetas,n_realisations,n_processes=64):
    m = MyManager()
    m.start()
    n_theta = len(thetas)
    final_result_gamma_second_order = m.np_zeros((n_theta,n_realisations))
    final_result_gamma_third_order = m.np_zeros((n_theta*(n_theta+1)*(n_theta+2)//6,n_realisations))

    with Pool(processes=n_processes) as p:
        args = [[power_spectrum,final_result_gamma_second_order,final_result_gamma_third_order,npix,thetas,(i**3+250*i)%2**32,i] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(aperture_mass_correlation_gaussian_random_field_kernel,args),total=n_realisations):
            pass

    return final_result_gamma_second_order,final_result_gamma_third_order




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

    elif(power_spectrum_filename is not None):
        power_spectrum = np.loadtxt(power_spectrum_filename)
        savepath = '/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/SLICS_powerspectrum/'

    res_2pt,res_3pt = compute_aperture_mass_correlations_of_gaussian_random_fields(power_spectrum,n_pix,[2,4,8,16],args.realisations,n_processes=args.processes)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savename = 'npix_'+str(n_pix)+'_fieldsize_'+str(np.int(np.round(global_fieldsize_deg)))
    if(args.substract_mean):
        savename += '_mean_substracted'
    np.save(savepath+'map_squared_'+savename,res_2pt)
    np.save(savepath+'map_cubed_'+savename,res_3pt)



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
