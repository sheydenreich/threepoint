from utility import create_lognormal_random_field, extract_power_spectrum, get_gaussian_power_spectrum
import numpy as np
from tqdm import tqdm
import multiprocessing.managers
from multiprocessing import Pool
import scipy.integrate as integrate
from scipy.interpolate import interp1d

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)



def compute_power_spectra_of_lognormal_random_fields(alpha,nbins,n_realisations,n_processes,input_powerspectrum,sigma=None):
    m = MyManager()
    m.start()
#     n_theta = len(thetas)
    final_results = m.np_zeros((nbins,n_realisations))

    if sigma is None:
        sigma = np.sqrt(integrate.quad(lambda ell: ell*input_powerspectrum(ell),10,10**5)[0]/(2.*np.pi))

    with Pool(processes=n_processes) as p:
        args = [[final_results,alpha,input_powerspectrum,nbins,i,sigma] for i in range(n_realisations)]
        for i in tqdm(p.imap_unordered(power_spectrum_of_lognormal_field_kernel,args),total=n_realisations):
#         for i in p.imap_unordered(power_spectrum_of_lognormal_field_kernel,args):

            pass
    return final_results

def power_spectrum_of_lognormal_field_kernel(kwargs):
    results,alpha,input_powerspectrum,nbins,random_seed,sigma = kwargs
    field = create_lognormal_random_field(input_powerspectrum,alpha,random_seed=random_seed,sigma=sigma)
    ps,kbins = extract_power_spectrum(field,10.*np.pi/180,bins=nbins)
    results[:,random_seed] = ps
    # print(np.min(field),np.max(field),alpha)
    if(random_seed==0):
        print(kbins)
        print(ps)


if(__name__=="__main__"):
    # read reference power spectrum
    ps_slics = np.loadtxt("/users/sven/public_html/public_data/plots_fisher_HOWLS/power_spectra/model_spectra/powerspectrum_SLICS_fine.dat")
    ells = ps_slics[:,0]
    vals = ps_slics[:,1]

    psfunc_slics = interp1d(ells,vals,fill_value=0,bounds_error=False)

    # standard-deviation of GRF = correlation function at zero
    sigma = np.sqrt(integrate.quad(lambda ell: ell*psfunc_slics(ell),10,10**5)[0]/(2.*np.pi))

    ells = np.geomspace(10,10000)

    # generate 'gaussianized' power spectra
    alphas = [0.5,0.7,0.9]
    all_psg = {}
    for alpha in tqdm(alphas):
        # print(alpha)
        psg = get_gaussian_power_spectrum(psfunc_slics,alpha,sigma)
        all_psg[str(alpha)] = psg

    # create lognormal random fields, extract their power spectra, profit
    all_power_spectra = {}
    for alpha in alphas:
        res = compute_power_spectra_of_lognormal_random_fields(alpha,30,128,64,all_psg[str(alpha)],sigma)
        all_power_spectra[str(alpha)] = res
        np.save("../../results/power_spectrum_extracted_alpha_{}".format(alpha),res)
    