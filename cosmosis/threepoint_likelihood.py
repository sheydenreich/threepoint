# Quick and dirty cosmosis module for likelihood 
# (should be replaced by proper likelihood module in the future)
from __future__ import division

from sympy import maximum

from cosmosis.datablock import names
import numpy as np

def create_masks(n_theta):
    n_theta_3pt = n_theta*(n_theta+1)*(n_theta+2)//6
    n_theta_joint = n_theta+n_theta_3pt

    masks = {}

    mask_2pt = np.zeros(n_theta_joint,dtype=bool)
    mask_2pt[:n_theta] = True
    masks["map2"] = mask_2pt

    mask_3pt = np.zeros(n_theta_joint,dtype=bool)
    mask_3pt[n_theta:] = True
    masks["map3"] = mask_3pt

    mask_3pt_diag = np.zeros(n_theta_joint,dtype=bool)
    counter = n_theta
    for i in range(n_theta):
        for j in range(i,n_theta):
            for k in range(j,n_theta):
                if(i==j==k):
                    mask_3pt_diag[counter] = True
                counter += 1
    masks["map3_diag"] = mask_3pt_diag

    masks["joint"] = np.ones(n_theta_joint,dtype=bool)

    return masks

def create_mask_factors(maximum_bin_factor,n_theta,remove_64):
    mask = np.ones(n_theta+n_theta*(n_theta+1)*(n_theta+2)//6,dtype=bool)
    counter = n_theta
    if(maximum_bin_factor==0):
        maximum_bin_factor = 10**n_theta
    for i in range(n_theta):
        for j in range(i,n_theta):
            for k in range(j,n_theta):
                if(2**k/2**i>maximum_bin_factor):
                    mask[counter]=False
                if(remove_64 and k==n_theta-1):
                    mask[counter]=False
                counter += 1
    return mask

def setup(options):
    """Setup function for cosmosis. Reads in Covariance and measurements and inverts covariance (using numpy)
    Args:
        options (datablock): Contains everything given in the pipeline.ini
    Returns:
        [data, cov_inv]: List containing measurements (as np.array) and inverse covariance (as np.array)
    """
    fn_data=options["threepoint_likelihood", "data"]
    fn_cov=options["threepoint_likelihood", "cov_mat"]
    fn_cov_inv=options["threepoint_likelihood", "cov_inv"]
    likelihoods = options["threepoint_likelihood","likelihoods"]

    maximum_bin_factor = options["threepoint_likelihood","maximum_bin_factor"]
    mask_maximum_bin_factor = None

    n_theta = options["threepoint_likelihood","theta_bins"]



    calculate_Map2 = ("Map2" in likelihoods)
    calculate_Map3 = ("Map3" in likelihoods)
    calculate_gamma = ("gamma" in likelihoods)

    account_for_emulator_error = options.get_bool("threepoint_likelihood","account_for_emulator_error",
                                                    default=False)
    
    account_for_cutoff = options.get_bool("threepoint_likelihood","account_for_cutoff",
                                          default=False)

    remove_64 = options.get_bool("threepoint_likelihood","remove_64",default=False)

    N_sim = options["threepoint_likelihood", "N_sim"]

    print("Using covariance matrix from "+fn_cov)
    print("Using data vector from "+fn_data)

    if(N_sim>0):
        print("Sample covariance matrix extracted from {} Simulations".format(N_sim))
        print("Using Sellentin+Heavens likelihood")
    else:
        print("Analytic covariance matrix")
        print("Using gaussian likelihood")

    if(calculate_Map2):
        print("Accounting for Map2 likelihood")

    if(calculate_Map3):
        print("Accounting for Map3 likelihood")
    try:
        data=np.loadtxt(fn_data, comments='#')
    except Exception as e:
        print("Could not load {}: {}".format(fn_data,e))
        print("Trying as numpy format.")
        data = np.load(fn_data)

    try:
        cov=np.loadtxt(fn_cov, comments='#')
    except Exception as e:
        print("Could not load {}: {}".format(fn_cov,e))
        print("Trying as numpy format.")
        cov = np.load(fn_cov)
        
    N=len(data)
    try:
        cov=cov.reshape((N,N))

    except Exception as e:
        print("Could not reshape covariance. Trying again after removing elements from datavector.")
        mask = np.zeros(N,dtype = bool)
        counter = 0
        for i in range(n_theta+1):
            for j in range(i,n_theta+1):
                for k in range(j,n_theta+1):
                    if(k!=n_theta):
                        mask[counter]=True
                    counter += 1
        data = data[mask]
        N = len(data)
        cov=cov.reshape((N,N))

    if(options.get_bool("threepoint_likelihood","diag_only",default=False)):
        print("Adjusting data vector to only diagonal entries!")
        masks = create_masks(n_theta)
        mask_64 = create_mask_factors(0,n_theta,remove_64)
        mymask = masks["map3_diag"]
        mymask[:n_theta] = True
        mask_maximum_bin_factor = mask_64[mymask]

        mymask = np.logical_and(mymask,mask_64)
        data = data[mymask]
        cov = cov[:,mymask]
        cov = cov[mymask]

    else:

        mask_maximum_bin_factor = create_mask_factors(maximum_bin_factor,n_theta,remove_64)
        print(mask_maximum_bin_factor)
        data = data[mask_maximum_bin_factor]
        cov = cov[:,mask_maximum_bin_factor]
        cov = cov[mask_maximum_bin_factor]
        print("Removed maximum bin factor. Number of entries remaining: {}".format(len(data)-n_theta))

    if(account_for_emulator_error):
        print("Accounting for Emulator uncertainty")
        fn_cov_split = fn_cov.split(".")
        cov_add = np.loadtxt(fn_cov_split[0]+"_emulator."+fn_cov_split[1])
        cov += cov_add
        
    if(account_for_cutoff):
        fieldsize = options["threepoint_likelihood","fieldsize"]
        theta_max = options["threepoint_likelihood","theta_max"]
        fieldsize_new = fieldsize-8*theta_max
        modification_factor = fieldsize_new**2/fieldsize**2
        print("Decreasing covariance matrix to account for cut-off of boundary.")
        print("Previous fieldsize: {}, new fieldsize: {}, modification factor: {}".format(fieldsize,fieldsize_new,modification_factor))
        cov *= modification_factor

    if not calculate_Map2:
        print("Removing Map2 from data")
        data = np.delete(data,np.arange(n_theta))
        cov = np.delete(cov,np.arange(n_theta),axis=0)
        cov = np.delete(cov,np.arange(n_theta),axis=1)
        try:
            mask_maximum_bin_factor = np.delete(mask_maximum_bin_factor,np.arange(n_theta))
        except Exception:
            pass
    


    cov_inv=np.linalg.inv(cov)

    if not np.allclose(np.dot(cov,cov_inv),np.eye(cov.shape[0]),atol=1e-8):
        raise ValueError("Error: Covariance Matrix not invertible")

    # np.savetxt(fn_cov_inv, cov_inv)
    

    return [data, cov_inv, calculate_Map2, calculate_Map3, calculate_gamma, N_sim, mask_maximum_bin_factor]
    


def execute(block, config):
    """Execute function for cosmosis. Takes measurements, covs and model values and calculates\
        Gaussian (log)likelihood
    Args:
        block ([datablock]): Main datablock containing output from previous modules and values.ini
        config (list): output of setup, config[0] are the measurements and config[1] is the inverse\
            covariance
    Returns:
        0: designates success (needed for cosmosis reasons)
    """
    
    [data, cov_inv, calculate_Map2, calculate_Map3, calculate_gamma, N_sim, mask_maximum_bin_factor]=config
    
    model = np.zeros(0)

    if(calculate_Map2):
        # print(block["threepoint", "Map2s"])
        model = np.concatenate((model,block["threepoint", "Map2s"]))

    if(calculate_Map3):
        # print(block["threepoint", "Map3s"])
        model = np.concatenate((model,block["threepoint", "Map3s"]))

    if(calculate_gamma):
        model = np.concatenate((model,block["threepoint", "pc_gamma"]))

    # print(model.shape,data.shape)
    # import sys
    # sys.exit()
    if(mask_maximum_bin_factor is not None):
        model = model[mask_maximum_bin_factor]

    delta = model - data
    chi2=np.einsum("i,ij,j",delta,cov_inv,delta)
    log_det=0#1.0/np.log(np.linalg.det(cov_inv))

    if(N_sim>0):
        likelihood = -0.5 * log_det - 0.5 * N_sim * np.log(1 + chi2 / (N_sim - 1.))
        # print("Likelihood: ",likelihood)
    else:
        likelihood=-0.5*(chi2+log_det)

    block [names.likelihoods, "threepoint_likelihood_LIKE"] = likelihood

    return 0


def cleanup(config):
    """Does nothing, since python is a cool language with a garbage collector :)
    Args:
        config: output of setup
    """
    pass