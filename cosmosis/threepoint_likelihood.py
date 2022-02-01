# Quick and dirty cosmosis module for likelihood 
# (should be replaced by proper likelihood module in the future)

from cosmosis.datablock import names
import numpy as np

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
    calculate_Map2 = ("Map2" in likelihoods)
    calculate_Map3 = ("Map3" in likelihoods)
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

    data=np.loadtxt(fn_data, comments='#')
    cov=np.loadtxt(fn_cov, comments='#')
    N=len(data)
    cov=cov.reshape((N,N))
    cov_inv=np.linalg.inv(cov)

    if not np.allclose(np.dot(cov,cov_inv),np.eye(cov.shape[0]),atol=1e-8):
        raise ValueError("Error: Covariance Matrix not invertible")

    np.savetxt(fn_cov_inv, cov_inv)

    return [data, cov_inv, calculate_Map2, calculate_Map3, N_sim]
    


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
    
    [map3_data, cov_inv, calculate_Map2, calculate_Map3, N_sim]=config
    
    model = np.zeros(0)

    if(calculate_Map2):
        # print(block["threepoint", "Map2s"])
        model = np.concatenate((model,block["threepoint", "Map2s"]))

    if(calculate_Map3):
        # print(block["threepoint", "Map3s"])
        model = np.concatenate((model,block["threepoint", "Map3s"]))

    # print(model)
    delta = model - map3_data
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
