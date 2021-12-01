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

    data=np.loadtxt(fn_data, comments='#')[:,3]
    cov=np.loadtxt(fn_cov, comments='#')[:,-1]
    N=len(data)
    cov=cov.reshape((N,N))
    cov_inv=np.linalg.inv(cov)

    np.savetxt(fn_cov_inv, cov_inv)

    return [data, cov_inv]
    


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
    
    [map3_data, cov_inv]=config
    

    map3_model=block["threepoint", "Map3s"]

    chi2=(map3_model-map3_data).dot(cov_inv).dot(map3_model-map3_data)
    log_det=0#1.0/np.log(np.linalg.det(cov_inv))

    likelihood=-0.5*(chi2+log_det)

    block [names.likelihoods, "threepoint_likelihood_LIKE"] = likelihood

    return 0


def cleanup(config):
    """Does nothing, since python is a cool language with a garbage collector :)

    Args:
        config: output of setup
    """
    pass
