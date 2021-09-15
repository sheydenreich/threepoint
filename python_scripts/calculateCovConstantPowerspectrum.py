import numpy as np
from cov_constantPowerspectrum import cov_constantPowerspectrum
import sys

""" Script for calculating the <Map3> covariance of a field with constant powerspectrum
"""



# Set survey characteristics

sigma=0.3 # Shapenoise
n = 46.6 # source galaxy density [arcmin^-2]
sidelength= 536 # Sidelength [arcmin]

A = sidelength*sidelength # Survey area [arcmin^2]
unit="arcmin" # Angular units of n and A

# Output File
fn_out=f"../../Covariance_randomField/results/covariance_analytic_{sigma:.1f}_{n:.1f}_{sidelength:.0f}.dat"

# Set thetas
thetas_1d=np.array([1, 2, 4, 8, 16])
unit_thetas="arcmin"

thetas1, thetas2, thetas3, thetas4, thetas5, thetas6=np.meshgrid(thetas_1d, thetas_1d, thetas_1d, thetas_1d, thetas_1d, thetas_1d)
thetas=np.c_[thetas1.ravel(), thetas2.ravel(), thetas3.ravel(), thetas4.ravel(), thetas5.ravel(), thetas6.ravel()]


# Initialize calculator

Cov = cov_constantPowerspectrum(sigma, n, A, unit=unit)
result_cov=np.zeros(len(thetas))


# Do calculation
for i, theta in enumerate(thetas):
    result_cov[i]=Cov.covariance(theta, unit=unit_thetas)


# Output
np.savetxt(fn_out, np.column_stack((thetas, result_cov)))
