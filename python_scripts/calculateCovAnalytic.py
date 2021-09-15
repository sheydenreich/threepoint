import numpy as np
from cov_analytic import cov_analytic
import sys

""" Script for calculating the <Map3> covariance of a gaussian field with powerspectrum P(l)=p1*l^2*exp(-p2*l^2)
"""



# Set survey characteristics

p1=1e-8 # Parameter p1 of Powerspectrum
p2 =1e-8 # Parameter p2 of Powerspectrum
sidelength= 536 # Sidelength [arcmin]

A = sidelength*sidelength # Survey area [arcmin^2]
unit="arcmin" # Angular units of A

# Output File
fn_out=f"../../Covariance_randomField/results/covariance_analytic_p1_{p1:.2e}_p2_{p2:.2e}_side_{sidelength:.0f}.dat"
print("Outputfile:", fn_out)

# Set thetas
thetas_1d=np.array([1, 2, 4, 8, 16])
unit_thetas="arcmin"

thetas1, thetas2, thetas3, thetas4, thetas5, thetas6=np.meshgrid(thetas_1d, thetas_1d, thetas_1d, thetas_1d, thetas_1d, thetas_1d)
thetas=np.c_[thetas1.ravel(), thetas2.ravel(), thetas3.ravel(), thetas4.ravel(), thetas5.ravel(), thetas6.ravel()]


# Initialize calculator

Cov = cov_analytic(p1, p2, A, unit=unit)
result_cov=np.zeros(len(thetas))


# Do calculation
for i, theta in enumerate(thetas):
    result_cov[i]=Cov.covariance(theta, unit=unit_thetas)


# Output
np.savetxt(fn_out, np.column_stack((thetas, result_cov)))
