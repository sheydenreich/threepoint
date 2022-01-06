import numpy as np
from numpy.core.numeric import Inf
from cov_constantPowerspectrum_finiteField import cov_constantPowerspectrum_finiteField
from cov_constantPowerspectrum import cov_constantPowerspectrum

thetaMax = 8.93 # Fieldlength [deg]
sigma = 0.3 # Shapenoise (2 components)
n = 4096*4096/10/10 # number density [1/deg²]
folder="/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields/"


thetas=np.array([[2, 2, 2], [2, 2, 4], [2, 2, 8], [2, 2, 16],
                [2, 4, 4], [2, 4, 8], [2, 4, 16], [2, 8, 8], 
                [2, 8, 16], [2, 16, 16], [4, 4, 4], [4, 4, 8], 
                [4, 4, 16], [4, 8, 8], [4, 8, 16], [4, 16, 16], 
                [8, 8, 8], [8, 8, 16], [8, 16, 16], [16, 16, 16] ]) # Aperture scale radii [arcmin]

N=len(thetas) # Number of aperture scale radius combinations

Term1=np.zeros((N, N)) # First Term, using Eq. 100
Term2=np.zeros((N, N)) # Second Term, using Eq. 103
InfiniteField=np.zeros((N, N)) # Cov for infinite field, using Eq. 29

covarianceCalculator = cov_constantPowerspectrum(sigma, n, thetaMax*thetaMax, unit='deg')
covarianceCalculatorFiniteField =cov_constantPowerspectrum_finiteField(sigma, n, thetaMax*thetaMax, unit='deg')

for i, thetas1 in enumerate(thetas):
    for j, thetas2 in enumerate(thetas):
        Term1[i,j]=covarianceCalculatorFiniteField.term1_total(thetas=np.array([thetas1[0],thetas1[1], thetas1[2], thetas2[0],thetas2[1], thetas2[2]]), unit="arcmin")
        Term2[i,j]=covarianceCalculatorFiniteField.term2_total(thetas=np.array([thetas1[0],thetas1[1], thetas1[2], thetas2[0],thetas2[1], thetas2[2]]), unit="arcmin")
        InfiniteField[i,j]=covarianceCalculator.covariance(thetas=np.array([thetas1[0],thetas1[1], thetas1[2], thetas2[0],thetas2[1], thetas2[2]]), unit="arcmin")


np.savetxt(folder+f"cov_shapenoise_term1Analytical_sigma_{sigma}_n_{n}_thetaMax_{thetaMax}.dat", Term1)
np.savetxt(folder+f"cov_shapenoise_term2Analytical_sigma_{sigma}_n_{n}_thetaMax_{thetaMax}.dat", Term2)
np.savetxt(folder+f"cov_shapenoise_infiniteField_sigma_{sigma}_n_{n}_thetaMax_{thetaMax}.dat", InfiniteField)