
from statistics import stdev
import numpy as np


theta=10
thetaMax=theta-8*16/60
n=30*60*60#4096.0**2/theta**2
Nlos=926 #4096
type="slics" #"cosmicShear"

folder = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/SLICS/" #"/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_"+type+"/"
filename_in = "map_cubed" #f"map_cubed_from_gamma_npix_4096_fieldsize_{theta}_{Nlos}.npy"
filename_out = f"cov_{type}_fft_sigma_0.26_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat"
filename_out_uncertainty = f"covUncertainty_{type}_fft_sigma_0.26_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat"

# folder = "/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_cosmicShear/"
# filename_in = "powerspectrum_SLICSmap_cubed_from_gamma_npix_4096_fieldsize_10.npy"
# filename_out = "cov_cosmicShear_fft_sigma_0.3_n_167772.16_thetaMax_8.93.dat"
# filename_out_uncertainty = "covUncertainty_cosmicShear_fft_sigma_0.3_n_167772.16_thetaMax_8.93.dat"


data = np.loadtxt(folder+filename_in)
#data=data[1:,1:,1:,:,:Nlos]
data=data.T
print(data.shape)

# N = len(data[0])
# ixs = []
# for i in range(N):
#     for j in range(i, N):
#         for k in range(j, N):
#             ix = i*N**2+j*N+k
#             ixs.append(ix)


# data = data.reshape(N*N*N, Nlos)
# data = data[ixs]

cov = np.cov(data)

np.savetxt(folder+filename_out, cov)

# covs = np.zeros((len(ixs), len(ixs), Nlos//256))
covs=np.zeros((20, 20, Nlos//256))

for i in range(Nlos//256):
    tmp = data[:, i*256:(i+1)*256]
    covs[:, :, i] = np.cov(tmp)

cov_uncertainty = np.std(covs, axis=2)

np.savetxt(folder+filename_out_uncertainty, cov_uncertainty)
print(cov_uncertainty.shape)
