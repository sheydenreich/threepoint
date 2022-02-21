import numpy as np
import pyccl as ccl
import pyccl.background as background
import matplotlib.pyplot as plt

h=0.6898
cosmo=ccl.Cosmology(Omega_c=0.2568, Omega_b=0.0473, h=h, n_s=0.96, sigma8=0.826)

k_arr=np.geomspace(1e-2, 1e2, 128)

hmd_200m=ccl.halos.MassDef200m()
cM=ccl.halos.ConcentrationDuffy08(hmd_200m)

z=0.2
a=1/(1+z)
m=1e14/0.6898

profile=ccl.halos.HaloProfileNFW(cM, truncated=True)
u=profile.fourier(cosmo, k_arr, m, a, mass_def=hmd_200m)/m

data=np.genfromtxt("/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS/test_nfw.dat")


plt.figure()
plt.title('NFW Profile in Fourier Space')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.01, 100)
plt.ylim(1e-3, 1)
plt.plot(k_arr/h, u*u*u, label="ccl")
plt.plot(data[:,0], data[:,1]*data[:,1]*data[:,1], label="threepoint")
plt.legend()
plt.show()

conc=cM.get_concentration(cosmo, m, a)
r200=hmd_200m.get_radius(cosmo, m, a)/a

rhobar=background.rho_x(cosmo, a, "matter")
print(r200*h)