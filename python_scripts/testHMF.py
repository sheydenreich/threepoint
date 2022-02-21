import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

h=0.6898

cosmo=ccl.Cosmology(Omega_c=0.2568, Omega_b=0.0473, h=0.6898, n_s=0.96, sigma8=0.826)

# # Cosmology
# cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
#                       h=0.67, A_s=2.1e-9, n_s=0.96)

m_arr = np.geomspace(1E11,1E16,128)
z_arr=[0.0]

plt.figure()
plt.xlim(np.power(10, 11), 1e16)
plt.ylim(1e-27, 1e-12)
plt.xscale('log')
plt.yscale('log')

data=np.genfromtxt("/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS/test_hmf.dat")
for z in z_arr:
    a=1/(1.0+z)
    hmf=ccl.halos.MassFuncSheth99(cosmo)
    dndlogm=hmf.get_mass_function(cosmo, m_arr, a) # dn/dlog10(m) [1/Mpc³]
    dndm=dndlogm/m_arr/np.log(10) #dn/dm [1/Msun/Mpc³]

    m_arr_h=m_arr*h #m [Msun h^-1]
    dndm_h=dndm/np.power(h,4) # dn/dm [h⁴/Msun/Mpc³]
    plt.plot(data[:,0], data[:,1], ls='', marker='+')
    plt.plot(m_arr_h, dndm_h, label=f'z={z}')

plt.legend()
plt.show()


plt.figure()
plt.xlim(np.power(10, 11), 1e16)
plt.ylim(0, 16)
plt.xscale('log')
for z in z_arr:
    a=1/(1.0+z)
    hmf=ccl.halos.HaloBiasSheth99(cosmo)
    bias=hmf.get_halo_bias(cosmo, m_arr, a) 

    m_arr_h=m_arr*h #m [Msun h^-1]

    plt.plot(m_arr_h, bias, label=f'z={z}')
plt.legend()
plt.show()