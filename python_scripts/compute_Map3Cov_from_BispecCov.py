import numpy as np


def uhat(eta):
    temp = 0.5 * eta * eta
    return temp * np.exp(-temp)

def LambdaSq(ell1, ell2, ell3):
    result = 2 * ell1 * ell1 * ell2 * ell2 + 2 * ell1 * ell1 * ell3 * ell3 + 2 * ell2 * ell2 * ell3 * ell3 - np.power(ell1, 4) - np.power(ell2, 4) - np.power(ell3, 4)
    #print(ell1, ell2, ell3, result)
    if result<=0:
        return 0
    result = 16. / result
    return result

def CovMap3(thetas_123, thetas_456, CovB, ells, dElls):
        
    result = 0
    Nells=len(dElls)

    for i, dEll1 in enumerate(dElls):
        ell1=ells[i]
        u1=uhat(ell1*thetas_123[0])
        u4=uhat(ell1*thetas_456[0])
        for j, dEll2 in enumerate(dElls):
            ell2=ells[j]
            u2=uhat(ell2*thetas_123[1])
            u5=uhat(ell2*thetas_456[1])
            for k, dEll3 in enumerate(dElls):
                ell3=ells[k]
                u3=uhat(ell3*thetas_123[2])
                u6=uhat(ell3*thetas_456[2])

                ix=(i)*Nells*Nells+(j)*Nells+(k)
                covB=CovB[ix]

                l=LambdaSq(ell1, ell2, ell3)

                tmp=1
                tmp*=ell1*ell2*ell3
                tmp*=ell1*ell2*ell3
                tmp*=dEll1*dEll1*dEll2*dEll2*dEll3*dEll3
                tmp*=u1*u2*u3*u4*u5*u6
                tmp*=covB*l
                result+=tmp

    
    result/=np.power(2*np.pi, 6)

    return result


out_folder="/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS/"

fn_cov_bispec=out_folder+"model_cov_bispectrum_approx_degenerate_triangles_scalecuts.dat"
data= np.loadtxt(fn_cov_bispec)

cov_bispec=data[:,3]

logLmin=np.log10(66.8505)
logLmax=4
Nbins=26
deltaEll=(logLmax-logLmin)/(Nbins-1)



ells=np.array([np.power(10, (i+0.5)*deltaEll) for i in range(Nbins)])*np.power(10, logLmin)

dElls=np.array([np.power(10, (i+1)*deltaEll)-np.power(10, (i)*deltaEll) for i in range(Nbins)])*np.power(10, logLmin)

print(ells)
print(dElls)

thetas=np.array([[2, 2, 2], [2, 2, 4], [2, 2, 8], [2, 2, 16], [2, 4, 4], [2, 4, 8], [2, 4, 16], [2, 8, 8], [2, 8, 16], [2, 16, 16], [4, 4, 4], [4, 4, 8], [4, 4, 16], [4, 8, 8], [4, 8, 16], [4, 16, 16], [8, 8, 8], [8, 8, 16], [8, 16, 16], [16, 16, 16]])
thetas=thetas/60*np.pi/180
Nthetas=len(thetas)

cov_Map3=np.zeros((Nthetas, Nthetas))

for i, theta_123 in enumerate(thetas):
    for j, theta_456 in enumerate(thetas):
        cov_Map3[i, j] = CovMap3(theta_123, theta_456, cov_bispec, ells, dElls)
        print(theta_123, theta_456, cov_Map3[i,j])
    
fn_cov_map3=out_folder+"cov_map3_from_bispec_cov.dat"
np.savetxt(fn_cov_map3, cov_Map3)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

im=plt.imshow(cov_Map3, norm=LogNorm(vmin=1e-24, vmax=1e-17))
plt.colorbar(im)
plt.savefig(out_folder+"cov_map3_from_bispec_cov.png")
plt.show()