import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

data= np.loadtxt("../tests/TestMapMapMapCubature.dat")

N=5
thetas=np.array([2,4,8,16,32])
val=data[:,3]

fig, ax=plt.subplots(ncols=N, nrows=1, figsize=((N+1)*5, 5), sharey=True, sharex=True)
plt.subplots_adjust(wspace=0,hspace=0)

for i in range(0, N):
    phi=data[i][2]
    X=data[i::N,0]
    Y=data[i::N,1]
    Z=data[i::N,3]
    
    Z = Z.reshape(N,N)
#    Z = np.transpose(Z)
    Z = np.flipud(Z)
  
    ax[i].set_xlabel(r'$\theta_1$ [arcmin]')
    ax[i].set_ylabel(r'$\theta_2$ [arcmin]')
    ax[i].text(0,0,r'$\theta_3=$%.1f arcmin'%(phi), bbox=dict(facecolor='white'))

    # Set x and y ticks (needs to be changed by hand according to data!)
    ax[i].set_xticks([0,1,2,3,4])
    ax[i].set_xticklabels([f'{thetas[0]:.1f}', f'{thetas[1]:.1f}',  f'{thetas[2]:.1f}', f'{thetas[3]:.1f}',  f'{thetas[4]:.1f}'])

    # Set Cmap
    my_cmap = copy.copy(cm.CMRmap) # copy the default cmap
    my_cmap.set_bad(color='k', alpha=1.)
    
    im= ax[i].imshow(Z, norm=LogNorm())#, norm=colors.LogNorm(vmin=1e-70, vmax=1e-20))
          
ax[0].set_yticks([4,3,2,1,0])
ax[0].set_yticklabels([f'{thetas[0]:.1f}', f'{thetas[1]:.1f}',  f'{thetas[2]:.1f}', f'{thetas[3]:.1f}',  f'{thetas[4]:.1f}'])    
cbar_ax = fig.add_axes()
fig.colorbar(im, cax=cbar_ax)
plt.savefig("../tests/MapMapMapCubature.png", dpi=300)
