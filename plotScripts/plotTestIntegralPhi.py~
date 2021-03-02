import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

data= np.loadtxt("../tests/TestIntegrand.dat")

N=10
l1=np.logspace(-3, 4, N)
l2=np.logspace(-3, 4, N)
phi=np.linspace(1e-4, 2*np.pi-1e-4, N)
val=data[:,3]

fig, ax=plt.subplots(ncols=10, nrows=1, figsize=(11*5, 5), sharey=True, sharex=True)
plt.subplots_adjust(wspace=0,hspace=0)

for i in range(0, N):
    phi=data[i][2]
    X=data[i::N,0]
    Y=data[i::N,1]
    Z=data[i::N,3]
    
    Z = Z.reshape(N,N)
#    Z = np.transpose(Z)
    Z = np.flipud(Z)
  
    ax[i].set_xlabel(r'$l_1$')
    ax[i].set_ylabel(r'$l_2$')
    ax[i].text(0.5,0.5,r'$\phi=$%.3f'%(phi), bbox=dict(facecolor='white'))

    # Set x and y ticks (needs to be changed by hand according to data!)
    ax[i].set_xticks([1,5,9])
    ax[i].set_xticklabels([f'{l1[0]:.2f}',  f'{l1[4]:.2f}',  f'{l1[8]:.2f}'])
    ax[0].set_yticks([9,5,1])
    ax[0].set_yticklabels([f'{l2[4]:.2f}', f'{l2[4]:.2f}', f'{l2[8]:.2f}'])

    # Set Cmap
    my_cmap = copy.copy(cm.CMRmap) # copy the default cmap
    my_cmap.set_bad(color='k', alpha=1.)
    
    im= ax[i].imshow(Z+1e-70, norm=LogNorm())#, norm=colors.LogNorm(vmin=1e-70, vmax=1e-20))
          

cbar_ax = fig.add_axes()
fig.colorbar(im, cax=cbar_ax)
plt.savefig("../tests/TestsIntegrand.png", dpi=300)
