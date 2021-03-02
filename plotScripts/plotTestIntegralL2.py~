import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

data= np.loadtxt("../tests/TestIntegralPhi.dat")

N=10
l1=np.logspace(-3, 4, N)
l2=np.logspace(-3, 4, N)
val=data[:,2]

fig, ax=plt.subplots(ncols=1, nrows=1, figsize=(2*5, 5))
plt.subplots_adjust(wspace=0,hspace=0)

val = val.reshape(N,N)
#    Z = np.transpose(Z)
val = np.flipud(val)
  
ax.set_xlabel(r'$l_1$')
ax.set_ylabel(r'$l_2$')

    # Set x and y ticks (needs to be changed by hand according to data!)
ax.set_xticks([1,5,9])
ax.set_xticklabels([f'{l1[0]:.2f}',  f'{l1[4]:.2f}',  f'{l1[8]:.2f}'])
ax.set_yticks([9,5,1])
ax.set_yticklabels([f'{l2[4]:.2f}', f'{l2[4]:.2f}', f'{l2[8]:.2f}'])

    # Set Cmap
my_cmap = copy.copy(cm.CMRmap) # copy the default cmap
my_cmap.set_bad(color='k', alpha=1.)
    
im= ax.imshow(val+1e-70, norm=LogNorm())#, norm=colors.LogNorm(vmin=1e-70, vmax=1e-20))
          

cbar_ax = fig.add_axes()
fig.colorbar(im, cax=cbar_ax)
plt.savefig("../tests/TestsIntegralPhi.png", dpi=300)
