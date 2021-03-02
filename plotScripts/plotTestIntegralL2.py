import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

data= np.loadtxt("../tests/TestIntegralL2.dat")

plt.xscale('log')
plt.yscale('log')
plt.plot(data[:,0], data[:,1])
plt.xlabel('$l_1$')
plt.savefig("../tests/TestsIntegralL2.png", dpi=300)
