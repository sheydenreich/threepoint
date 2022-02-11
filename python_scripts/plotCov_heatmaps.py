import numpy as np
import matplotlib.pyplot as plt

from helpers_plot import initPlot
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colorbar as mcb
from matplotlib.colors import LogNorm

import argparse

description="""Script for plotting C_meas, T_1^\infty, T_1, T_2, and T_1+T_2 for one theta_max as heatmaps
"""

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--cov_type', default='none', type=str, help='Type of covariance that is plotted, can be either slics, shapenoise or cosmicShear, default: %(default)s')

parser.add_argument('--sigma', default=0.0, type=float, help='Shapenoise. default: %(default)s')

parser.add_argument('--dir', type=str, help='Directory with files, and output directory, default: %(default)s', default='./')

parser.add_argument('--sidelength', default=10, type=float, help='Sidelength of field (without cut) in deg, default: %(default)s')
args=parser.parse_args()

initPlot(titlesize=20)

cov_type = args.cov_type
sigma = args.sigma
folder= args.dir
sidelength=args.sidelength

if (cov_type != 'slics' and cov_type != 'shapenoise' and cov_type != 'cosmicShear'):
    print("Cov type not specified")
    exit

# Set Thetas labels
thetas_ind = np.array([[2, 2, 2], [2, 2, 4], [2, 2, 8], [2, 2, 16],
                       [2, 4, 4], [2, 4, 8], [2, 4, 16], [2, 8, 8],
                       [2, 8, 16], [2, 16, 16], [4, 4, 4], [4, 4, 8],
                       [4, 4, 16], [4, 8, 8], [4, 8, 16], [4, 16, 16],
                       [8, 8, 8], [8, 8, 16], [8, 16, 16], [16, 16, 16]])
thetas_labels = []
for thetas in thetas_ind:
    thetas_labels.append(f"{thetas[0]}' {thetas[1]}' {thetas[2]}'")
N = len(thetas_ind)
thetas_ticks = np.arange(0, N)

# load data
n=4096.0*4096.0/sidelength/sidelength
thetaMax = sidelength-8*16/60
if (cov_type == 'slics'):
    sigma=0.26
    n=108000.00
    thetaMax=7.87
    # cov_term1Numerical = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    # cov_term2Numerical = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    cov_infiniteField = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    # cov_term4Numerical = np.loadtxt(folder+f'cov_infinite_term4Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    cov_fft = np.loadtxt(folder+f'cov_slics_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')

elif (cov_type == 'shapenoise'):
    cov_term1Numerical = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    cov_term2Numerical = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    cov_infiniteField = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    cov_fft = np.loadtxt(folder+f'cov_shapenoise_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
elif (cov_type == 'cosmicShear'):
    cov_term1Numerical = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    cov_term2Numerical = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    cov_infiniteField = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    cov_fft = np.loadtxt(folder+f'cov_cosmicShear_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
else:
    print("Cov type not specified")
    exit


# Do Plot
fig= plt.figure(figsize=(25, 10))

grid=ImageGrid(fig, 111, nrows_ncols=(1, 5), axes_pad=0.15, share_all=True, cbar_location="right", cbar_mode="single", cbar_size="7%", cbar_pad=0.15)

grid[0].set_ylabel(r'$(\theta_1, \theta_2, \theta_3)$')
grid[0].set_yticks(thetas_ticks)
grid[0].set_yticklabels(thetas_labels)

grid[0].set_xlabel(r'$(\theta_4, \theta_5, \theta_6)$')
grid[0].set_xticks(thetas_ticks)
grid[0].set_xticklabels(thetas_labels, rotation=90)

grid[1].set_xlabel(r'$(\theta_4, \theta_5, \theta_6)$')
grid[1].set_xticks(thetas_ticks)
grid[1].set_xticklabels(thetas_labels, rotation=90)

grid[2].set_xlabel(r'$(\theta_4, \theta_5, \theta_6)$')
grid[2].set_xticks(thetas_ticks)
grid[2].set_xticklabels(thetas_labels, rotation=90)

grid[3].set_xlabel(r'$(\theta_4, \theta_5, \theta_6)$')
grid[3].set_xticks(thetas_ticks)
grid[3].set_xticklabels(thetas_labels, rotation=90)

grid[4].set_xlabel(r'$(\theta_4, \theta_5, \theta_6)$')
grid[4].set_xticks(thetas_ticks)
grid[4].set_xticklabels(thetas_labels, rotation=90)

grid[0].set_title(r"$C_{\hat{M}_\mathrm{ap}^3}^\mathrm{meas}$")
im = grid[0].imshow(cov_fft, norm=LogNorm(vmin=1e-23, vmax=1e-16))  

#grid[1].set_title(r"$C_{\hat{M}_\mathrm{ap}^3} = T_1+T_2$")     
#im = grid[1].imshow(cov_term1Numerical+cov_term2Numerical, norm=LogNorm(vmin=1e-25, vmax=1e-19)) 

grid[2].set_title(r"$C_{\hat{M}_\mathrm{ap}^3}^\infty = T^\infty_1$")     
im = grid[2].imshow(cov_infiniteField, norm=LogNorm(vmin=1e-23, vmax=1e-16))  

grid[3].set_title(r"$T_1$")  
#im = grid[3].imshow(cov_term1Numerical, norm=LogNorm(vmin=1e-25, vmax=1e-19))    
grid[4].set_title(r"$T_2$")  
#im = grid[4].imshow(cov_term2Numerical, norm=LogNorm(vmin=1e-25, vmax=1e-19)) 


grid[4].text(19, 0, r"$\vartheta_\textrm{max}=$"+f"{thetaMax:.2f}°", verticalalignment='top', horizontalalignment='right',bbox=dict(facecolor='white', alpha=1))  
grid[4].text(19, 19, cov_type, verticalalignment='bottom', horizontalalignment='right',bbox=dict(facecolor='white', alpha=1))  
grid[4].cax.cla()
mcb.Colorbar(grid[4].cax, im)

plt.savefig(folder+f"all_covs_thetaMax_{thetaMax:.2f}.png", facecolor="white", dpi=300)
#plt.show()