import numpy as np
import matplotlib.pyplot as plt

from helpers_plot import initPlot
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colorbar as mcb
from matplotlib.colors import LogNorm


initPlot(titlesize=20)

cov_type = "cosmicShear" #"shapenoise"  # Can be 'slics', 'shapenoise' or 'cosmicShear'
sigma = 0.0
sidelength = 10  # in Deg!

if (cov_type == 'slics'):
    folder = "/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS/"
elif (cov_type == 'shapenoise'):
    folder = "/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_shapenoise/"
elif (cov_type == 'cosmicShear'):
    folder = "/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_cosmicShear/"
else:
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
n = 4096.0*4096.0/sidelength/sidelength
thetaMax = sidelength-8*16/60
if (cov_type == 'slics'):
    cov_term2Numerical = np.loadtxt(folder+f'cov_slics_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    cov_infiniteField = np.loadtxt(folder+f'cov_slics_infiniteField_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    cov_fft = np.loadtxt(folder+f'cov_slics_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')*0.775 #Factor, is because 4*32 arcmin was cut off, not 4*16 arcmin
    cov_infiniteFieldNG = np.loadtxt(folder+f'cov_slics_infiniteFieldNG_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')

elif (cov_type == 'shapenoise'):
    cov_term1Analytical = np.loadtxt(folder+f'cov_shapenoise_term1Analytical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    cov_term2Analytical = np.loadtxt(folder+f'cov_shapenoise_term2Analytical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    cov_term1Numerical = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    cov_term2Numerical = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    cov_infiniteField = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    cov_fft = np.loadtxt(folder+f'cov_shapenoise_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    covUncertainty_fft=np.loadtxt(folder+f'covUncertainty_shapenoise_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
elif (cov_type == 'cosmicShear'):
    cov_term1Numerical = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    cov_term2Numerical = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    cov_infiniteField = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
    cov_fft = np.loadtxt(folder+f'cov_cosmicShear_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    covUncertainty_fft=np.loadtxt(folder+f'covUncertainty_cosmicShear_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
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
im = grid[0].imshow(cov_fft, norm=LogNorm(vmin=1e-25, vmax=1e-19))  

grid[1].set_title(r"$C_{\hat{M}_\mathrm{ap}^3} = T_1+T_2$")     
im = grid[1].imshow(cov_term1Numerical+cov_term2Numerical, norm=LogNorm(vmin=1e-25, vmax=1e-19)) 

grid[2].set_title(r"$C_{\hat{M}_\mathrm{ap}^3}^\infty = T^\infty_1$")     
im = grid[2].imshow(cov_infiniteField, norm=LogNorm(vmin=1e-25, vmax=1e-19))  

grid[3].set_title(r"$T_1$")  
im = grid[3].imshow(cov_term1Numerical, norm=LogNorm(vmin=1e-25, vmax=1e-19))    
grid[4].set_title(r"$T_2$")  
im = grid[4].imshow(cov_term2Numerical, norm=LogNorm(vmin=1e-25, vmax=1e-19)) 


grid[4].text(19, 0, r"$\vartheta_\textrm{max}=$"+f"{thetaMax:.2f}°", verticalalignment='top', horizontalalignment='right',bbox=dict(facecolor='white', alpha=1))  
grid[4].text(19, 19, cov_type, verticalalignment='bottom', horizontalalignment='right',bbox=dict(facecolor='white', alpha=1))  
grid[4].cax.cla()
mcb.Colorbar(grid[4].cax, im)

plt.savefig(folder+f"all_covs_thetaMax_{thetaMax:.2f}.png", facecolor="white", dpi=300)
