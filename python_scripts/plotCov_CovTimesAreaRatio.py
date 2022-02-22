import numpy as np
import matplotlib.pyplot as plt

from helpers_plot import initPlot
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colorbar as mcb
import matplotlib.cm as cm
import argparse
from matplotlib.colors import LogNorm

description="""Script for plotting Cov*Area
"""

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--cov_type', default='none', type=str, help='Type of covariance that is plotted, can be either slics, shapenoise or cosmicShear, default: %(default)s')

parser.add_argument('--sigma', default=0.0, type=float, help='Shapenoise. default: %(default)s')

parser.add_argument('--dir', type=str, help='Directory with files, and output directory, default: %(default)s', default='./')

args=parser.parse_args()

initPlot(fontsize=30, labelsize=30)

cov_type = args.cov_type
sigma = args.sigma
folder= args.dir

if (cov_type != 'slics' and cov_type != 'shapenoise' and cov_type != 'cosmicShear'):
    print("Cov type not specified")
    exit

# Set Thetas labels
thetas_ind=np.array([[2, 2, 2], [2, 2, 4], [2, 2, 8], [2, 2, 16],
                        [2, 4, 4], [2, 4, 8], [2, 4, 16], [2, 8, 8], 
                        [2, 8, 16], [2, 16, 16], [4, 4, 4], [4, 4, 8], 
                        [4, 4, 16], [4, 8, 8], [4, 8, 16], [4, 16, 16], 
                        [8, 8, 8], [8, 8, 16], [8, 16, 16], [16, 16, 16] ])
thetas_labels=[]
for thetas in thetas_ind:
    thetas_labels.append(f"{thetas[0]}' {thetas[1]}' {thetas[2]}'")
N=len(thetas_ind)
thetas_ticks=np.arange(0, N)
theta=5
n = 4096.0*4096.0/theta/theta
thetaMax = theta-8*16/60
cov_term1Numerical_small = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
cov_term2Numerical_small = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
cov_infiniteField_small = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
cov_fft_small = np.loadtxt(folder+f'cov_cosmicShear_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
cov_term1Numerical_small=cov_infiniteField_small
thetaMaxSmallRad=thetaMax*np.pi/60


sidelengths=np.array([10, 15])
Nsides=len(sidelengths)
fig= plt.figure(figsize=(10*Nsides+2, 20+2))
cmap=cm.get_cmap('RdBu', 12)
grid=ImageGrid(fig, 111, nrows_ncols=(1, Nsides), axes_pad=0.15, share_all=True, aspect=True, cbar_location="right", cbar_mode="single", cbar_size="3%", cbar_pad=0.15)

# Set Xaxes labels


for i, theta in enumerate(sidelengths):
    n = 4096.0*4096.0/theta/theta
    thetaMax = theta-8*16/60
    grid[i].set_xlabel(r'$(\theta_4, \theta_5, \theta_6)$')
    grid[i].set_xticks(thetas_ticks)
    grid[i].set_xticklabels(thetas_labels, rotation=90)
    # Load data
    if (cov_type == 'slics'):
        cov_term2Numerical = np.loadtxt(folder+f'cov_slics_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
        cov_infiniteField = np.loadtxt(folder+f'cov_slics_infiniteField_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
        cov_fft = np.loadtxt(folder+f'cov_slics_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')*0.775 #Factor, is because 4*32 arcmin was cut off, not 4*16 arcmin
        cov_infiniteFieldNG = np.loadtxt(folder+f'cov_slics_infiniteFieldNG_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')

    elif (cov_type == 'shapenoise'):
        cov_term1Numerical = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')

        cov_term2Numerical = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
        cov_infiniteField = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
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

    cov_term1Numerical=cov_infiniteField
    thetaMaxRad=thetaMax*np.pi/60
    cov_measured=cov_term1Numerical+cov_term2Numerical
    cov_rescaled=(cov_term1Numerical_small+cov_term2Numerical_small)*thetaMaxSmallRad*thetaMaxSmallRad/thetaMaxRad/thetaMaxRad
    diff=2*(cov_measured-cov_rescaled)/(cov_measured+cov_rescaled)
    # Add plots

    # Set yLabel
    grid[i].set_ylabel(r'$(\theta_1, \theta_2, \theta_3)$')
    grid[i].set_yticks(thetas_ticks)
    grid[i].set_yticklabels(thetas_labels)



    im = grid[i].imshow(diff,  vmin=-1.5, vmax=1.5, cmap=cmap)  
    grid[i].text(19, 0, r"$\vartheta_\textrm{max}=$"+f"{thetaMax:.2f}°", verticalalignment='top', horizontalalignment='right',bbox=dict(facecolor='white', alpha=1))  


grid[Nsides-1].text(19, 19, cov_type, verticalalignment='bottom', horizontalalignment='right',bbox=dict(facecolor='white', alpha=1))  
grid[Nsides-1].cax.cla()
cb=mcb.Colorbar(grid[Nsides-1].cax, im)
cb.set_label(r"$2\,\frac{C_{\hat{M}_\mathrm{ap}^3}-C^\mathrm{resc}_{\hat{M}_\mathrm{ap}^3}}{C_{\hat{M}_\mathrm{ap}^3}+C^\mathrm{resc}_{\hat{M}_\mathrm{ap}^3}} $", fontsize=40)
plt.savefig(folder+f"CovTimesAreaRatio.png", facecolor="white", dpi=300)