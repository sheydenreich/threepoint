import numpy as np
import matplotlib.pyplot as plt

from helpers_plot import initPlot
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colorbar as mcb
import matplotlib.cm as cm
import argparse

description="""Script for plotting the difference between the measured Covariance and the 
Model-Covariance as a heatmap
Outputs the fractional difference between C_meas and T_1^\infty, and the fractional difference between C_meas and T_1+T_2
"""

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--cov_type', default='none', type=str, help='Type of covariance that is plotted, can be either slics, shapenoise or cosmicShear, default: %(default)s')

parser.add_argument('--sigma', default=0.0, type=float, help='Shapenoise. default: %(default)s')

parser.add_argument('--dir', type=str, help='Directory with files, and output directory, default: %(default)s', default='./')

args=parser.parse_args()

initPlot(fontsize=30, titlesize=40, labelsize=28)

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


sidelengths=np.array([10])
Nsides=len(sidelengths)
fig= plt.figure(figsize=(10*Nsides+2, 20+2))
cmap=cm.get_cmap('RdBu', 32)
grid=ImageGrid(fig, 111, nrows_ncols=(Nsides, 2), axes_pad=0.15, share_all=True, aspect=True, cbar_location="right", cbar_mode="single", cbar_size="3%", cbar_pad=0.15)

# Set Xaxes labels
grid[0].set_title(r"$2\,\frac{C_{\hat{M}_\mathrm{ap}^3}^\mathrm{meas}-C_{\hat{M}_\mathrm{ap}^3}^{\infty}}{C_{\hat{M}_\mathrm{ap}^3}^\mathrm{meas} + C_{\hat{M}_\mathrm{ap}^3}^{\infty}}$", pad=40)
grid[1].set_title(r"$2\,\frac{C_{\hat{M}_\mathrm{ap}^3}^\mathrm{meas}-C_{\hat{M}_\mathrm{ap}^3}}{C_{\hat{M}_\mathrm{ap}^3}^\mathrm{meas} + C_{\hat{M}_\mathrm{ap}^3}}$", pad=40)
grid[(Nsides-1)*2].set_xlabel(r'$(\theta_4, \theta_5, \theta_6)$')
grid[(Nsides-1)*2].set_xticks(thetas_ticks)
grid[(Nsides-1)*2].set_xticklabels(thetas_labels, rotation=90)

grid[(Nsides-1)*2+1].set_xlabel(r'$(\theta_4, \theta_5, \theta_6)$')
grid[(Nsides-1)*2+1].set_xticks(thetas_ticks)
grid[(Nsides-1)*2+1].set_xticklabels(thetas_labels, rotation=90)

for i, theta in enumerate(sidelengths):
    n = 4096.0*4096.0/theta/theta
    thetaMax = theta-8*16/60

    # Load data
    if (cov_type == 'slics'):
        sigma=0.26
        n=108000.00
        thetaMax=7.87
        # cov_term1Numerical = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
        cov_term2Numerical = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
        cov_infiniteField = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
        # cov_term4Numerical = np.loadtxt(folder+f'cov_infinite_term4Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
        cov_fft = np.loadtxt(folder+f'cov_slics_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
        covUncertainty_fft=np.loadtxt(folder+f'covUncertainty_slics_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    
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
    diff_infinite=2*(cov_fft-cov_infiniteField)/(cov_fft+cov_infiniteField)
    diff_finite=2*(cov_fft-(cov_term1Numerical+cov_term2Numerical))/(cov_fft+cov_term1Numerical+cov_term2Numerical)

    # Add plots

    # Set yLabel
    grid[i*2].set_ylabel(r'$(\theta_1, \theta_2, \theta_3)$')
    grid[i*2].set_yticks(thetas_ticks)
    grid[i*2].set_yticklabels(thetas_labels)



    im = grid[i*2].imshow(diff_infinite, vmin=-5, vmax=5, cmap=cmap)  

    im = grid[i*2+1].imshow(0.5*(diff_finite+diff_finite.T), vmin=-5, vmax=5, cmap=cmap)  
    grid[i*2+1].text(19, 0, r"$\vartheta_\textrm{max}=$"+f"{thetaMax:.2f}°", verticalalignment='top', horizontalalignment='right',bbox=dict(facecolor='white', alpha=1))  

grid[Nsides*2-1].text(19, 19, cov_type, verticalalignment='bottom', horizontalalignment='right',bbox=dict(facecolor='white', alpha=1))  
grid[Nsides*2-1].cax.cla()
mcb.Colorbar(grid[Nsides*2-1].cax, im)

plt.savefig(folder+f"diffsCmeasCmodel.png", facecolor="white", dpi=300)