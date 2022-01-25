import numpy as np
import matplotlib.pyplot as plt

from helpers_plot import initPlot
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colorbar as mcb
import matplotlib.cm as cm

initPlot(titlesize=20)

cov_type = "shapenoise" # Can be 'slics' or 'shapenoise' or 'cosmicShear' cov
sigma = 0.3

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


sidelengths=np.array([5, 10, 15])
Nsides=len(sidelengths)
fig= plt.figure(figsize=(10*Nsides+2, 20+2))
cmap=cm.get_cmap('RdBu', 20)
grid=ImageGrid(fig, 111, nrows_ncols=(Nsides, 2), axes_pad=0.15, share_all=True, aspect=True, cbar_location="right", cbar_mode="single", cbar_size="3%", cbar_pad=0.15)

# Set Xaxes labels
grid[0].set_title(r"$\frac{\langle\hat{M}^3\hat{M}^3 \rangle_\mathrm{meas}-T^\infty_1}{\sigma_\mathrm{meas}}$", pad=20)
grid[1].set_title(r"$\frac{\langle\hat{M}^3\hat{M}^3 \rangle_\mathrm{meas}-(T_1+T_2)}{\sigma_\mathrm{meas}}$", pad=20)
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
        cov_term1Numerical = np.loadtxt(folder+f'cov_cosmicShear_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
        cov_term2Numerical = np.loadtxt(folder+f'cov_cosmicShear_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
        cov_infiniteField = np.loadtxt(folder+f'cov_cosmicShear_infiniteField_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
        cov_fft = np.loadtxt(folder+f'cov_cosmicShear_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')*0.775
        covUncertainty_fft=np.loadtxt(folder+f'covUncertainty_cosmicShear_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    else:
        print("Cov type not specified")
        exit

    diff_infinite=(cov_fft-cov_infiniteField)/(covUncertainty_fft)
    diff_finite=(cov_fft-(cov_term1Numerical+cov_term2Numerical))/(covUncertainty_fft)

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

plt.savefig(folder+f"diffsCmeasCmodelUncertainty.png", facecolor="white", dpi=300)