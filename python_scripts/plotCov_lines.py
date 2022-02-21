import numpy as np
import matplotlib.pyplot as plt

from matplotlib.offsetbox import AnchoredText
from helpers_plot import initPlot, finalizePlot


import argparse

description="""Script for plotting C_meas, T_1^\infty, T_1, T_2, and T_1+T_2 for one theta_max as lineplots
"""

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--cov_type', default='none', type=str, help='Type of covariance that is plotted, can be either slics, shapenoise or cosmicShear, default: %(default)s')

parser.add_argument('--sigma', default=0.0, type=float, help='Shapenoise. default: %(default)s')

parser.add_argument('--dir', type=str, help='Directory with files, and output directory, default: %(default)s', default='./')

args=parser.parse_args()

initPlot(titlesize=20)

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
    thetas_labels.append(f"({thetas[0]}', {thetas[1]}', {thetas[2]}')")
N=len(thetas_ind)
thetas_ticks=np.arange(0, N)


sidelengths=np.array([5, 10, 15]) #np.array([5, 10, 15])

for theta in sidelengths:
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

        cov_term1Numerical_gpu = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')

        cov_term2Numerical_gpu = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
        cov_infiniteField = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
        cov_infiniteField_gpu = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
        
        cov_fft = np.loadtxt(folder+f'cov_shapenoise_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
        covUncertainty_fft=np.loadtxt(folder+f'covUncertainty_shapenoise_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
    elif (cov_type == 'cosmicShear'):
        cov_term1Numerical = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')

        cov_term2Numerical = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
        cov_infiniteField = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
        cov_fft = np.loadtxt(folder+f'cov_cosmicShear_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')
        covUncertainty_fft=np.loadtxt(folder+f'covUncertainty_cosmicShear_fft_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}.dat')

        cov_term1Numerical_gpu = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')

        cov_term2Numerical_gpu = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
        cov_infiniteField_gpu = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu.dat')
        cov_infiniteField_lMin = np.loadtxt(folder+f'cov_infinite_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu_lMin.dat')
        cov_term2Numerical_lMin = np.loadtxt(folder+f'cov_square_term2Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu_lMin.dat')
        #cov_term1Numerical_lMin = np.loadtxt(folder+f'cov_square_term1Numerical_sigma_{sigma}_n_{n:.2f}_thetaMax_{thetaMax:.2f}_gpu_lMin.dat')
    else:
        print("Cov type not specified")
        exit

    
    # Do plots
    N =len(thetas_ind)
    thetas_ticks=np.arange(0, N)
    for i, thetas1 in enumerate(thetas_ind):
        fig, ax=plt.subplots(figsize=(21,12))
        ax.set_yscale('log')
        ax.set_xlabel(r'$\theta_4, \theta_5, \theta_6$ [arcmin]')
        ax.set_xticks(thetas_ticks)
        ax.set_xticklabels(thetas_labels)

        if(cov_type=='slics'):
            ax.plot(cov_infiniteField[i], color='xkcd:bright blue', label='Infinite Field G')
            ax.plot(cov_infiniteFieldNG[i], color='xkcd:purple', label='Infinite Field NG')
            ax.plot(cov_infiniteFieldNG[i]+cov_infiniteField[i], color='xkcd:pink', label='Infinite Field NG+G')
            
            ax.plot(cov_term2Numerical[i], color='xkcd:brick red', label='Term2, numerical integration', ls='--')

            ax.plot(cov_fft[i]/15, color='xkcd:black', label='from FFT (/15)')
            ax.plot(cov_term2Numerical[i]+cov_infiniteField[i], color='xkcd:british racing green', label='Term2 + Infinite Field G')
        elif (cov_type == 'shapenoise'):
            ax.errorbar(np.arange(0, N),cov_fft[i], yerr=covUncertainty_fft[i], color='xkcd:black', label='from FFT')
            ax.plot(cov_infiniteField[i], color='C4', label='Original formula')
            ax.plot(cov_infiniteField_gpu[i], color='xkcd:pink', label='Original formula (GPU)')
            
            ax.plot(cov_term2Analytical[i]+cov_term1Analytical[i], color='C0', label='Term1 + Term2 (analytical)', ls='-')
            ax.plot(cov_term1Numerical[i]+cov_term2Numerical[i], color='C1', label='Term1 + Term2 (numerical)', ls='-')
            
            ax.plot(cov_term1Analytical[i], color='C0', label='Term1, analytical', ls='--')
            ax.plot(np.arange(0,N), cov_term1Numerical[i], color='C1', label='Term1, numerical integration', ls='--')
            ax.plot(np.arange(0,N), cov_term1Numerical_gpu[i], color='xkcd:green', label='Term1, numerical integration', ls='--')
            
            
            ax.plot(cov_term2Analytical[i], color='C0', label='Term2, analytical', ls=':')
            ax.plot(np.arange(0, N), cov_term2Numerical[i], color='C1', label='Term2, numerical integration', ls=':')
            ax.plot(np.arange(0, N), cov_term2Numerical_gpu[i], color='C1', label='Term2, numerical integration', ls=':')
        elif (cov_type == 'cosmicShear'):
            ax.errorbar(np.arange(0, N),cov_fft[i], yerr=covUncertainty_fft[i], color='xkcd:black', label='from FFT')
            ax.plot(cov_infiniteField[i], color='xkcd:red', label=r'$T_1^\infty$ (with $\ell_\mathrm{min}=2\pi/\vartheta_\mathrm{max}$)', ls='--')
            ax.plot(np.arange(0, N), cov_term2Numerical_lMin[i], color='C4', label=r'$T_2$, (with $\ell_\mathrm{min}=2\pi/\vartheta_\mathrm{max}$)', ls=':')
            #ax.plot(np.arange(0, N), cov_term1Numerical_lMin[i], color='C4', label=r'$T_1$, (with $\ell_\mathrm{min}=2\pi/\vartheta_\mathrm{max}$)', ls='--')
            ax.plot(cov_term1Numerical_gpu[i]+cov_term2Numerical_lMin[i], color='C4', label=r'$T_1 + T_2$ (with $\ell_\mathrm{min}=2\pi/\vartheta_\mathrm{max}$)')
            #ax.plot(cov_term1Numerical[i]+cov_term2Numerical[i], color='C1', label='Term1 + Term2 (numerical)', ls='-')
            
            #ax.plot(np.arange(0,N), cov_term1Numerical[i], color='C1', label='Term1, numerical integration', ls='--')
            
            #ax.plot(np.arange(0, N), cov_term2Numerical[i], color='C1', label='Term2, numerical integration', ls=':')
            ax.plot(cov_infiniteField_gpu[i], color='xkcd:pink', label=r'$T_1^\infty$ (GPU)')
            ax.plot(np.arange(0, N), cov_term2Numerical_gpu[i], color='xkcd:green', label=r'$T_2$, numerical integration (GPU)', ls=':')
            ax.plot(np.arange(0,N), cov_term1Numerical_gpu[i], color='xkcd:green', label=r'$T_1$, numerical integration (GPU)', ls='--')
            ax.plot(cov_term1Numerical_gpu[i]+cov_term2Numerical_gpu[i], color='xkcd:green', label=r'$T_1+T_2$ (numerical, GPU)', ls='-')

        else:
            print('Cov type not specified')
        at=AnchoredText(r"$\vartheta_\textrm{max}=$"+f"{thetaMax:.2f} deg", loc='lower left')
        ax.add_artist(at)
        
        finalizePlot(ax, title=cov_type+r' $\theta_1, \theta_2, \theta_3=$'+f"{thetas1[0]}', {thetas1[1]}', {thetas1[2]}'", outputFn=folder+f"cov_{thetas1[0]}_{thetas1[1]}_{thetas1[2]}_thetaMax_{thetaMax:.2f}.png", loc_legend='upper right', showplot=False)
        plt.clf()