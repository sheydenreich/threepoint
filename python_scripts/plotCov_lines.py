import numpy as np
import matplotlib.pyplot as plt

from matplotlib.offsetbox import AnchoredText
from helpers_plot import initPlot, finalizePlot


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
    thetas_labels.append(f"({thetas[0]}', {thetas[1]}', {thetas[2]}')")
N=len(thetas_ind)
thetas_ticks=np.arange(0, N)


sidelengths=np.array([5, 10, 15, 20])

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
            ax.plot(cov_term2Analytical[i]+cov_term1Analytical[i], color='C0', label='Term1 + Term2 (analytical)', ls='-')
            ax.plot(cov_term1Numerical[i]+cov_term2Numerical[i], color='C1', label='Term1 + Term2 (numerical)', ls='-')
            
            ax.plot(cov_term1Analytical[i], color='C0', label='Term1, analytical', ls='--')
            ax.plot(np.arange(0,N), cov_term1Numerical[i], color='C1', label='Term1, numerical integration', ls='--')
            
            ax.plot(cov_term2Analytical[i], color='C0', label='Term2, analytical', ls=':')
            ax.plot(np.arange(0, N), cov_term2Numerical[i], color='C1', label='Term2, numerical integration', ls=':')
        elif (cov_type == 'cosmicShear'):
            ax.errorbar(np.arange(0,N), cov_fft[i]/0.98, yerr=covUncertainty_fft[i]/0.98/4, color='xkcd:black', label='from FFT')
            ax.plot(cov_infiniteField[i], color='C4', label='Original formula')
            
            ax.plot(cov_term1Numerical[i]+cov_term2Numerical[i], color='C1', label='Term1 + Term2 (numerical)', ls='-')
            #ax.plot(cov_term1Round[i]+cov_term2Round[i], color='C2', label='Term1 + Term2, round survey (numerical)', ls='-')

            ax.plot(np.arange(0,N), cov_term1Numerical[i], color='C1', label='Term1, numerical integration', ls='--')
            #ax.plot(np.arange(0,N),cov_term1Round[i], color='C2', label='Term1, round (numerical)', ls='--')

            ax.plot(np.arange(0, N), cov_term2Numerical[i], color='C1', label='Term2, numerical integration', ls=':')
            #ax.plot(cov_term2Round[i], color='C2', label='Term2, round (numerical)', ls=':')
        else:
            print('Cov type not specified')
        at=AnchoredText(r"$\vartheta_\textrm{max}=$"+f"{thetaMax:.2f} deg", loc='lower left')
        ax.add_artist(at)
        
        finalizePlot(ax, title=cov_type+r' $\theta_1, \theta_2, \theta_3=$'+f"{thetas1[0]}', {thetas1[1]}', {thetas1[2]}'", outputFn=folder+f"cov_{thetas1[0]}_{thetas1[1]}_{thetas1[2]}_thetaMax_{thetaMax:.2f}.png", loc_legend='upper right', showplot=False)
        plt.clf()