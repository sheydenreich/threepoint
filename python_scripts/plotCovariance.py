import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from helpers_plot import initPlot
from helpers_plot import finalizePlot
from helpers_covariance_plots import plotCov
from helpers_covariance_plots import getCovFromMap3
from helpers_covariance_plots import readCovFromFile

constantPowerspectrum=True
analyticPowerspectrum_l=False
analyticPowerspectrum_lSq=False

# Filenames
if constantPowerspectrum:
    fn_numeric="../../Covariance_randomField/results/covariance_ccode_0.3_46.6_536_pcubature.dat"
    fn_analytic="../../Covariance_randomField/results/covariance_analytic_0.3_46.6_536.dat"
    fn_data="../../Covariance_randomField/results/map_cubed_only_shapenoise.npy"
    fn_data_noZeroPadding="/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/map_cubed_only_shapenoise_without_zeropadding.npy"
    
    dir_out="../../Covariance_randomField/results/plots_constantPowerspectrum/"
    includeAnalytical=True
    includeData=True
    includeData_noZeroPadding=True
elif analyticPowerspectrum_l:
    fn_numeric="../../Covariance_randomField/results/covariance_ccode_analytical_powerspectrum_x_exp_minus_x.dat"
    fn_data="../../Covariance_randomField/results/map_cubed_gaussian_random_field_x_exp_minus_x.npy"
    dir_out="../../Covariance_randomField/results/plots_x_exp_minus_x/"
    includeAnalytical=False
    includeData=True
    includeData_noZeroPadding=False
elif analyticPowerspectrum_lSq:
    fn_numeric="../../Covariance_randomField/results/covariance_ccode_analytical_powerspectrum_xSq_exp_minus_xSq.dat"
    fn_data="../../Covariance_randomField/results/map_cubed_gaussian_random_field_xsq_exp_minus_xsq.npy"
    fn_analytic="../../Covariance_randomField/results/covariance_analytic_p1_1.00e-08_p2_1.00e-08_side_536.dat"
    dir_out="../../Covariance_randomField/results/plots_xSq_exp_minus_xSq/"
    includeAnalytical=True
    includeData=True
    includeData_noZeroPadding=False   

# Set Thetas
thetas_1d=np.array([1, 2, 4, 8, 16])
Nthetas=len(thetas_1d)

# Set number of LOS
Nlos=2000

# Read in covariance from file
cov_numeric=readCovFromFile(fn_numeric)
if includeAnalytical:
    cov_analytic=readCovFromFile(fn_analytic)

# Get Covariance for data
if includeData:
    cov_data=getCovFromMap3(fn_data, Nlos, Nthetas)
    cov_data_half=getCovFromMap3(fn_data, int(Nlos/2), Nthetas)

if includeData_noZeroPadding:
    cov_data_noZeroPadding=getCovFromMap3(fn_data_noZeroPadding, Nlos, Nthetas)
    
# Initialize Plot
initPlot()

# Set thetas
thetas1, thetas2, thetas3=np.meshgrid(thetas_1d, thetas_1d, thetas_1d)
thetas_all=np.c_[thetas1.ravel(), thetas2.ravel(), thetas3.ravel()]
N=len(thetas_all)
# Create Plot
for i, theta in enumerate(thetas_all):
    fig, ax=plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
    if includeAnalytical:
        c_a=cov_analytic[i*N: (i+1)*N]
    c_n=cov_numeric[i*N: (i+1)*N]
    if includeData:
        c_d=cov_data[i*N: (i+1)*N]
        c_dh=cov_data_half[i*N: (i+1)*N]

    if includeData_noZeroPadding:
        c_dnz=cov_data_noZeroPadding[i*N: (i+1)*N]
        
    theta1=theta[0]
    theta2=theta[1]
    theta3=theta[2]

    fn_out=dir_out+"cov_"+f"{theta1}"+"_"+f"{theta2}"+"_"+f"{theta3}"+".png"

    if includeAnalytical:
        plotCov(ax, c_a, theta1, theta2, theta3, "analytical", color='xkcd:blue', ls='-')
    plotCov(ax, c_n, theta1, theta2, theta3, "numerical", color='xkcd:red', ls='--')
    if includeData:
        plotCov(ax, c_d, theta1, theta2, theta3, f"data ({Nlos} Realisations)", color='xkcd:tree green', ls='-')
        plotCov(ax, c_dh, theta1, theta2, theta3, f"data ({int(Nlos/2)} Realisations)", color='xkcd:aqua', ls='--')

    if includeData_noZeroPadding:
        plotCov(ax, c_dnz, theta1, theta2, theta3, f"data, no ZeroPadding", color='xkcd:orange', ls=':')
        
    finalizePlot(ax, tightlayout=True, outputFn=fn_out, showplot=False, loc_legend='lower left')
    plt.close()