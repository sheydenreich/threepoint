import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from helpers_plot import initPlot
from helpers_plot import finalizePlot
from helpers_covariance_plots import plotCov
from helpers_covariance_plots import getCovFromMap3
from helpers_covariance_plots import readCovFromFile

includeAnalytical=False #Set to true if analytical estimate is available

# Filenames
fn_numeric="../../Covariance_randomField/results/covariance_ccode_analytical_powerspectrum_x_exp_minus_x.dat"
fn_data="../../Covariance_randomField/results/map_cubed_gaussian_random_field_x_exp_minus_x.npy"
if includeAnalytical:
    fn_analytic="results/covariance_analytic_0.3_46.6_536.dat"

# Output Directory for plots
dir_out="../../Covariance_randomField/results/plots_x_exp_minus_x/"

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
cov_data=getCovFromMap3(fn_data, Nlos, Nthetas)
cov_data_half=getCovFromMap3(fn_data, int(Nlos/2), Nthetas)

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
    c_d=cov_data[i*N: (i+1)*N]
    c_dh=cov_data_half[i*N: (i+1)*N]
    
    theta1=theta[0]
    theta2=theta[1]
    theta3=theta[2]

    fn_out=dir_out+"cov_"+f"{theta1}"+"_"+f"{theta2}"+"_"+f"{theta3}"+".png"

    if includeAnalytical:
        plotCov(ax, c_a, theta1, theta2, theta3, "analytical", color='xkcd:blue', ls='-')
    plotCov(ax, c_n, theta1, theta2, theta3, "numerical", color='xkcd:red', ls='--')
    plotCov(ax, c_d, theta1, theta2, theta3, f"data ({Nlos} Realisations)", color='xkcd:tree green', ls='-')
    plotCov(ax, c_dh, theta1, theta2, theta3, f"data ({int(Nlos/2)} Realisations)", color='xkcd:aqua', ls='--')
    
    finalizePlot(ax, tightlayout=True, outputFn=fn_out, showplot=False, loc_legend='lower left')
    plt.close()
