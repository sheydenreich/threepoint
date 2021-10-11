DIR_BIN=../cuda_version/

DIR_RESULTS=~/OneDrive/1_Work/ThreePtStatistics/results_HOWLS/model

FILE_NZ=../necessary_files/n_z_DUSTGRAIN_euclid_nz_cosmos15_vis24.5cut_fu08fit_dz0.01.cat

FILE_THETAS=../necessary_files/HOWLS_thetas.dat

STENCIL_SIZE=0.04

$DIR_BIN/calculateDerivativeApertureStatistics.x ../necessary_files/DUSTGRAIN_fiducial.dat $FILE_THETAS $DIR_RESULTS/MapMapMap_derivatives_bispec_DUSTGRAIN.dat 0 $STENCIL_SIZE 1 $FILE_NZ