DIR_BIN=../cuda_version/

DIR_RESULTS=~/OneDrive/1_Work/ThreePtStatistics/results_HOWLS/

FILE_NZ=../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat

FILE_THETAS=../necessary_files/HOWLS_thetas.dat

$DIR_BIN/calculateApertureStatistics.x ../necessary_files/SLICS_cosmo.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_SLICS_fiducial.dat 1 $FILE_NZ

