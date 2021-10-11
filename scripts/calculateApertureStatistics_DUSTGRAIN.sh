DIR_BIN=../cuda_version/

DIR_RESULTS=~/OneDrive/1_Work/ThreePtStatistics/results_HOWLS/

FILE_NZ=../necessary_files/n_z_DUSTGRAIN_euclid_nz_cosmos15_vis24.5cut_fu08fit_dz0.01.cat

FILE_THETAS=../necessary_files/HOWLS_thetas.dat


# FIDUCIAL
#$DIR_BIN/calculateApertureStatistics.x ../necessary_files/DUSTGRAIN_fiducial.dat $FILE_THETAS $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_fiducial.dat 1 $FILE_NZ > test

# Omegas
$DIR_BIN/calculateApertureStatistics.x ../necessary_files/DUSTGRAIN_Om02.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om02.dat 1 $FILE_NZ

$DIR_BIN/calculateApertureStatistics.x   ../necessary_files/DUSTGRAIN_Om0300912.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om0300912.dat 1 $FILE_NZ

#$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_Om0325988.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om0325988.dat 1 $FILE_NZ

#$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_Om04.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om04.dat 1 $FILE_NZ

# s8
#$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s80707210.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s80707210.dat 1 $FILE_NZ

#$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s80808240.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s80808240.dat 1 $FILE_NZ

#$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s80875594.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s80875594.dat 1 $FILE_NZ

#$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s80976624.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s80976624.dat 1 $FILE_NZ

# w
#$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w_-0.84.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-0.84.dat 1 $FILE_NZ

#$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w_-0.96.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-0.96.dat 1 $FILE_NZ

#$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w_-1.04.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-1.04.dat 1 $FILE_NZ

#$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w_-1.16.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-1.16.dat 1 $FILE_NZ
