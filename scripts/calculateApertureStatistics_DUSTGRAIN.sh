DIR_BIN=../cuda_version/

DIR_RESULTS=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/results_HOWLS/model/


FILE_NZ=../necessary_files/n_z_DUSTGRAIN_euclid_nz_cosmos15_vis24.5cut_fu08fit_dz0.01.cat

FILE_THETAS=../necessary_files/HOWLS_thetas.dat


# FIDUCIAL


# # Omegas
$DIR_BIN/calculateApertureStatistics.x ../necessary_files/DUSTGRAIN_Om02.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om02.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_Om04.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om04.dat 1 $FILE_NZ

$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_Om0237816.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om0237816.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_Om0275632.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om0275632.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_Om0300912.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om0300912.dat 1 $FILE_NZ

$DIR_BIN/calculateApertureStatistics.x   ../necessary_files/DUSTGRAIN_Om0313448.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om0313448.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_Om0325988.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om0325988.dat 1 $FILE_NZ

$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_Om0342298666667.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om0342298666667.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_Om0342298666667.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_Om0342298666667.dat 1 $FILE_NZ



# # s8
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s80842.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s80842.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s8070721.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s8070721.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s8075214.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s8075214.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s8079707.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s8079707.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s80875594.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s80875594.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s80976624.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s80976624.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s808080824.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s808080824.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s809317493333.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s809317493333.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_s80886874666667.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_s80886874666667.dat 1 $FILE_NZ





# # w
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w-0.84.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-0.84.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w-0.96.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-0.96.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w-0.893333.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-0.893333.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w-0.946667.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-0.946667.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w-1.0.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-1.0.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w-1.04.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-1.04.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w-1.16.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-1.16.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w-1.0533.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-1.0533.dat 1 $FILE_NZ
$DIR_BIN/calculateApertureStatistics.x  ../necessary_files/DUSTGRAIN_w-1.106667.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec_DUSTGRAIN_w_-1.106667.dat 1 $FILE_NZ