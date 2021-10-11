DIR_BIN=../cuda_version/

DIR_RESULTS=~/OneDrive/1_Work/ThreePtStatistics/results_MR/

FILE_NZ=../necessary_files/nz_MR.dat

FILE_THETAS=../necessary_files/our_thetas.dat

$DIR_BIN/calculateApertureStatistics.x ../necessary_files/MR_cosmo.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec.dat 1 $FILE_NZ

