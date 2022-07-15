DIR_BIN=../cuda_version/
DIR_NZ=../necessary_files/nz_MR.dat
DIR_COSMO=../necessary_files/MR_cosmo.dat


DIR_RESULTS=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/

$DIR_BIN/calculateBispectrum.x $DIR_COSMO $DIR_NZ equilateral 0 $DIR_RESULTS/bispec_model_equilateral.dat
$DIR_BIN/calculateBispectrum.x $DIR_COSMO $DIR_NZ equilateral 1 $DIR_RESULTS/bispec_model_equilateral_average.dat

$DIR_BIN/calculateBispectrum.x $DIR_COSMO $DIR_NZ flattened 0 $DIR_RESULTS/bispec_model_flattened.dat
$DIR_BIN/calculateBispectrum.x $DIR_COSMO $DIR_NZ flattened 1 $DIR_RESULTS/bispec_model_flattened_average.dat

$DIR_BIN/calculateBispectrum.x $DIR_COSMO $DIR_NZ squeezed_500 0 $DIR_RESULTS/bispec_model_squeezed_500.dat
$DIR_BIN/calculateBispectrum.x $DIR_COSMO $DIR_NZ squeezed_500 1 $DIR_RESULTS/bispec_model_squeezed_500_average.dat

$DIR_BIN/calculateBispectrum.x $DIR_COSMO $DIR_NZ squeezed_5000 0 $DIR_RESULTS/bispec_model_squeezed_5000.dat
$DIR_BIN/calculateBispectrum.x $DIR_COSMO $DIR_NZ squeezed_5000 1 $DIR_RESULTS/bispec_model_squeezed_5000_average.dat
