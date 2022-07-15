DIR_BIN=../cuda_version
DIR_NZ=../necessary_files/nz_MR.dat
DIR_COSMO=../necessary_files/MR_cosmo.dat
APP=_bins.dat


DIR_RESULTS=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR

for bin in "7" "10" "15" "20"; do

echo $DIR_BIN/calculateGamma.x $DIR_COSMO ../necessary_files/config_gamma_0p1_to_240_$bin$APP $DIR_RESULTS/gammas_0p1_to_240_$bin$APP 1 $DIR_NZ

$DIR_BIN/calculateGamma.x $DIR_COSMO ../necessary_files/config_gamma_0p1_to_240_$bin$APP $DIR_RESULTS/gammas_0p1_to_240_$bin$APP 1 $DIR_NZ

done