# GAUSSIAN RANDOM FIELDS
# The Gaussian Random Fields are modelled with the cosmology of the Takahashi simulations
# and shapenoise from the KiDS
# They come in three sizes: 10°, 15°, and 20°
# This script models the analytic covariance



DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/GaussianRandomFields/
mkdir -p $DIR

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
echo Using GPU $CUDA_VISIBLE_DEVICES

### T_1^\infty
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_10deg.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_15deg.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_20deg.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log

### T_2


timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_10deg.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_15deg.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_20deg.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log

### T_1

timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_10deg.dat 1 0 0 0 0 0 square &> $DIR/${timestamp}.log
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_15deg.dat 1 0 0 0 0 0 square &> $DIR/${timestamp}.log
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_20deg.dat 1 0 0 0 0 0 square &> $DIR/${timestamp}.log

