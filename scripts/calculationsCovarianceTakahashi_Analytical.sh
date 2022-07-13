# Takahashi
# The Takashi are KiDSlike, i.e. their size is 859.436692686 sq.deg. and their shapenoise is sigma=0.374766594
# This script models the analytic covariance


DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/Takahashi/
mkdir -p $DIR

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
echo Using GPU $CUDA_VISIBLE_DEVICES

# T_1^\infty
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log

# T_2
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log

# T_4
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
./cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log

# T_5
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log

# T_7
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log


# T_6
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log

