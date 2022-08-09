# SLICS
# The SLICS are 10 x 10 sq.deg large and have a shapenoise of \sigma=0.37
# This script models the analytic covariance


#DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/SLICS_singleZ/
DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS_singleZ/

mkdir -p $DIR

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
echo Using GPU $CUDA_VISIBLE_DEVICES

# T_1^\infty
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log

# T_2
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log

# T_4
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log

# T_5
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log



# T_6
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log

# T_7
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
