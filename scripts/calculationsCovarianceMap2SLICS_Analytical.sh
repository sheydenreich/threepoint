# SLICS
# The SLICS are 10 x 10 sq.deg large and have a shapenoise of \sigma=0.37
# This script models the analytic covariance


#DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/SLICS_theta_4_to_32/
DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS_theta_4_to_16/

mkdir -p $DIR
echo $DIR
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
echo Using GPU $CUDA_VISIBLE_DEVICES

# Gauss
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
echo $DIR/${timestamp}.log
../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_16.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 1 0 infinite &> $DIR/${timestamp}.log

# Non-Gauss
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
echo $DIR/${timestamp}.log
../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_16.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 1 infinite &> $DIR/${timestamp}.log
