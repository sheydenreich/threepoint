# Script for calculating all analytic covariances for the sidelengths 5, 10, and 15 deg (for use on Lailas PC)


# First: Shapenoise only

DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_shapenoise/
mkdir -p $DIR
# T_1^\infty
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 1 0 0 infinite
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 1 0 0 infinite
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 1 0 0 infinite

# T_2
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 0 1 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 0 1 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 0 1 0 square

# T_1
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 1 0 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 1 0 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 1 0 0 square

# Second: Cosmic Shear only

DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_cosmicShear/
mkdir -p $DIR
# T_1^\infty
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 1 0 0 infinite
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 1 0 0 infinite
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 1 0 0 infinite

# T_2
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 0 1 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 0 1 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 0 1 0 square

# T_1
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 1 0 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 1 0 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 1 0 0 square

