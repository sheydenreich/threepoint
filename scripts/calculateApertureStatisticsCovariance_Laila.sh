# Script for calculating all analytic covariances for the sidelengths 5, 10, and 15 deg (for use on Lailas PC)


# First: Shapenoise only

DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_shapenoise/

# T_1^\infty
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 2.87 0.3 671088.64 deg 1 0 0 infinite 1
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 7.87 0.3 167772-16 deg 1 0 0 infinite 1
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 12.87 0.3 74565.40 deg 1 0 0 infinite 1

# T_2
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 2.87 0.3 671088.64 deg 0 1 0 square 1
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 7.87 0.3 167772-16 deg 0 1 0 square 1
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 12.87 0.3 74565.40 deg 0 1 0 square 1

# T_1
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 2.87 0.3 671088.64 deg 1 0 0 square 1
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 7.87 0.3 167772-16 deg 1 0 0 square 1
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 12.87 0.3 74565.40 deg 1 0 0 square 1

# Second: Cosmic Shear only

DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/GaussianRandomFields_cosmicShear/

# T_1^\infty
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 2.87 0.0 671088.64 deg 1 0 0 infinite 0
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 7.87 0.0 167772-16 deg 1 0 0 infinite 0
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 12.87 0.0 74565.40 deg 1 0 0 infinite 0

# T_2
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 2.87 0.0 671088.64 deg 0 1 0 square 0
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 7.87 0.0 167772-16 deg 0 1 0 square 0
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 12.87 0.0 74565.40 deg 0 1 0 square 0

# T_1
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 2.87 0.0 671088.64 deg 1 0 0 square 0
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 7.87 0.0 167772-16 deg 1 0 0 square 0
../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_deg.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR 12.87 0.0 74565.40 deg 1 0 0 square 0

