# Script for calculating all analytic covariances for the sidelengths 5, 10, and 15 deg (for use in AIfA)


# # First: Shapenoise only

# DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/GaussianRandomFields_shapenoise/
# mkdir -p $DIR
# # T_1^\infty
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 1 0 0 infinite
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 1 0 0 infinite
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 1 0 0 infinite

# # T_2
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 0 1 0 square
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 0 1 0 square
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 0 1 0 square

# # T_2
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 1 0 0 square
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 1 0 0 square
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 1 0 0 square

# # Second: Cosmic Shear only

# DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/GaussianRandomFields_cosmicShear/
# mkdir -p $DIR
# # T_1^\infty
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 1 0 0 infinite
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 1 0 0 infinite
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 1 0 0 infinite

# # T_2
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 0 1 0 square
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 0 1 0 square
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 0 1 0 square

# # T_2
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 1 0 0 square
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 1 0 0 square
# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 1 0 0 square

# Third: SLICS

#DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/SLICS/
#mkdir -p $DIR
# T_1^\infty
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 1 0 0 0 0 0 infinite &> /vol/lensgpu/ssd/llinke/${timestamp}.log

# T_2
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 1 0 0 0 0 square &> /vol/lensgpu/ssd/llinke/${timestamp}.log

# T_4
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=1 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 1 0 0 0 infinite &> /vol/lensgpu/ssd/llinke/${timestamp}.log

# T_5
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=1 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 1 0 0 infinite &> /vol/lensgpu/ssd/llinke/${timestamp}.log

# T_6
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=1 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 0 1 0 square &> /vol/lensgpu/ssd/llinke/${timestamp}.log

# T_7
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=1 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 0 0 1 infinite &> /vol/lensgpu/ssd/llinke/${timestamp}.log


# Fourth: MS
#DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/MS/
#mkdir -p $DIR
# # T_1^\infty
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#echo $DIR/${timestamp}.log
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 1 0 0 0 0 infinite &> $DIR/${timestamp}.log

# # T_2
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 1 0 0 0 square &> $DIR/${timestamp}.log

# # T_1
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 1 0 0 0 0 square &> $DIR/${timestamp}.log

# T_4
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log


# T_5
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log

# T_6
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log

# T_7
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log


#SSC
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsSSC.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat &> $DIR/${timestamp}.log

#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/testBispecSSC.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat &> $DIR/${timestamp}.log


#TAKAHASHI
DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/Takahashi/
mkdir -p $DIR
T_1^\infty
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
echo $DIR/${timestamp}.log
CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log

# T_2
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log


# T_4^\infty
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log


# T_5^\infty
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log

# T_6
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log

# T_7
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
