# Script for calculating all analytic covariances for the sidelengths 5, 10, and 15 deg (for use on Lailas PC)


# First: Shapenoise only

# DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_shapenoise/
# mkdir -p $DIR
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

#DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_cosmicShear/
#mkdir -p $DIR
# T_1^\infty
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 1 0 0 infinite
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 1 0 0 infinite
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 1 0 0 infinite

# T_2
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 0 1 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 0 1 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 0 1 0 square

# T_1
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 1 0 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 1 0 0 0 0 square
#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 1 0 0 square

# Third: SLICS

# DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS/
# mkdir -p $DIR

# # Map2 Gauss infinity
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# echo $DIR/${timestamp}.log
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 1 0 infinite &> $DIR/${timestamp}.log

# # Map2 Gauss square
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# echo $DIR/${timestamp}.log
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 1 0 square &> $DIR/${timestamp}.log


# # T_1^\infty
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"

#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_addedShapenoise.dat 1 0 0 0 0 infinite &> $DIR/${timestamp}.log

# # T_2
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_addedShapenoise.dat 0 1 0 0 0 square &> $DIR/${timestamp}.log

# # T_1
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 1 0 0 square &> /vol/lensgpu/ssd/llinke/${timestamp}.log

# # T_4
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_sigmaTwice.dat 0 0 1 0 0 infinite &> $DIR/${timestamp}.log


# # # T_5
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log

# # # T_6
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log

# T_7
#timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log


# Map2 Gauss infinity
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# echo $DIR/${timestamp}.log
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat  $DIR ../necessary_files/Covariance_SLICS.dat 1 0 infinite &> $DIR/${timestamp}.log

# # Map2 Gauss square
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# echo $DIR/${timestamp}.log
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat  $DIR ../necessary_files/Covariance_SLICS.dat 1 0 square &> $DIR/${timestamp}.log

# # Map2 NonGauss infinity
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# echo $DIR/${timestamp}.log
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat  $DIR ../necessary_files/Covariance_SLICS.dat 0 1 infinite &> $DIR/${timestamp}.log

# # Map2 NonGauss square
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# echo $DIR/${timestamp}.log
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat  $DIR ../necessary_files/Covariance_SLICS.dat 0 1 square &> $DIR/${timestamp}.log



# Fourth: MS
DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/MS/
mkdir -p $DIR


# # # Map2 Gauss infinity
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# echo $DIR/${timestamp}.log
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 1 0 infinite &> $DIR/${timestamp}.log

# # Map2 Gauss square
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# echo $DIR/${timestamp}.log
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 1 0 square &> $DIR/${timestamp}.log




# # Map2 NonGauss square
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# echo $DIR/${timestamp}.log
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 1 square &> $DIR/${timestamp}.log

# Map2 Gauss infinity
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
echo $DIR/${timestamp}.log
CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS_shapenoise.dat 1 0 infinite &> $DIR/${timestamp}.log

# Map2 NonGauss infinity
timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
echo $DIR/${timestamp}.log
CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS_shapenoise.dat 0 1 infinite &> $DIR/${timestamp}.log


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


# Calculate Powerspec, Bispec and Bispec Cov for SLICS
#../classed_version/calculateBispectrumAndCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat  /home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS/ 10 0.37 108000 deg






# # KiDS Like SLICS

# DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS_KiDS_South/
# mkdir -p $DIR
# # T_1^\infty
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"

# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log


# # T_2
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 1 0 0 0 0 rectangle &> $DIR/${timestamp}.log

# # T_1
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"

# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 1 0 0 0 0 0 rectangle &> $DIR/${timestamp}.log

# # T_4^\infty
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"

# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log

# # T_5^\infty
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"

# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log

# # T_6
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 0 0 0 1 0 rectangle &> $DIR/${timestamp}.log

# # T_7^\infty
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"

# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log

# DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS_KiDS_North/
# mkdir -p $DIR

# # T_2
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 1 0 0 0 0 rectangle &> $DIR/${timestamp}.log

# # T_1
# timestamp="$(date +"%Y_%m_%d_%H_%M_%S")"
# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 1 0 0 0 0 0 rectangle &> $DIR/${timestamp}.log
