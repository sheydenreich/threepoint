##############
## EXAMPLES FOR USAGE OF THESE CODES
## @author Laila Linke, llinke@astro.uni-bonn.de
###

DIR_BIN=../cuda_version/
#DIR_RESULTS=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/Takahashi/
DIR_RESULTS=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS_theta_4_to_16/

# FILE_NZ=../necessary_files/nofz_kids1000_takahashi.dat
# FILE_THETAS=exampleThetas.dat
# FILE_COSMOLOGY=../necessary_files/Takahashi_cosmo.dat
# FILE_COVARIANCE=../necessary_files/Covariance_Takahashi_KiDSlike.dat


FILE_NZ=../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat
FILE_THETAS=exampleThetas.dat
FILE_COSMOLOGY=../necessary_files/SLICS_cosmo.dat
FILE_COVARIANCE=../necessary_files/Covariance_SLICS.dat


$DIR_BIN/calculateMap3SSC.x $FILE_COSMOLOGY $FILE_THETAS $FILE_NZ $DIR_RESULTS $FILE_COVARIANCE square 


#$DIR_BIN/calculateMap2Map3Covariance.x $FILE_COSMOLOGY $FILE_THETAS $FILE_NZ $DIR_RESULTS $FILE_COVARIANCE 0 1 0 infinite

#$DIR_BIN/calculateMap2Map3Covariance.x $FILE_COSMOLOGY $FILE_THETAS $FILE_NZ $DIR_RESULTS $FILE_COVARIANCE 0 0 1 infinite 


#$DIR_BIN/calculateMap2Map3Covariance.x $FILE_COSMOLOGY $FILE_THETAS $FILE_NZ $DIR_RESULTS $FILE_COVARIANCE 1 0 0 square 


#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
