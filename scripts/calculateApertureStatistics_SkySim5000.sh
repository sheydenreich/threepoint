DIR_BIN=../cuda_version/

DIR_RESULTS=/home/laila/OneDrive/1_Work/5_Projects/04_DESC_HOS/Sprint2_FirstHACCTests/Results/


FILE_THETAS=../necessary_files/Our_thetas_4_to_32.dat

FILE_COSMO=../necessary_files/SkySim5000_cosmo_changed.dat

$DIR_BIN/calculateApertureStatistics.x $FILE_COSMO $FILE_THETAS $DIR_RESULTS/Map3_theo_tomo_0_changed.dat ../necessary_files/SkySim5000_bin1.dat
$DIR_BIN/calculateApertureStatistics.x $FILE_COSMO $FILE_THETAS $DIR_RESULTS/Map3_theo_tomo_1_changed.dat ../necessary_files/SkySim5000_bin2.dat
$DIR_BIN/calculateApertureStatistics.x $FILE_COSMO $FILE_THETAS $DIR_RESULTS/Map3_theo_tomo_2_changed.dat ../necessary_files/SkySim5000_bin3.dat
$DIR_BIN/calculateApertureStatistics.x $FILE_COSMO $FILE_THETAS $DIR_RESULTS/Map3_theo_tomo_3_changed.dat ../necessary_files/SkySim5000_bin4.dat
$DIR_BIN/calculateApertureStatistics.x $FILE_COSMO $FILE_THETAS $DIR_RESULTS/Map3_theo_tomo_4_changed.dat ../necessary_files/SkySim5000_bin5.dat


# #Bin 1
# $DIR_BIN/calculateApertureStatistics.x $FILE_COSMO $FILE_THETAS $DIR_RESULTS/Map3_theo_tomo_0.dat ../necessary_files/SkySim5000_bin1.dat

# #Bin 2
# $DIR_BIN/calculateApertureStatistics.x $FILE_COSMO $FILE_THETAS $DIR_RESULTS/Map3_theo_tomo_1.dat ../necessary_files/SkySim5000_bin2.dat

# #Bin 3
# $DIR_BIN/calculateApertureStatistics.x $FILE_COSMO $FILE_THETAS $DIR_RESULTS/Map3_theo_tomo_2.dat ../necessary_files/SkySim5000_bin3.dat

# #Bin 4
# $DIR_BIN/calculateApertureStatistics.x $FILE_COSMO $FILE_THETAS $DIR_RESULTS/Map3_theo_tomo_3.dat ../necessary_files/SkySim5000_bin4.dat

# #Bin 5
# $DIR_BIN/calculateApertureStatistics.x $FILE_COSMO $FILE_THETAS $DIR_RESULTS/Map3_theo_tomo_4.dat ../necessary_files/SkySim5000_bin5.dat

