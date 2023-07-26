DIR_BIN=../cuda_version/

DIR_RESULTS=/home/laila/OneDrive/1_Work/5_Projects/09_Map3InFS/Results/



FILE_THETAS=../necessary_files/Our_thetas_4_to_32.dat

FILE_COSMOLOGY=../necessary_files/Flagship_cosmo.dat

LOS=117

FILE_NZ=../necessary_files/Flagship_n_z_slice$LOS.dat


#$DIR_BIN/calculateApertureStatistics.x $FILE_COSMOLOGY $FILE_THETAS $DIR_RESULTS/MapMapMap_theo_$LOS.dat $FILE_NZ


LOS=146

FILE_NZ=../necessary_files/Flagship_n_z_slice$LOS.dat


$DIR_BIN/calculateApertureStatistics.x $FILE_COSMOLOGY $FILE_THETAS $DIR_RESULTS/MapMapMap_theo_$LOS.dat $FILE_NZ

