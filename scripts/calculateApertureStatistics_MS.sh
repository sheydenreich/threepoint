DIR_BIN=../cuda_version/

DIR_RESULTS=/vol/aibn238/data1/llinke/5_Projects/02_3ptStatistics/results_MR/ #~/OneDrive/5_Projects/02_3ptStatistics/results_MR/

FILE_NZ=../necessary_files/nz_MR.dat

FILE_THETAS=../necessary_files/Our_thetas.dat

export CUDA_VISIBLE_DEVICES=1 #$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
echo Using GPU $CUDA_VISIBLE_DEVICES

#$DIR_BIN/calculateApertureStatistics.x ../necessary_files/MR_cosmo.dat $FILE_THETAS  $DIR_RESULTS/MapMapMap_bispec.dat 1 $FILE_NZ

$DIR_BIN/calculateMap4.x ../necessary_files/MR_cosmo.dat $FILE_THETAS $DIR_RESULTS/Map4_trispec_diag_newCode 1 $FILE_NZ

#$DIR_BIN/calculateMap6.x ../necessary_files/MR_cosmo.dat $FILE_THETAS $DIR_RESULTS/Map6_trispec 1 $FILE_NZ

#$DIR_BIN/calculatePowerspectrum_halomodel.x ../necessary_files/MR_cosmo.dat $DIR_RESULTS/Powerspec_halomodel 1 $FILE_NZ

#$DIR_BIN/calculateTrispectrum_halomodel.x ../necessary_files/MR_cosmo.dat $DIR_RESULTS/Trispec_halomodel2 1 $FILE_NZ

#$DIR_BIN/calculateTrispectrum3D_halomodel.x ../necessary_files/MR_cosmo.dat $DIR_RESULTS/Trispec3D_halomodel2 1 $FILE_NZ

#$DIR_BIN/calculateNFW.x ../necessary_files/MR_cosmo.dat $DIR_RESULTS/NFW.dat 1 $FILE_NZ

#$DIR_BIN/calculateHMF.x ../necessary_files/MR_cosmo.dat $DIR_RESULTS/HMF.dat 1 $FILE_NZ


