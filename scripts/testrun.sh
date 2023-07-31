DIR_BIN=../cuda_version/

COSMO_PARAM_FILE=../necessary_files/DUSTGRAIN_fiducial.dat

Z_COMBI_FILE=../python_scripts/Model_predictions/config_files/z_combis.dat

TH_COMBI_FILE=../python_scripts/Model_predictions/config_files/theta_combis.dat

OUTPUT_FILE=test.dat

N_TOMO=5

NZ1=../python_scripts/Model_predictions/config_files/nofz/nofz_KiDS1000_SLICS_bin1.dat
NZ2=../python_scripts/Model_predictions/config_files/nofz/nofz_KiDS1000_SLICS_bin2.dat
NZ3=../python_scripts/Model_predictions/config_files/nofz/nofz_KiDS1000_SLICS_bin3.dat
NZ4=../python_scripts/Model_predictions/config_files/nofz/nofz_KiDS1000_SLICS_bin4.dat
NZ5=../python_scripts/Model_predictions/config_files/nofz/nofz_KiDS1000_SLICS_bin5.dat

OUTPUTMODE=full

$DIR_BIN/calculateApertureStatistics.x $COSMO_PARAM_FILE $Z_COMBI_FILE $TH_COMBI_FILE $OUTPUT_FILE $N_TOMO $NZ1 $NZ2 $NZ3 $NZ4 $NZ5 $OUTPUTMODE