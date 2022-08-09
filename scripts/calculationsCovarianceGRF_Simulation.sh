# GAUSSIAN RANDOM FIELDS
# The Gaussian Random Fields are modelled with the cosmology of the Takahashi simulations
# and shapenoise from the KiDS
# They come in three sizes: 10°, 15°, and 20°
# This script creates GRFs and measures Map3 in them


## FFT SIMULATION

NPROC=6 #128
NREAL=4096 #4096

#DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/GaussianRandomFields/
DIR="/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields/"
mkdir -p $DIR

python ../python_scripts/computeMap3_GaussianFields.py --npix 4096 --fieldsize 10 --power_spectrum -1 --power_spectrum_filename ../necessary_files/p_ell_takahashi_shapenoise_kidslike.dat --processes $NPROC --realisations $NREAL --savepath $DIR
python ../python_scripts/computeMap3_GaussianFields.py --npix 4096 --fieldsize 15 --power_spectrum -1 --power_spectrum_filename ../necessary_files/p_ell_takahashi_shapenoise_kidslike.dat --processes $NPROC --realisations $NREAL --savepath $DIR
python ../python_scripts/computeMap3_GaussianFields.py --npix 4096 --fieldsize 20 --power_spectrum -1 --power_spectrum_filename ../necessary_files/p_ell_takahashi_shapenoise_kidslike.dat --processes $NPROC --realisations $NREAL --savepath $DIR
