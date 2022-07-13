# SLICS
# The SLICS are 10 x 10 sq.deg large and have a shapenoise of \sigma=0.37
# This script creates GRFs and measures Map3 in them


NPROC=64
NREAL=4096


DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/SLICS_theta_4_to_32/
mkdir -p $DIR

python ../python_scripts/computeMap3_SLICS.py --processes $NPROC --savepath $DIR
