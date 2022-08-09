# SLICS
# The SLICS are 10 x 10 sq.deg large and have a shapenoise of \sigma=0.37
# This script creates GRFs and measures Map3 in them


NPROC=6
NREAL=4096


DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/MS
#/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/SLICS_theta_4_to_32/
mkdir -p $DIR

python ../python_scripts/computeMap2_MS.py --processes $NPROC --savepath $DIR
