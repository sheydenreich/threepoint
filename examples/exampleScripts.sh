##############
## EXAMPLES FOR USAGE OF THESE CODES
## @author Laila Linke, llinke@astro.uni-bonn.de
###

DIR_BIN=../cuda_version/
DIR_RESULTS=./
FILE_NZ=exampleNz.dat
FILE_THETAS=exampleThetas.dat
FILE_COSMOLOGY=exampleCosmology.param
FILE_COVARIANCE=exampleCovariance.param
FILE_GAMMA=exampleGammaconfig.param

### CALCULATION OF <Map3>
$DIR_BIN/calculateApertureStatistics.x $FILE_COSMOLOGY $FILE_THETAS $DIR_RESULTS/Map3.dat $FILE_NZ

### CALCULATION OF 3PT CORR FUNC
$DIR_BIN/calculateGamma.x $FILE_COSMOLOGY $FILE_GAMMA $DIR_RESULTS/Gamma.dat $FILE_NZ

