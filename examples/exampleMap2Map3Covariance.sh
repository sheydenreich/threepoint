##############
## EXAMPLES FOR USAGE OF MAP3_MAP2 CROSS COVARIANCE CALCULATION
## @author Laila Linke, llinke@astro.uni-bonn.de
###

DIR_BIN=../cuda_version/
DIR_RESULTS=./
FILE_NZ=exampleNz.dat
FILE_THETAS=exampleThetas.dat
FILE_COSMOLOGY=exampleCosmology.param
FILE_COVARIANCE=exampleCovariance.param


# T3 for infinite survey
$DIR_BIN/calculateMap2Map3Covariance.x $FILE_COSMOLOGY $FILE_THETAS $FILE_NZ $DIR_RESULTS $FILE_COVARIANCE 0 1 0 infinite

# T4 for infinite survey
$DIR_BIN/calculateMap2Map3Covariance.x $FILE_COSMOLOGY $FILE_THETAS $FILE_NZ $DIR_RESULTS $FILE_COVARIANCE 0 0 1 infinite 

# T2 for square survey
$DIR_BIN/calculateMap2Map3Covariance.x $FILE_COSMOLOGY $FILE_THETAS $FILE_NZ $DIR_RESULTS $FILE_COVARIANCE 1 0 0 square 


