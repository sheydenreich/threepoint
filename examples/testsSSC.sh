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


$DIR_BIN/calculateMap3SSC.x $FILE_COSMOLOGY $FILE_THETAS $FILE_NZ $DIR_RESULTS $FILE_COVARIANCE square 


#$DIR_BIN/calculateMap2Map3Covariance.x $FILE_COSMOLOGY $FILE_THETAS $FILE_NZ $DIR_RESULTS $FILE_COVARIANCE 0 1 0 infinite

#$DIR_BIN/calculateMap2Map3Covariance.x $FILE_COSMOLOGY $FILE_THETAS $FILE_NZ $DIR_RESULTS $FILE_COVARIANCE 0 0 1 infinite 


#$DIR_BIN/calculateMap2Map3Covariance.x $FILE_COSMOLOGY $FILE_THETAS $FILE_NZ $DIR_RESULTS $FILE_COVARIANCE 1 0 0 square 


