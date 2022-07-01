#ifndef APERTURESTATISTICSCOVARIANCE_CUH
#define APERTURESTATISTICSCOVARIANCE_CUH

#include <string>
#include <vector>

#include "cuda_helpers.cuh"
extern int type; // defines survey geometry, can be 0, 1, 2, 3, corresponding to: 'circle', 'square', 'infinite', 'rectangle'

/**
 * @brief Gives extent of survey [rad].
 * if type = 'circular', this is the radius.
 * if type = 'square', this is the side length.
 * if type = 'infinite', this is sqrt(A).
 * if type = 'rectangle', this is the longer side
 */
extern __constant__ double dev_thetaMax;
extern double thetaMax;
extern double area;
extern __constant__ double dev_area;

// if type='rectangle", this is the smaller side [rad]
extern __constant__ double dev_thetaMax_smaller;
extern double thetaMax_smaller;

extern __constant__ double dev_lMin;
extern double lMin;

/**
 * @brief Writes a covariance matrix (or one part of it) to a file
 *
 * @param values Values of the covariance matrix, sorted in row-major order
 * @param N Number of rows  (= Number of cols)
 * @param filename File to which will be written. Exception is thrown, if file cannot be created (e.g. if folder doesn't exist)
 */
void writeCov(const std::vector<double> &values, const int &N, const std::string &filename);

/**
 * @brief Calculates the first Term in the Gaussian Covariance with all permutations
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_1, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_1 + 5 Permutations
 */
double T1_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief Calculates the second Term in the Gaussian Covariance with all permutations. Returns 0 for type='infinite'
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_2, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_2 + 8 Permutations
 */
double T2_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief Calculates the first Term in the NonGaussian Covariance with all permutations. Throws an exception if type='square'.
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_4, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_4 + 8 Permutations
 */
double T4_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief Calculates the second Term in the Non-Gaussian Covariance with all permutations. Throws an exception if type is not 'infinite
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_5, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_5 + 8 Permutations
 */
double T5_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief Calculates the third Term in the Non-Gaussian Covariance with all permutations. Throws an exception if type is not 'square'
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_6, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_6 + 5 Permutations
 */
double T6_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief Calculates the fourh Term in the Non-Gaussian Covariance with all permutations. Throws an exception if type is not 'infinite'
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_7, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_7 (doesnt have any permutations)
 */
double T7_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief First Term of Gaussian Covariance for one permutation
 *
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 */
double T1(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6);

/**
 * @brief Second Term of Gaussian Covariance for one permutation
 * Throws exception if type is not "circle", "square" or "rectangle"
 *
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 */
double T2(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6);

/**
 * @brief First Term of NonGaussian Covariance for one permutation
 * Throws exception if type is not "infinite"
 *
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 */
double T4(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6);

/**
 * @brief Second Term of NonGaussian Covariance for one permutation
 * Throws exception if type is not "infinite"
 *
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 */
double T5(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6);

/**
 * @brief Third Term of NonGaussian Covariance for one permutation
 * Throws exception if type is not "square" or "rectangle"
 *
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 */
double T6(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6);

/**
 * @brief Fourth Term of NonGaussian Covariance for one permutation
 * Throws exception if type is not "infinite"
 *
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 */
double T7(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6);

/**
 * @brief Wrapper of integrand_T1 for the cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param container Pointer to ApertureStatisticsCovarianceContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_T1(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

/**
 * @brief Wrapper of integrand_T2_part1 for the cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param container Pointer to ApertureStatisticsCovarianceContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_T2_part1(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

/**
 * @brief Wrapper of integrand_T2_part2 for the cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param container Pointer to ApertureStatisticsCovarianceContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_T2_part2(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

/**
 * @brief Wrapper of integrand_T4 for the cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param container Pointer to ApertureStatisticsCovarianceContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_T4(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

/**
 * @brief Wrapper of integrand_T5 for the cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param container Pointer to ApertureStatisticsCovarianceContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_T5(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

/**
 * @brief Wrapper of integrand_T6 for the cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param container Pointer to ApertureStatisticsCovarianceContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_T6(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

/**
 * @brief Wrapper of integrand_T7 for the cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param container Pointer to ApertureStatisticsCovarianceContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_T7(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

/**
 * @brief Integrand of Term1 for circular survey
 *
 * @param vars Integration parameters (6 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T1_circle(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value);

/**
 * @brief Integrand of Term1 for square survey
 *
 * @param vars Integration parameters (6 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T1_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value);

/**
 * @brief Integrand of Term1 for infinite survey
 *
 * @param vars Integration parameters (3 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T1_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value);

/**
 * @brief Integrand of Term1 for rectangular survey
 *
 * @param vars Integration parameters (6 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T1_rectangle(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value);

/**
 * @brief Integrand for first part of Term2, applicable to both circular and square survey
 *
 * @param vars Integration parameters (1 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T2_part1(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);

/**
 * @brief Integrand for second part of Term2 for circular survey
 *
 * @param vars Integration parameters (1 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T2_part2_circle(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);

/**
 * @brief Integrand for second part of Term2 for square survey
 *
 * @param vars Integration parameters (2 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T2_part2_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);


/**
 * @brief Integrand for second part of Term2 for rectangular survey
 *
 * @param vars Integration parameters (2 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T2_part2_rectangle(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);

/**
 * @brief Integrand for Term 4 for infinite survey
 *
 * @param vars Integration parameters (5 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T4_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3,
                                      double theta4, double theta5, double theta6, double *value);

/**
 * @brief Integrand for Term 5 for infinite survey
 *
 * @param vars Integration parameters (6 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T5_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3,
                                      double theta4, double theta5, double theta6, double *value);

/**
 * @brief Integrand for Term 6 for square survey
 *
 * @param vars Integration parameters (7 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T6_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3,
                                    double theta4, double theta5, double theta6, double *value);


   /**
 * @brief Integrand for Term 6 for rectangular survey
 *
 * @param vars Integration parameters (7 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T6_rectangle(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3,
                                    double theta4, double theta5, double theta6, double *value);

/**
 * @brief Integrand for Term 7 for infinite survey
 *
 * @param vars Integration parameters (6 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T7_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3,
                                      double theta4, double theta5, double theta6, double *value, double mMin, double mMax);

/**
 * @brief Geometric factor for circular survey
 *
 * @param ell |ellvec|
 */
__device__ double G_circle(const double &ell);

/**
 * @brief Geometric factor for square survey
 *
 * @param ellX ell_x
 * @param ellY ell_y
 */
__device__ double G_square(const double &ellX, const double &ellY);


/**
 * @brief Geometric factor for rectangular survey
 *
 * @param ellX ell_x
 * @param ellY ell_y
 */
__device__ double G_rectangle(const double &ellX, const double &ellY);

struct ApertureStatisticsCovarianceContainer
{
   // First aperture radii [rad]
   std::vector<double> thetas_123;

   // Second aperture radii [rad]
   std::vector<double> thetas_456;

   double lMin, lMax, phiMin, phiMax, mMin, mMax, zMin, zMax;
};

struct CovMap2Container
{
   double theta1, theta2; //[rad]

   double lMin, lMax;
};

/**
 * @brief Initialization for covariance
 * Copies thetaMax, sigma, n, lMin and whether powerspec is constant from host to device
 */
void initCovariance();

//// THE FOLLOWING IS FOR TESTING T4 FOR AN ANALYTICAL BISPECTRUM ////

__global__ void integrand_T4_testBispec_analytical(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3,
                                                   double theta4, double theta5, double theta6, double *value);

int integrand_T4_testBispec_analytical(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

double T4_testBispec_analytical(const double &theta1, const double &theta2, const double &theta3,
                                const double &theta4, const double &theta5, const double &theta6);

__global__ void integrand_T4_infinite_testBispec(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3,
                                                 double theta4, double theta5, double theta6, double *value);

int integrand_T4_testBispec(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

double T4_testBispec(const double &theta1, const double &theta2, const double &theta3,
                     const double &theta4, const double &theta5, const double &theta6);

__device__ double testBispec(double &l1, double &l2, double &l3);



//// THE FOLLOWING IS FOR THE MAP2 COVARIANCE

double Cov_Map2_Gauss(const double& theta1, const double& theta2);

int integrand_Cov_Map2_Gauss(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

__global__ void integrand_Cov_Map2_Gauss_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);

__global__ void integrand_Cov_Map2_Gauss_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);

double Cov_Map2_NonGauss(const double& theta1, const double& theta2);

int integrand_Cov_Map2_NonGauss(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

__global__ void integrand_Cov_Map2_NonGauss_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);

__global__ void integrand_Cov_Map2_NonGauss_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);

#endif // APERTURESTATISTICSCOVARIANCE_CUH