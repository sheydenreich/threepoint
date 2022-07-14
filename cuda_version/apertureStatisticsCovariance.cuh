#ifndef APERTURESTATISTICSCOVARIANCE_CUH
#define APERTURESTATISTICSCOVARIANCE_CUH

#include <string>
#include <vector>

#include "cuda_helpers.cuh"

/**
 * @file apertureStatisticsCovariance.cuh
 * This file declares routines needed for the analytic calculation of Aperture statistic covariances with CUDA
 * There are functions for the covariance of <Map³> and <Map²>, but only the covariance for <Map³> has been rigorously tested.
 * The covariance can be calculated for different survey geometries and the formulae are derived from the real space estimator.
 * All aperture statistics use the Exponential aperture filter, first introduced by Crittenden + (2002) (https://ui.adsabs.harvard.edu/abs/2002ApJ...568...20C/abstract)
 * Routines are defined in apertureStatistics.cu
 * @author Laila Linke
 */

/***************** GENERAL DEFINITIONS **************************************************/

extern int type; // defines survey geometry, can be 0, 1, 2, 3, corresponding to: 'circle', 'square', 'infinite', 'rectangle'

/**
 * @brief Gives extent of survey [rad]. Defined in constant memory on Device
 * if type = 'circular', this is the radius.
 * if type = 'square', this is the side length.
 * if type = 'infinite', this is sqrt(Area).
 * if type = 'rectangle', this is the longer side
 */
extern __constant__ double dev_thetaMax;
extern double thetaMax; // Same as dev_thetaMax but for Host

// Area of survey [rad^2]. Defined in constant memory on Device
extern __constant__ double dev_area;

extern double area; // Same as dev_area but for host

// if type='rectangle", this is the smaller side [rad]
extern __constant__ double dev_thetaMax_smaller;
extern double thetaMax_smaller;

extern __constant__ double dev_lMin; // Minimal ell for integrations [1/rad]. Defined on Device
extern double lMin;                  // Same as dev_lMin but for host

/**
 * @brief Initialization for covariance
 * Copies all necessary constants from host to device
 * This NEEDS to be run before doing any of the calculations here!
 */
void initCovariance();

/**
 * @brief Writes a covariance matrix to a file
 * Exceptions are thrown, if file cannot be created or not the right number of values is given
 *
 * @param values Values of the covariance matrix, sorted in row-major order
 * @param N Number of rows  (= Number of cols)
 * @param filename File to which will be written.
 */
void writeCov(const std::vector<double> &values, const int &N, const std::string &filename);

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

/******************* FOR COVARIANCE OF <Map^3> *****************************************/

/**
 * @brief Calculates the first Term in the Gaussian Covariance of Map³ with all permutations
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_1, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_1 + 5 Permutations
 */
double T1_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief Calculates the second Term in the Gaussian Covariance of Map³ with all permutations. Returns 0 for type='infinite'
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_2, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_2 + 8 Permutations
 */
double T2_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief Calculates the first Term in the NonGaussian Covariance of Map³ with all permutations. Throws an exception if type='square'.
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_4, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_4 + 8 Permutations
 */
double T4_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief Calculates the second Term in the Non-Gaussian Covariance of Map³ with all permutations. Throws an exception if type is not 'infinite
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_5, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_5 + 8 Permutations
 */
double T5_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief Calculates the third Term in the Non-Gaussian Covariance of Map³ with all permutations. Throws an exception if type is not 'square'
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_6, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_6 + 5 Permutations
 */
double T6_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief Calculates the fourh Term in the Non-Gaussian Covariance of Map³ with all permutations. Throws an exception if type is not 'infinite'
 *
 * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
 * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
 * @return double T_7, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_7 (doesnt have any permutations)
 */
double T7_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

/**
 * @brief First Term of Gaussian Covariance of Map³ for one permutation
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
 * @brief Second Term of Gaussian Covariance of Map³ for one permutation
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
 * @brief First Term of NonGaussian Covariance of Map³ for one permutation
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
 * @brief Second Term of NonGaussian Covariance of Map³ for one permutation
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
 * @brief Third Term of NonGaussian Covariance of Map³ for one permutation
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
 * @brief Fourth Term of NonGaussian Covariance of Map³ for one permutation
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
 * @brief Container for variables needed for the Map³ Cov calculation
 *
 */
struct ApertureStatisticsCovarianceContainer
{
   // First aperture radii [rad]
   std::vector<double> thetas_123;

   // Second aperture radii [rad]
   std::vector<double> thetas_456;

   // Integration borders
   double lMin, lMax;     //[1/rad]
   double phiMin, phiMax; //[rad]
   double mMin, mMax;     //[Msun/h]
   double zMin, zMax;     //[unitless]
};

/************************** FOR <Map²> COVARIANCE ***************************************/

/**
 * @brief Calculates the Gaussian Covariance of Map²
 * Uses the Revised Halofit Powerspectrum and Limberintegration
 *
 * @param theta1 First Aperture radius [rad]
 * @param theta2 Second aperture Radius [rad]
 * @return double C_G(theta_1, theta2)
 */
double Cov_Map2_Gauss(const double &theta1, const double &theta2);

/**
 * @brief Calculates the Non-Gaussian Covariance of Map²
 * Uses the Trispectrum from the halomodel (Only the 1-halo term)
 *
 * @param theta1 First Aperture radius [rad]
 * @param theta2 Second aperture Radius [rad]
 * @return double C_NG(theta_1, theta2)
 */
double Cov_Map2_NonGauss(const double &theta1, const double &theta2);

/**
 * @brief Wrapper of integrand_Cov_Map2_Gauss for the cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param container Pointer to ApertureStatisticsCovarianceContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_Cov_Map2_Gauss(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

/**
 * @brief Wrapper of integrand_Cov_Map2_NonGauss for the cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param container Pointer to ApertureStatisticsCovarianceContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_Cov_Map2_NonGauss(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

/**
 * @brief Integrand of Gaussian Cov of Map2 for infinite survey
 *
 * @param vars Integration Parameters (1D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_Cov_Map2_Gauss_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);

/**
 * @brief Integrand of Gaussian Cov of Map2 for square survey
 *
 * @param vars Integration Parameters (4D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_Cov_Map2_Gauss_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);

/**
 * @brief Integrand of Non-Gaussian Cov of Map2 for infinite survey
 *
 * @param vars Integration Parameters (3D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_Cov_Map2_NonGauss_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);

/**
 * @brief Integrand of Non-Gaussian Cov of Map2 for infinite survey
 *
 * @param vars Integration Parameters (7D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_Cov_Map2_NonGauss_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value);

/**
 * @brief Container for variables needed for the Map³ Cov calculation
 *
 */
struct CovMap2Container
{
   double theta1, theta2; //[rad]

   // Integration borders
   double lMin, lMax; //[1/rad]
};

#endif // APERTURESTATISTICSCOVARIANCE_CUH