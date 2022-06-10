#ifndef APERTURESTATISTICS_CUH
#define APERTURESTATISTICS_CUH

#include <vector>

#include "cuda_helpers.cuh"

#include "bispectrum.cuh"

/**
 * @file apertureStatistics.cuh
 * This file declares routines needed for the aperture statistics calculation with CUDA
 * Routines are defined in apertureStatistics.cu
 * These functions mirror the methods of class apertureStatistics in the pure c++ version
 * @author Laila Linke
 */

/**
 * @brief Filter function in Fourier space (Crittenden)
 * \f$ \hat{u}(\eta)=\frac{\eta^2}{2}\exp(-\frac{\eta^2}{2}) \f$
 * @param eta Parameter of filter function (=l*theta)
 * @return Value of filter function
 */
__device__ double uHat(double eta);

/**
 * @brief product of uHat Filter functions in Fourier space (Crittenden)
 * @param l1 ell1
 * @param l2 ell2
 * @param l3 ell3
 * @param thetas array of theta1,theta2,theta3 [rad]
 * @return Value of uHat(l1*theta1)*uHat(l2*theta2)*uHat(l3*theta3)
 */
__device__ double uHat_product(const double &l1, const double &l2, const double &l3, double *thetas);

/**
 * @brief all permutations for product of uHat Filter functions in Fourier space (Crittenden)
 * @param l1 ell1
 * @param l2 ell2
 * @param l3 ell3
 * @param thetas array of theta1,theta2,theta3 [rad]
 * @return Sum of all six uHat_product permutations
 */
__device__ double uHat_product_permutations(const double &l1, const double &l2, const double &l3, double *thetas);

/**
 * @brief GPU integrandn for second order aperure statistics
 * @param ell
 * @param z
 * @param theta aperture radius [rad]
 */
__device__ double dev_integrand_Map2(const double &ell, const double &z, double theta);

/**
 * @brief CPU-interface for dev_integrand_Map2 with cubature
 *
 * @param vars
 * @param ndim
 * @param npts
 * @param theta
 * @param value
 * @return __global__
 */
__global__ void integrand_Map2(const double *vars, unsigned ndim, int npts, double theta, double *value);

/**
 * @brief Cubature-wrapper for integrand_Map2 function
 *
 * @param ndim
 * @param npts
 * @param vars
 * @param thisPtr
 * @param fdim
 * @param value
 * @return int
 */
static int integrand_Map2(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

/**
 * @brief Second-order aperture mass statistics, modeled from revised Halofit non-linear power spectrum with limber-integration
 * @param theta aperture radius [rad]
 * @return value of second order aperture statistics for aperture radius theta
 */
double Map2(double theta);

// THE MAP2 COVARIANCES NEED TO CHECKED --> ARE THESE STILL VALID?

// /**
//  * @brief Integrand of Gaussian Covariance of MapMap
//  * @param ell ell
//  * @param z redshift
//  * @param theta_1 First Aperture radius [rad]
//  * @param theta_2 Second Aperture radius [rad]
//  * @param shapenoise_contribution Contribution of shapenoise to power spectrum
//  * @return value of integrand
//  */
// __device__ double dev_integrand_Gaussian_Map2_covariance(const double& ell, const double& z,
//   const double& theta_1, const double& theta_2, const double& shapenoise_contribution);

// /**
//  * @brief Wrapper for Integrand of Gaussian Covariance of MapMap
//  * @param vars integration variables ell,phi
//  * @param ndim number of integration variables (fixed at 2!)
//  * @param npts number of points to be evaluated
//  * @param theta_1 First Aperture radius [rad]
//  * @param theta_2 Second Aperture radius [rad]
//  * @param shapenoise_contribution Contribution of shapenoise to power spectrum
//  * @param value return value of integrand
//  */
// __global__ void integrand_Gaussian_Map2_Covariance(const double* vars, unsigned ndim, int npts, double theta_1, double theta_2,
//   double* value, double shapenoise_contribution);

// /**
//  * @brief Integrand for cubature library
//  * See https://github.com/stevengj/cubature for documentation
//  * @param ndim Number of dimensions of integral (here: 2, automatically assigned by integration)
//  * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
//  * @param vars Array containing integration variables
//  * @param thisPts Pointer to ApertureStatisticsCovarianceContainer instance
//  * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
//  * @param value Value of integral
//  * @return 0 on success
//  */
// int integral_Gaussian_Map2_Covariance(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);

//   /**
//  * @brief Gaussian Covariance of Aperturestatistics calculated from Bispectrum
//  * This uses the hcubature_v routine from the cubature library
//  * @param theta_1 First Aperture radius [rad]
//  * @param theta_2 Second Aperture radius [rad]
//  * @param shapenoise Include shapenoise
//  * @param covPar struct with parameters for covariance calculation [survey area, variance of shapenoise, galaxy number density]
//  */
// double Gaussian_Map2_Covariance(double theta_1, double theta_2, const covarianceParameters covPar, bool shapenoise);

/**
 * @brief Integrand of MapMapMap
 * @warning This is different from Eq 58 in Schneider, Kilbinger & Lombardi (2003) because the Bispectrum is defined differently!
 * \f$ \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3)]\f$
 * @param vars Integration parameters (ell1 [1/rad], ell2 [1/rad], phi [rad])
 * @param ndim Number of dimensions of integral (here: 3)
 * @param npts Number of integration points
 * @param theta1 Aperture radii [rad]
 * @param theta2 Aperture radii [rad]
 * @param theta3 Aperture radii [rad]
 * @return value of integrand
 */
__global__ void integrand_Map3_kernel(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double *value);

/**
 * @brief Integrand for cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (here: 3, automatically assigned by integration)
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param thisPts Pointer to ApertureStatisticsContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_Map3(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

__global__ void integrand_Map4_kernel(const double *vars, unsigned ndim, int npts,
                                      double theta1, double theta2, double theta3, double theta4, double *value,
                                      double lMin, double lMax, double phiMin, double phiMax,
                                      double mMin, double mMax, double zMin, double zMax);

/**
 * @brief Wrapper of integrand_Map4_kernel for the cuba library
 * See http://www.feynarts.de/cuba/ for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param xx Array containing integration variables
 * @param ncomp Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param ff Value of integral
 * @param userdata Pointer to ApertureStatisticsContainer instance
 * @param nvec Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @return 0 on success
 */
static int integrand_Map4(const int *ndim, const double *xx,
                          const int *ncomp, double *ff, void *userdata, const int *nvec);

__global__ void integrand_Map6_kernel(const double *vars, unsigned ndim, int npts,
                                      double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value,
                                      double lMin, double lMax, double phiMin, double phiMax,
                                      double mMin, double mMax, double zMin, double zMax);

/**
 * @brief Wrapper of integrand_Map6_kernel for the cuba library
 * See http://www.feynarts.de/cuba/ for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param xx Array containing integration variables
 * @param ncomp Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param ff Value of integral
 * @param userdata Pointer to ApertureStatisticsContainer instance
 * @param nvec Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @return 0 on success
 */
static int integrand_Map6(const int *ndim, const double *xx,
                          const int *ncomp, double *ff, void *userdata, const int *nvec);
/**
 * @brief Aperturestatistics calculated from Bispectrum
 * @warning This is NOT Eq 58 from Schneider, Kilbinger & Lombardi (2003), but a third of that, due to the bispectrum definition
 * This uses the pcubature_v routine from the cubature library
 * @param thetas Aperture Radii, array should contain 3 values [rad]
 * @param phiMin Minimal phi [rad] (optional, default: 0)
 * @param phiMax Maximal phi [rad] (optional, default: 6.283185307)
 * @param lMin Minimal ell (optional, default: 1)
 */
double MapMapMap(const std::vector<double> &thetas, const double &phiMin = 0, const double &phiMax = 6.283185307, const double &lMin = 1);

double Map4(const std::vector<double> &thetas, const double &phiMin = 0, const double &phiMax = 6.283185307, const double &lMin = 1);

double Map6(const std::vector<double> &thetas, const double &phiMin = 0, const double &phiMax = 6.283185307, const double &lMin = 1);

struct ApertureStatisticsContainer
{
  /** Apertureradii [rad]*/
  std::vector<double> thetas;

  double lMin, lMax;
  double phiMin, phiMax;
  double mMin, mMax;
  double zMin, zMax;
};

#endif // APERTURESTATISTICS_CUH
