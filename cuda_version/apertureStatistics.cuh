#ifndef APERTURESTATISTICS_CUH
#define APERTURESTATISTICS_CUH

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
__device__ double uHat_product(const double& l1, const double& l2, const double& l3, double* thetas);

  /**
   * @brief all permutations for product of uHat Filter functions in Fourier space (Crittenden)
   * @param l1 ell1
   * @param l2 ell2
   * @param l3 ell3
   * @param thetas array of theta1,theta2,theta3 [rad]
   * @return Sum of all six uHat_product permutations
   */
  __device__ double uHat_product_permutations(const double& l1, const double& l2, const double& l3, double* thetas);

    /**
   * @brief Integrand of Gaussian Covariance of MapMapMap
   * @warning derived by Laila Linke, not yet published
   * @param l1 ell1
   * @param l2 ell2
   * @param phi angle between l1 and l2 [rad]
   * @param z redshift
   * @param thetas_123 Aperture radii 1,2,3 [rad]
   * @param thetas_456 Aperture radii 1,2,3 [rad]
   * @return value of integrand
   */
  __device__ double dev_integrand_Gaussian_Aperture_Covariance(const double& l1, const double& l2, const double& phi, const double& z, 
    double* thetas_123, double* thetas_456, const double& shapenoise_contribution);

    /**
   * @brief Wrapper for Integrand of Gaussian Covariance of MapMapMap
   * @param vars integration variables l1,l2,phi,z
   * @param ndim number of integration variables (fixed at 4!)
   * @param npts number of points to be evaluated
   * @param thetas_123 Aperture radii 1,2,3 [rad]
   * @param thetas_456 Aperture radii 1,2,3 [rad]
   * @param value return value of integrand
   */
  __global__ void integrand_Gaussian_Map3_Covariance(const double* vars, unsigned ndim, int npts, double* thetas_123, double* thetas_456, double* value);

    /**
   * @brief Gaussian Covariance of Aperturestatistics calculated from Bispectrum
   * This uses the hcubature_v routine from the cubature library
   * @param thetas_123 Aperture Radii, array should contain 3 values [rad]
   * @param thetas_456 Aperture Radii, array should contain 3 values [rad]
   * @param survey area [rad^2]
   */
   double Gaussian_MapMapMap_Covariance(double* thetas_123, double* thetas_456, const covarianceParameters covPar, bool shapenoise);

  /**
   * @brief Integrand for cubature library
   * See https://github.com/stevengj/cubature for documentation
   * @param ndim Number of dimensions of integral (here: 4, automatically assigned by integration)
   * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
   * @param vars Array containing integration variables
   * @param thisPts Pointer to ApertureStatisticsCovarianceContainer instance
   * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
   * @param value Value of integral
   * @return 0 on success
   */
  int integral_Gaussian_Map3_Covariance(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);

  /**
   * @brief Integrand of MapMapMap
   * @warning This is different from Eq 58 in Schneider, Kilbinger & Lombardi (2003) because the Bispectrum is defined differently!
   * \f$ \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3)]\f$
   * @param l1 ell1
   * @param l2 ell2
   * @param phi angle between l1 and l2 [rad]
   * @param theta1 Aperture radii [rad]
   * @param theta2 Aperture radii [rad]
   * @param theta3 Aperture radii [rad]
   * @return value of integrand
   */
__global__ void integrand_Map3(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double* value);


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
int integral_Map3(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);

  /**
   * @brief Aperturestatistics calculated from Bispectrum
   * @warning This is NOT Eq 58 from Schneider, Kilbinger & Lombardi (2003), but a third of that, due to the bispectrum definition
   * This uses the hcubature_v routine from the cubature library
   * @param thetas Aperture Radii, array should contain 3 values [rad]
   * @param phiMin Minimal phi [rad]
   * @param phiMax Maximal phi [rad]
   * @param lMin Minimal ell
   */
double MapMapMap(double* thetas, const double& phiMin, const double& phiMax, const double& lMin);
  

struct ApertureStatisticsContainer
{
  /** Apertureradii [rad]*/
  double theta1;
  double theta2;
  double theta3;
};

struct ApertureStatisticsCovarianceContainer
{
  /** Apertureradii [rad]*/
  double* thetas_123;
  double* thetas_456;
  double shapenoise_powerspectrum;
};




#endif //APERTURESTATISTICS_CUH
