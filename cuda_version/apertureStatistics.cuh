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
   * @brief GPU integrandn for second order aperure statistics
   * @param ell 
   * @param z 
   * @param theta aperture radius [rad]
   * @param shapenoise_contribution whether to include shapenoise in Power Spectrum
   */
  __device__ double dev_integrand_Map2(const double& ell, const double& z, double theta, const double& shapenoise_contribution);

  /**
   * @brief CPU-interface for dev_integrand_Map2 with cubature
   * 
   * @param vars 
   * @param ndim 
   * @param npts 
   * @param theta 
   * @param value 
   * @param shapenoise_contribution 
   * @return __global__ 
   */
  __global__ void integrand_Map2(const double* vars, unsigned ndim, int npts, double theta, double* value, double shapenoise_contribution);

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
  static int integrand_Map2(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);

  /**
   * @brief Second-order aperture mass statistics, modeled from revised Halofit non-linear power spectrum with limber-integration
   * @param theta aperture radius [rad]
   * @param covPar paramfile with survey specifics (needed for shapenoise term)
   * @param shapenoise bool for shapenoise on/off
   * @return value of second order aperture statistics for aperture radius theta
   */
  double Map2(double theta, const covarianceParameters covPar, bool shapenoise);

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
   * @brief Integrand of Gaussian Covariance of MapMap
   * @param ell ell
   * @param z redshift
   * @param theta_1 First Aperture radius [rad]
   * @param theta_2 Second Aperture radius [rad]
   * @param shapenoise_contribution Contribution of shapenoise to power spectrum
   * @return value of integrand
   */
  __device__ double dev_integrand_Gaussian_Map2_covariance(const double& ell, const double& z, 
    const double& theta_1, const double& theta_2, const double& shapenoise_contribution);
  
  /**
   * @brief Wrapper for Integrand of Gaussian Covariance of MapMap
   * @param vars integration variables ell,phi
   * @param ndim number of integration variables (fixed at 2!)
   * @param npts number of points to be evaluated
   * @param theta_1 First Aperture radius [rad]
   * @param theta_2 Second Aperture radius [rad]
   * @param shapenoise_contribution Contribution of shapenoise to power spectrum
   * @param value return value of integrand
   */
  __global__ void integrand_Gaussian_Map2_Covariance(const double* vars, unsigned ndim, int npts, double theta_1, double theta_2,
    double* value, double shapenoise_contribution);

  /**
   * @brief Integrand for cubature library
   * See https://github.com/stevengj/cubature for documentation
   * @param ndim Number of dimensions of integral (here: 2, automatically assigned by integration)
   * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
   * @param vars Array containing integration variables
   * @param thisPts Pointer to ApertureStatisticsCovarianceContainer instance
   * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
   * @param value Value of integral
   * @return 0 on success
   */
  int integral_Gaussian_Map2_Covariance(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);


    /**
   * @brief Gaussian Covariance of Aperturestatistics calculated from Bispectrum
   * This uses the hcubature_v routine from the cubature library
   * @param theta_1 First Aperture radius [rad]
   * @param theta_2 Second Aperture radius [rad]
   * @param shapenoise Include shapenoise
   * @param covPar struct with parameters for covariance calculation [survey area, variance of shapenoise, galaxy number density]
   */
  double Gaussian_Map2_Covariance(double theta_1, double theta_2, const covarianceParameters covPar, bool shapenoise);


  /**
   * @brief Eq. (137) of Laila's derivation
   * @param ell_x x-coordinate of ell-vector
   * @param ell_y y-coordinate of ell-vector
   * @param theta_max side-length of square survey
   */
  __device__ double G_A(double ell_x, double ell_y, double theta_max);

  /**
   * @brief First integrand in (136) of Laila's derivation
   * @param ell ell-vector
   * @param z redshift
   * @param theta_1 first aperture radius [rad]
   * @param theta_2 second aperture radius [rad]
   * @param shapenoise_contribution value of shapenoise contribution to power specturm
   */
  __device__ double dev_integrand_1_for_L2(double ell, double z, double theta_1, double theta_2, double shapenoise_contribution);

  /**
   * @brief Standard wrappers for cubature integration on GPU of dev_integrand_1_for_L2
   */
  __global__ void integrand_1_for_L2(const double* vars, unsigned ndim, int npts, double theta_1, double theta_2,
    double* value, double shapenoise_contribution);
    int integrand_1_for_L2(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);
  

  /**
   * @brief Second integrand in (136) of Laila's derivation
   * @param ell_x x-coordinate of ell-vector
   * @param ell_y y-coordinate of ell-vector
   * @param z redshift
   * @param theta_1 first aperture radius [rad]
   * @param theta_2 second aperture radius [rad]
   * @param theta_max side-length of square survey
   * @param shapenoise_contribution value of shapenoise contribution to power specturm
   */
  __device__ double integrand_2_for_L2(double ell_x, double ell_y, double theta_1, double theta_2, double theta_max, double shapenoise_contribution);

  /**
   * @brief Standard wrappers for cubature integration on GPU of dev_integrand_2_for_L2
   */
  __global__ void integrand_2_for_L2(const double* vars, unsigned ndim, int npts, double theta_1, double theta_2, double theta_max,
    double* value, double shapenoise_contribution);
    int integrand_2_for_L2(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);
  
  /**
   * @brief L2 function defined in Eq.(136) of Laila's derivation
   * 
   * @param theta_1 input aperture radius [rad]
   * @param theta_2 input aperture radius [rad]
   * @param theta_3 input aperture radius [rad]
   * @param theta_4 input aperture radius [rad]
   * @param theta_5 input aperture radius [rad]
   * @param theta_6 input aperture radius [rad] 
   * @param covPar container containing covariance parameters
   * @param shapenoise flag for including shapenoise in Power Spectrum
   * @return double 
   */
    double L2(double theta_1, double theta_2, double theta_3, double theta_4, double theta_5, double theta_6,
      const covarianceParameters covPar, bool shapenoise);
    
  /**
   * @brief Finite-field term for Gaussian covariance of third-order aperture statistics
   * 
   * @param thetas_123 first three filter radii [rad]
   * @param thetas_456 second three filter radii [rad]
   * @param covPar container containing covariance parameters
   * @param shapenoise flag for including shapenoise in Power Spectrum
   * @return double 
   */
    double Gaussian_MapMapMap_Covariance_term2(double* thetas_123, double* thetas_456, const covarianceParameters covPar, bool shapenoise);

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
  double theta_1;
  double theta_2;
  double shapenoise_powerspectrum;
  double theta_max;
  double* thetas_123;
  double* thetas_456;
};




#endif //APERTURESTATISTICS_CUH
