#ifndef APERTURESTATISTICS_CUH
#define APERTURESTATISTICS_CUH



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




#endif //APERTURESTATISTICS_CUH