#ifndef APERTURESTATISTICS_CUH
#define APERTURESTATISTICS_CUH

#include <vector>
#include "cuda_helpers.cuh"
#include "bispectrum.cuh"

/**
 * @file apertureStatistics.cuh
 * This file declares routines needed for the aperture statistics calculation with CUDA
 * There are functions for <Map²>, <Map³>, <Map⁴> and <Map⁶>
 * All aperture statistics use the Exponential aperture filter, first introduced by Crittenden + (2002) (https://ui.adsabs.harvard.edu/abs/2002ApJ...568...20C/abstract)
 * Routines are defined in apertureStatistics.cu
 * @author Laila Linke
 */

/**
 * @brief Filter function in Fourier space (Crittenden)
 * \f$ \hat{u}(\eta)=\frac{\eta^2}{2}\exp(-\frac{\eta^2}{2}) \f$
 * @param eta Parameter of filter function (=l*theta) [unitless]
 * @return Value of filter function
 */
__device__ double uHat(double eta);

/**
 * @brief Product of three uHat Filter functions in Fourier space (Crittenden)
 * \f$ \hat{u}(\ell_1\theta_1)\hat{u}(\ell_2\theta_2)\hat{u}(\ell_3\theta_3)
 * @param l1 ell1 [1/rad]
 * @param l2 ell2 [1/rad]
 * @param l3 ell3 [1/rad]
 * @param thetas array of theta1,theta2,theta3 [rad]
 * @return Value of uHat(l1*theta1)*uHat(l2*theta2)*uHat(l3*theta3)
 */
__device__ double uHat_product(const double &l1, const double &l2, const double &l3, double *thetas);

/**
 * @brief Integrand of <Map²>, interface to GPU
 * \f$ \ell P(\ell) \hat{u}^2(\theta\ell)\f$
 * @param vars Integration parameter ell [1/rad]
 * @param ndim Number of dimensions of integral (here: 1)
 * @param npts Number of integration points
 * @param theta Aperture radius [rad]
 * @param value value of integrand
 */
__global__ void integrand_Map2_kernel(const double *vars, unsigned ndim, int npts, double theta, double *value);

/**
 * @brief Integrand of <Map²> for cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (here: 1, automatically assigned by integration)
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param thisPts Pointer to ApertureStatisticsContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
 * @param value Value of integral
 * @return 0 on success
 */
static int integrand_Map2(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

/**
 * @brief <Map²>, modeled from revised Halofit non-linear power spectrum with limber-integration
 * See https://ui.adsabs.harvard.edu/abs/2012ApJ...761..152T/abstract for Revised Halofit
 * Assumes Flat-Sky
 * Uses hcubature_v routine from the cubature library
 * @param theta aperture radius [rad]
 * @return value of <Map²> for aperture radius theta
 */
double Map2(double theta);

/**
 * @brief Integrand of <Map³>, interface to GPU
 * \f$ \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3)]\f$
 * @warning This is different from Eq 58 in Schneider, Kilbinger & Lombardi (2003) because the Bispectrum is defined differently!
 * @param vars Integration parameters (ell1 [1/rad], ell2 [1/rad], phi [rad])
 * @param ndim Number of dimensions of integral (here: 3)
 * @param npts Number of integration points
 * @param theta1 Aperture radii [rad]
 * @param theta2 Aperture radii [rad]
 * @param theta3 Aperture radii [rad]
 * @param value value of integrand
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

/**
 * @brief <Map³>, modelled from BiHalofit Bispectrum with Limber-Integration
 * @warning This is NOT Eq 58 from Schneider, Kilbinger & Lombardi (2003), but a third of that, due to the bispectrum definition
 * This uses the hcubature_v routine from the cubature library
 * @param thetas Aperture Radii [rad], array should contain 3 values, program stops if other number of values is given
 * @param phiMin Minimal phi [rad] (optional, default: 0)
 * @param phiMax Maximal phi [rad] (optional, default: 2pi)
 * @param lMin Minimal ell (optional, default: 1)
 */
double MapMapMap(const std::vector<double> &thetas, const double &phiMin = 0, const double &phiMax = 2 * M_PI, const double &lMin = 1);

/**
 * @brief Integrand of <Map⁴>, interface to GPU
 * \f$ \ell_1 \ell_2 \ell_3 T(\ell_1, \ell_2, \ell_3, \ell_4)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3)\hat{u}(\theta_4\ell_4)]\f$
 * where $\ell_4=\ell_1^2 + \ell_2^2 + \ell_3^2 + 2 \ell_1\ell_2\cos(\phi_2 - \phi_1) + 2 \ell_2\ell_3\cos(\phi_2 - \phi_3) + 2 \ell_1\ell_3\cos(\phi_3 - \phi_1)\f$
 * @param vars Integration parameters (ell1 [1/rad], ell2 [1/rad], ell3 [1/rad], phi1 [rad], phi2 [rad], phi3 [rad], m [Msun/h], z)
 * @param ndim Number of dimensions of integral (here: 8)
 * @param npts Number of integration points
 * @param theta1 Aperture radii [rad]
 * @param theta2 Aperture radii [rad]
 * @param theta3 Aperture radii [rad]
 * @param theta4 Aperture radii [rad]
 * @param value value of integrand
 * @param lMin minimal ell of integration [1/rad]
 * @param lMax maximal ell of integration [1/rad]
 * @param phiMin minimal phi of integration [rad]
 * @param phiMax maximal phi of integration [rad]
 * @param mMin minimal halo mass [Msun/h]
 * @param mMax maximal halo mass [Msun/h]
 * @param zMin minimal redshift
 * @param zMax maximal redshift
 */
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

/**
 * @brief <Map⁴>, modelled with Trispectrum from halomodel. Trispectrum contains only 1-halo term!
 * This uses CUBA for the integration
 * @param thetas Aperture Radii [rad], array should contain 4 values, program stops if other number of values is given
 * @param phiMin Minimal phi [rad] (optional, default: 0)
 * @param phiMax Maximal phi [rad] (optional, default: 2pi)
 * @param lMin Minimal ell (optional, default: 1)
 */
double Map4(const std::vector<double> &thetas, const double &phiMin = 0, const double &phiMax = 2 * M_PI, const double &lMin = 1);

/**
 * @brief Integrand of <Map⁶>, interface to GPU
 * @param vars Integration parameters (ell1 [1/rad], ell2 [1/rad], ell3 [1/rad], ell4 [1/rad], ell5 [1/rad], phi1 [rad], phi2 [rad], phi3 [rad], phi4[rad], phi5[rad], m [Msun/h], z)
 * @param ndim Number of dimensions of integral (here: 12)
 * @param npts Number of integration points
 * @param theta1 Aperture radii [rad]
 * @param theta2 Aperture radii [rad]
 * @param theta3 Aperture radii [rad]
 * @param theta4 Aperture radii [rad]
 * @param theta5 Aperture radii [rad]
 * @param theta6 Aperture radii [rad]
 * @param value value of integrand
 * @param lMin minimal ell of integration [1/rad]
 * @param lMax maximal ell of integration [1/rad]
 * @param phiMin minimal phi of integration [rad]
 * @param phiMax maximal phi of integration [rad]
 * @param mMin minimal halo mass [Msun/h]
 * @param mMax maximal halo mass [Msun/h]
 * @param zMin minimal redshift
 * @param zMax maximal redshift
 */
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
 * @brief <Map⁶>, modelled with Pentaspectrum from halomodel. Pentaspectrum contains only 1-halo term!
 * This uses CUBA for the integration
 * @param thetas Aperture Radii [rad], array should contain 4 values, program stops if other number of values is given
 * @param phiMin Minimal phi [rad] (optional, default: 0)
 * @param phiMax Maximal phi [rad] (optional, default: 2pi)
 * @param lMin Minimal ell (optional, default: 1)
 */
double Map6(const std::vector<double> &thetas, const double &phiMin = 0, const double &phiMax = 6.283185307, const double &lMin = 1);

/**
 * @brief Container for variables needed in the aperture statistics integrations
 *
 */
struct ApertureStatisticsContainer
{
  /** Apertureradii [rad]*/
  std::vector<double> thetas;

  // Integration borders
  double lMin, lMax;     //[1/rad]
  double phiMin, phiMax; //[rad]
  double mMin, mMax;     //[Msun/h]
  double zMin, zMax;     //[unitless]
};

#endif // APERTURESTATISTICS_CUH
