#ifndef GAMMA_GPU_CUH
#define GAMMA_GPU_CUH

// THIS FUNCTIONS NEED MORE COMMENTS

#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_math.h>
#include <complex>
#include "cubature.h"
#include <cuComplex.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <memory>
#include <functional>

#include "bispectrum.cuh"

#define PERFORM_SUM_REDUCTION // Perform a sum reduction on shared memory of GPU. Alternative: Perform the sum in Host
#define CONVERT_TO_CENTROID // Convert the triangle center of the 3pcf to the centroid. Alternative: Orthocenter
// #define DEBUG_OUTPUT

// Accuracy settings for the Ogata+(2005) integration method of Bessel functions
static const double prec_h = 0.1; // smaller prec_h -> finer steps
static const int prec_k = int(3.31 / prec_h); // larger prec_h -> more zeros of the Bessel-function (larger maximum integral border)
extern __constant__ int dev_prec_k;

// Array for pre-computed weights of the Bessel integration from Ogata+(2005)
extern __constant__ double dev_array_psi[prec_k];
extern __constant__ double dev_array_product[prec_k];
extern __constant__ double dev_array_psi_J2[prec_k];
extern __constant__ double dev_array_product_J2[prec_k];

// r, u and v values for a triangle configuration according to Jarvis+(2004)
struct treecorr_bin
{
    double r, u, v;
};

/**
 * @brief First natural component of the shear 3pcf
 * \f$ \Gamma^0 $\f
 * x1, x2 and x3 are oriented counter-clockwise! Compare Schneider+(2005)
 * From Eq.(18) of Heydenreich+(2022) or Eq.(15) of Schneider+(2005)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param x3 third triangle sidelength [rad]
 * @param z_max maximum redshift of integration
 * @return std::complex<double> Gamma^0(x1,x2,x3)
 */
std::complex<double> gamma0(double x1, double x2, double x3, double z_max, int zbin1, int zbin2, int zbin3, double * dev_g, double * dev_p, int Ntomo);

/**
 * @brief Second natural component of the shear 3pcf
 * \f$ \Gamma^1 $\f
 * x1, x2 and x3 are oriented counter-clockwise! Compare Schneider+(2005)
 * From Eq.(18) of Heydenreich+(2022) or Eq.(18) of Schneider+(2005)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param x3 third triangle sidelength [rad]
 * @param z_max maximum redshift of integration
 * @return std::complex<double> Gamma^1(x1,x2,x3)
 */
std::complex<double> gamma1(double x1, double x2, double x3, double z_max, int zbin1, int zbin2, int zbin3, double * dev_g, double * dev_p, int Ntomo);

/**
 * @brief Third natural component of the shear 3pcf
 * \f$ \Gamma^2 $\f
 * x1, x2 and x3 are oriented counter-clockwise! Compare Schneider+(2005)
 * From Eq.(18) of Heydenreich+(2022) or Eq.(18) of Schneider+(2005)
 * Achieved by calculating Gamma^1(x2,x3,x1)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param x3 third triangle sidelength [rad]
 * @param z_max maximum redshift of integration
 * @return std::complex<double> Gamma^2(x1,x2,x3)
 */
std::complex<double> gamma2(double x1, double x2, double x3, double z_max, int zbin1, int zbin2, int zbin3, double * dev_g, double * dev_p, int Ntomo);

/**
 * @brief Fourth natural component of the shear 3pcf
 * \f$ \Gamma^3 $\f
 * x1, x2 and x3 are oriented counter-clockwise! Compare Schneider+(2005)
 * From Eq.(18) of Heydenreich+(2022) or Eq.(18) of Schneider+(2005)
 * Achieved by calculating Gamma^1(x3,x1,x2)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param x3 third triangle sidelength [rad]
 * @param z_max maximum redshift of integration
 * @return std::complex<double> Gamma^3(x1,x2,x3)
 */
std::complex<double> gamma3(double x1, double x2, double x3, double z_max, int zbin1, int zbin2, int zbin3, double * dev_g, double * dev_p, int Ntomo);

/**
 * @brief Wrapper to perform the integration for the calculation of Gamma^0/1
 * @param d_vars array of the variables [z,phi,psi]
 * @param d_result_array array for the integration results
 * @param max_idx maximum value of simultaneously executed calculations (GPU-dependent)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param x3 third triangle sidelength [rad]
 */
__global__ void compute_integrand_gamma0(double *d_vars, double *d_result_array, unsigned int max_idx, double x1, double x2, double x3, int zbin1, int zbin2, int zbin3, 
double *dev_g, double *dev_p, int Ntomo);
__global__ void compute_integrand_gamma1(double *d_vars, double *d_result_array, unsigned int max_idx, double x1, double x2, double x3, int zbin1, int zbin2, int zbin3, 
double *dev_g, double *dev_p, int Ntomo);

/**
 * @brief One of the three terms for the integration of Gamma^0
 * @param phi phi-value (see Eq. 18 of Heydenreich+2022)
 * @param psi psi-value (see Eq. 18 of Heydenreich+2022)
 * @param z redshift
 * @param k index for the sum in the Bessel integration (see Ogata+2005)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param x3 third triangle sidelength [rad]
 */
__device__ cuDoubleComplex one_integrand_gamma0(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3, int zbin1, int zbin2, int zbin3, 
double *dev_g, double *dev_p, int Ntomo);

/**
 * @brief Sum of the three terms for the integration of Gamma^0
 * @param phi phi-value (see Eq. 18 of Heydenreich+2022)
 * @param psi psi-value (see Eq. 18 of Heydenreich+2022)
 * @param z redshift
 * @param k index for the sum in the Bessel integration (see Ogata+2005)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param x3 third triangle sidelength [rad]
 */
__device__ __inline__ cuDoubleComplex full_integrand_gamma0(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3, int zbin1, int zbin2, int zbin3, 
double *dev_g, double *dev_p, int Ntomo);

/**
 * @brief Sum of the three terms for the integration of Gamma^1
 * @param phi phi-value (see Eq. 18 of Heydenreich+2022)
 * @param psi psi-value (see Eq. 18 of Heydenreich+2022)
 * @param z redshift
 * @param k index for the sum in the Bessel integration (see Ogata+2005)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param x3 third triangle sidelength [rad]
 */
__device__ cuDoubleComplex full_integrand_gamma1(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3, int zbin1, int zbin2, int zbin3, 
double *dev_g, double *dev_p, int Ntomo);

/**
 * @brief Multiplication of two complex numbers
 * @param x first number
 * @param y second number
 * @return (x1+i*x2)*(y1+i*y2)
 */
__host__ __device__ static __inline__ cuDoubleComplex cuCmul(cuDoubleComplex x, double y);

/**
 * @brief Exponential of i*x
 * @param x argument of exponential
 * @return exp(i*x) 
 */
__host__ __device__ static __inline__ cuDoubleComplex exp_of_imag(double x);

/**
 * @brief prefactor for the integration of Gamma^0
 * /f $exp(2i\bar{\beta}+i*(phi1-phi2-6*alpha3)) $ /f
 * See Eq. (18) in Heydenreich+(2022)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param x3 third triangle sidelength [rad]
 * @param phi phi-value (see Eq. 18 of Heydenreich+2022)
 * @param psi psi-value (see Eq. 18 of Heydenreich+2022)
 */
__device__ cuDoubleComplex prefactor(double x1, double x2, double x3, double phi, double psi);

/**
 * @brief A'_3 as defined in Eq.(19) of Heydenreich+(2022)
 * @param psi psi-value (see Eq. 18 of Heydenreich+2022)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param phi phi-value (see Eq. 18 of Heydenreich+2022)
 * @param varpsi interior angle of triangle, corresponds to psi_3 in Eq. (19) of Heydenreich+(2022)
 */
__device__ double A(double psi, double x1, double x2, double phi, double varpsi);

/**
 * @brief alpha_3 as defined in Eq.(22,23) of Heydenreich+(2022)
 * @param psi psi-value (see Eq. 18 of Heydenreich+2022)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param phi phi-value (see Eq. 18 of Heydenreich+2022)
 * @param varpsi interior angle of triangle, corresponds to psi_3 in Eq. (22,23) of Heydenreich+(2022)
 */
 __device__ double alpha(double psi, double x1, double x2, double phi, double varpsi);

 /**
 * @brief \bar{\beta} as defined in Eq.(20,21) of Heydenreich+(2022)
 * @param psi psi-value (see Eq. 18 of Heydenreich+2022)
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param phi phi-value (see Eq. 18 of Heydenreich+2022)
 * @param varpsi interior angle of triangle, corresponds to psi_3 in Eq. (20,21) of Heydenreich+(2022)
 */
__device__ double betabar(double psi, double phi);

/**
 * @brief Interior angle of a triangle between an1, an2 and opposite to opp
 * @param an1 first side of angle
 * @param an2 second side of angle
 * @param opp side opposing the angle
 * @return interior angle [rad] 
 */
__device__ __host__ double interior_angle(double an1, double an2, double opp);

/**
 * @brief Converts a shear 3pcf from orthocenter to the centroid
 * 
 * @param gamma shear 3pcf in orthocenter
 * @param x1 first triangle sidelength [rad]
 * @param x2 second triangle sidelength [rad]
 * @param x3 third triangle sidelength [rad]
 * @param conjugate_phi1 false for gamma0, true for gamma1
 * @return shear 3pcf in centroid
 */
std::complex<double> convert_orthocenter_to_centroid(std::complex<double> gamma, double x1, double x2, double x3, bool conjugate_phi1);

/**
 * @brief Calculates one rotation angle
 * According to Eq. (14) of Schneider et al. (2003)
 * @param x1 first triangle sidelength
 * @param x2 second triangle sidelength
 * @param x3 third triangle sidelength
 * @return rotation angle
 */
double one_rotation_angle_otc(double x1, double x2, double x3);

/**
 * @brief Calculates the height of the triangle
 * The length of the line that is orthogonal to x3 and connects to the intersection of x1 and x2
 * @param x1 first triangle sidelength
 * @param x2 second triangle sidelength
 * @param x3 third triangle sidelength
 * @return height of the triangle 
 */
double height_of_triangle(double x1, double x2, double x3);

/**
 * @brief Parametrised weighting function for the integration routine from Ogata+(2005)
 * /f $psi(t)=t*\tanh(\pi*\sinh(t)/2)$ /f
 * @param t input parameter
 * @return psi(t) 
 */
double psi(double t);

/**
 * @brief Derivative of the function psi(t)
 * /f $psip(t)=\frac{\sinh(\pi * \sinh(t)) + \pi * t * \cosh(t)}{\cosh(\pi * \sinh(t)) + 1}$ /f
 * @param t input parameter
 * @return (\frac{d}{dt}psi)(t)
 */
double psip(double t);

/**
 * @brief Cubature-wrapper for the integration of Gamma^0
 * 
 * @param ndim Number of dimensions of integral (here: 3, automatically assigned by integration)
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param fdata Pointer to GammaCudaContainer instance
 * @param fdim Dimensions of integral output (here: 2, automatically assigned by integration)
 * @param value Value of integral
 * @return 0 on success 
 */
int integrand_gamma0(unsigned ndim, size_t npts, const double *vars, void *fdata, unsigned fdim, double *value);

/**
 * @brief Cubature-wrapper for the integration of Gamma^1
 * 
 * @param ndim Number of dimensions of integral (here: 3, automatically assigned by integration)
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param fdata Pointer to GammaCudaContainer instance
 * @param fdim Dimensions of integral output (here: 2, automatically assigned by integration)
 * @param value Value of integral
 * @return 0 on success 
 */
int integrand_gamma1(unsigned ndim, size_t npts, const double *vars, void *fdata, unsigned fdim, double *value);

/**
 * @brief Calculates the weights for the Bessel integration in Ogata+(2005)
 * 
 */
void compute_weights_bessel();

/**
 * @brief Container for the triangle side-lengths x1,x2,x3
 * 
 */
struct GammaCudaContainer
{
    double x1, x2, x3;
    int zbin1, zbin2, zbin3;
    double *dev_g, *dev_p;
    int Ntomo;
};

#endif // GAMMA_GPU_CUH
