#ifndef GAMMA_GPU_CUH
#define GAMMA_GPU_CUH


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

#define PERFORM_SUM_REDUCTION
#define CONVERT_TO_CENTROID
// #define DEBUG_OUTPUT


static const double prec_h = 0.1;
static const int prec_k = int(3.31/prec_h);
extern __constant__ double dev_array_psi[prec_k];
extern __constant__ double dev_array_product[prec_k];
extern __constant__ double dev_array_psi_J2[prec_k];
extern __constant__ double dev_array_product_J2[prec_k];
extern __constant__ int dev_prec_k;

struct treecorr_bin
{
	double r,u,v;
};

std::complex<double> gamma0(double x1, double x2, double x3, double z_max);
std::complex<double> gamma1(double x1, double x2, double x3, double z_max);
std::complex<double> gamma2(double x1, double x2, double x3, double z_max);
std::complex<double> gamma3(double x1, double x2, double x3, double z_max);


__global__ void compute_integrand_gamma0(double* d_vars, double* d_result_array, unsigned int max_idx, double x1, double x2, double x3);
__global__ void compute_integrand_gamma1(double* d_vars, double* d_result_array, unsigned int max_idx, double x1, double x2, double x3);



__device__ cuDoubleComplex one_integrand_gamma0(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3);
__device__ __inline__ cuDoubleComplex full_integrand_gamma0(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3);
__device__ cuDoubleComplex full_integrand_gamma1(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3);



__host__ __device__ static __inline__ cuDoubleComplex cuCmul(cuDoubleComplex x, double y);
__host__ __device__ static __inline__ cuDoubleComplex exp_of_imag(double imag_part);
__device__ cuDoubleComplex prefactor(double x1, double x2, double x3, double phi, double psi);
__device__ double A(double psi, double x1, double x2, double phi, double varpsi);
__device__ double alpha(double psi, double x1, double x2, double phi, double varpsi);
__device__ double betabar(double psi, double phi);
__device__ __host__ double interior_angle(double an1, double an2, double opp);
std::complex<double> convert_orthocenter_to_centroid(std::complex<double> gamma, double x1, double x2, double x3, bool conjugate_phi1);
double one_rotation_angle_otc(double x1, double x2, double x3);
double height_of_triangle(double x1, double x2, double x3);

double psi(double t);
double psip(double t);
int integrand_gamma0(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value);
int integrand_gamma1(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value);


void compute_weights_bessel();

struct GammaCudaContainer
{
    double x1,x2,x3;
};

#endif //GAMMA_GPU_CUH
