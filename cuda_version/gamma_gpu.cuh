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



// #define CUDA_ERROR_CHECK //Comment out if not necessary
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}



__constant__ double d_A96[48];
__constant__ double d_W96[48];
__constant__ double d_h,d_sigma8,d_omb,d_omc,d_ns,d_w,d_om,d_ow,d_norm;
__constant__ double d_eps=1.0e-4;
__constant__ double d_dz,d_z_max;
__constant__ int d_n_redshift_bins;
__constant__ double d_H0_over_c = 100./299792.;
__constant__ double d_c_over_H0 = 2997.92;
__constant__ double c_f_K_array[512];
__constant__ double c_g_array[512];
__constant__ double c_n_eff_array[512];
__constant__ double c_r_sigma_array[512];
__constant__ double c_D1_array[512];
__constant__ double d_array_psi[512];
__constant__ double d_array_product[512];
__constant__ double d_array_psi_J2[512];
__constant__ double d_array_product_J2[512];
__constant__ int d_prec_k;

double eps = 1.0e-4;
__global__ void write_arrays();

struct cosmology
{
    double h,sigma8,omb,omc,ns,w,om,ow;
};
struct treecorr_bin
{
	double r,u,v;
};


// Gaussian Quadrature stuff goes here.
double A96[48]={                   /* abscissas for 96-point Gauss quadrature */
    0.016276744849603,0.048812985136050,0.081297495464426,0.113695850110666,
    0.145973714654897,0.178096882367619,0.210031310460567,0.241743156163840,
    0.273198812591049,0.304364944354496,0.335208522892625,0.365696861472314,
    0.395797649828909,0.425478988407301,0.454709422167743,0.483457973920596,
    0.511694177154668,0.539388108324358,0.566510418561397,0.593032364777572,
    0.618925840125469,0.644163403784967,0.668718310043916,0.692564536642172,
    0.715676812348968,0.738030643744400,0.759602341176648,0.780369043867433,
    0.800308744139141,0.819400310737932,0.837623511228187,0.854959033434602,
    0.871388505909297,0.886894517402421,0.901460635315852,0.915071423120898,
    0.927712456722309,0.939370339752755,0.950032717784438,0.959688291448743,
    0.968326828463264,0.975939174585137,0.982517263563015,0.988054126329624,
    0.992543900323763,0.995981842987209,0.998364375863182,0.999689503883231};

  double W96[48]={                     /* weights for 96-point Gauss quadrature */
    0.032550614492363,0.032516118713869,0.032447163714064,0.032343822568576,
    0.032206204794030,0.032034456231993,0.031828758894411,0.031589330770727,
    0.031316425596861,0.031010332586314,0.030671376123669,0.030299915420828,
    0.029896344136328,0.029461089958168,0.028994614150555,0.028497411065085,
    0.027970007616848,0.027412962726029,0.026826866725592,0.026212340735672,
    0.025570036005349,0.024900633222484,0.024204841792365,0.023483399085926,
    0.022737069658329,0.021966644438744,0.021172939892191,0.020356797154333,
    0.019519081140145,0.018660679627411,0.017782502316045,0.016885479864245,
    0.015970562902562,0.015038721026995,0.014090941772315,0.013128229566962,
    0.012151604671088,0.011162102099839,0.010160770535008,0.009148671230783,
    0.008126876925699,0.007096470791154,0.006058545504236,0.005014202742928,
    0.003964554338445,0.002910731817935,0.001853960788947,0.000796792065552};

__global__ void compute_integrand_gamma0(double* d_vars, double* d_result_array, unsigned int incr, double x1, double x2, double x3);
__device__ cuDoubleComplex one_integrand_gamma0(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3);
__device__ __inline__ cuDoubleComplex full_integrand_gamma0(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3);

__device__ double f_K_at_z(double z);
__device__ double E_inv(double z);
__device__ double E(double z);
__global__ void write_f_K_array(double* d_f_K_array);
__device__ double n_of_z(double z);
__global__ void write_g_array(double* d_g_array, double* d_f_K_array);
__device__ double d_linear_pk(double k);
__device__ double d_window(double x, int i);
__device__ double d_sigmam(double r, int j);
__device__ double lgr(double z);
__device__ double lgr_func(int j, double la, double y[2]);
__global__ void write_nonlinear_scales(double* d_D1_array, double* d_r_sigma_array, double* d_n_eff_array);
__device__ double calc_r_sigma(double z, double D1);
__device__ double bispec(double k1, double k2, double k3, double z, int idx, double didx);   // non-linear BS w/o baryons [(Mpc/h)^6]
__device__ double bispec_tree(double k1, double k2, double k3, double z, double D1);  // tree-level BS [(Mpc/h)^6]
__device__ double F2(double k1, double k2, double k3, double z, double D1, double r_sigma);
__device__ double F2_tree(double k1, double k2, double k3);  // F2 kernel in tree level
__device__ int compute_coefficients(int idx, double didx, double *D1, double *r_sigma, double *n_eff);
__device__ double integrand_bkappa(double z, double ell1, double ell2, double ell3);
__device__ double g_interpolated(int idx, double didx);
__device__ double f_K_interpolated(int idx, double didx);
__host__ __device__ static __inline__ cuDoubleComplex cuCmul(cuDoubleComplex x, double y);
__host__ __device__ static __inline__ cuDoubleComplex exp_of_imag(double imag_part);
__device__ cuDoubleComplex prefactor(double x1, double x2, double x3, double phi, double psi);
__device__ double A(double psi, double x1, double x2, double phi, double varpsi);
__device__ double alpha(double psi, double x1, double x2, double phi, double varpsi);
__device__ double betabar(double psi, double phi);
__device__ double interior_angle(double an1, double an2, double opp);
__device__ double GQ96(double F(double),double a,double b);



double linear_pk(double k, cosmology cosmo, double norm);
double window(double x, int i);
double sigmam(double r, int j, cosmology cosmo, double norm);
double psi(double t);
double psip(double t);
int integrand_gamma0(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value);
std::complex<double> gamma0(double x1, double x2, double x3, double z_max, unsigned int prec_k);

struct GammaCudaContainer
{
    double x1,x2,x3;
    unsigned int prec_k;
};

#endif //GAMMA_GPU_CUH
