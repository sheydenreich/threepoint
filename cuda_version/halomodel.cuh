#ifndef HALOMODEL_CUH
#define HALOMODEL_CUH

/**
 * @file halomodel.cuh
 * This file declares routines based on the halomodel
 * There are functions for the 1-halo term of the powerspectrum, the trispectrum and the pentaspectrum
 * Routines are defined in halomodel.cu
 * @author Laila Linke
 */


// General definitions

const double logMmin = 9; // Minimal halo mass (log(M/(Msun/h)))
const double logMmax = 17; // Maximal halo mass (log(M/(Msun/h)))
const int n_mbins = 128; // Number of bins for halo masses
extern __constant__ double devLogMmin, devLogMmax; // Same as logMmin and logMmax for Device
extern int __constant__ dev_n_mbins; // Same as n_mbins for Device

extern double sigma2_array[n_mbins]; // \sigma^2(m), i.e. fluctuations at mass scale
extern double dSigma2dm_array[n_mbins]; // d\sigma²/dm i.e. derivative of fluctuations wrt mass scale


extern __constant__ double dev_sigma2_array[n_mbins]; // Same as sigma2_array for Device
extern __constant__ double dev_dSigma2dm_array[n_mbins]; // Same as dSigma2dm_array for device

/**
 * @brief Initialization function
 * Calculates sigma^2(m) and dsigma²/dm and stores them on GPU
 * @warning Requires that bispectrum has been initialized!
 *
 */
void initHalomodel();

/**
 * @brief Halo mass function (Sheth & Tormen 1999)
 *
 * @param m Halo mass [h^-1 Msun]
 * @param z Redshift
 * @return dn/dm [h^4 Msun^-1 Mpc^-3]
 */
__host__ __device__ double hmf(const double &m, const double &z);

/**
 * Approximation to Si(x) and Ci(x) Functions
 * Same as GSL implementation, because they are not implemented in CUDA
 * @param x x value
 * @param si will contain Si(x)
 * @param ci will contain Ci(x)
 */
__host__ __device__ void SiCi(double x, double &si, double &ci);

/**
 * @brief r200 Radius of NFW profile
 *
 * @param m Halo mass [h^-1 Msun]
 * @param z Redshift
 * @return r_200 [h^-1 Mpc]
 */
__host__ __device__ double r_200(const double &m, const double &z);

/**
 * @brief Fouriertransform of normalized, truncated NFW profile (truncated at r_200)
 *
 * @param k Wave vector [h Mpc^-1]
 * @param m Halo mass [h^-1 Msun]
 * @param z Redshift
 */
__host__ __device__ double u_NFW(const double &k, const double &m, const double &z);

/**
 * @brief Concentration-mass/redshift relation of halos. Currently Duffy+(2008)
 *
 * @param m Halo mass [h^-1 Msun]
 * @param z Redshift
 */
__host__ __device__ double concentration(const double &m, const double &z);

/**
 * @brief Matter variance on mass scale m
 * I.e. Convolution of P(k) with spherical windowfunction of radius r(m)
 * @param m Halo mass [h^-1 Msun]
 * @return double
 */
double sigma2(const double &m);

/**
 * @brief Integrand for sigma2, wrapper for cubature
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param container Pointer to ApertureStatisticsCovarianceContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_sigma2(unsigned ndim, size_t npts, const double *k, void *thisPtr, unsigned fdim, double *value);

/**
 * @brief calculates sigma2 for various masses, puts them into the sigma2_array and copies them to device
 *
 */
void setSigma2();

/**
 * @brief Calculates derivative of sigma² with mass m
 *
 * @param m mass [h^-1 Msun]
 * @return double [h/Msun]
 */
double dSigma2dm(const double &m);

/**
 * @brief Integrand for dSigma2dm, wrapper for cubature
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param container Pointer to ApertureStatisticsCovarianceContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_dSigma2dm(unsigned ndim, size_t npts, const double *k, void *thisPtr, unsigned fdim, double *value);

/**
 * @brief calculates dsigma2/dm for various masses, puts them into the sigma2_array and copies them to device
 *
 */
void setdSigma2dm();

/**
 * @brief Get sigma2 at the mass m and redshift z. Uses linear interpolation between entries in sigma2 array
 * @warning sigma2 array needs to be set beforehand
 *
 * @param m mass [h^-1 Msun]
 * @param z redshift
 * @return double
 */
__host__ __device__ double get_sigma2(const double &m, const double &z);

/**
 * @brief Get dsigma2/dm at the mass m and redshift z. Uses linear interpolation between entries in sigma2 array
 * @warning sigma2 array needs to be set beforehand
 *
 * @param m mass [h^-1 Msun]
 * @param z redshift
 * @return double
 */
__host__ __device__ double get_dSigma2dm(const double &m, const double &z);


__device__ double halo_bias(const double& m, const double& z);



/**
 * @brief 1-Halo Term integrand for 2D-Trispectrum. Needs to be integrated over mass m and redshift z to give total 2D-Trispectrum
 *
 * @param m mass [h^-1 Msun]
 * @param z redshift
 * @param l1 ell1 [1/rad]
 * @param l2 ell2 [1/rad]
 * @param l3 ell3 [1/rad]
 * @param l4 ell4 [1/rad]
 * @return Integrand for Projected trispectrum (1-halo term only)
 */
__device__ double trispectrum_integrand(double m, double z, double l1, double l2,
                                        double l3, double l4);

/**
 * @brief 1-Halo Term of projected Trispectrum with performed Limber Integration. Needs to be integrated over mass m for total 2D-Trispectrum
 * Integrates over redshift from a to b using GQ on device
 *
 * @param a Minimal redshift
 * @param b Maximal redshift
 * @param m mass [h^-1 Msun]
 * @param l1 ell1 [1/rad]
 * @param l2 ell2 [1/rad]
 * @param l3 ell3 [1/rad]
 * @param l4 ell4 [1/rad]
 * @return Integrand for Projected trispectrum (1-halo term only)
 */
__device__ double trispectrum_limber_integrated(double a, double b, double m, double l1, double l2, double l3, double l4);


/**
 * @brief 1-Halo Term integrand for 2D-Tetraspectrum. Needs to be integrated over mass m and redshift z to give total 2D-Tetraspectrum
 *
 * @param m mass [h^-1 Msun]
 * @param z redshift
 * @param l1 ell1 [1/rad]
 * @param l2 ell2 [1/rad]
 * @param l3 ell3 [1/rad]
 * @param l4 ell4 [1/rad]
 * @param l5 ell5 [1/rad]
 * @return Integrand for Projected Tetraspectrum (1-halo term only)
 */
__device__ double tetraspectrum_integrand(double m, double z, double l1, double l2, double l3, double l4, double l5);

/**
 * @brief 1-Halo Term of projected Tetraspectrum with performed Limber Integration. Needs to be integrated over mass m for total 2D-Pentaspectrum
 * Integrates over redshift from a to b using GQ on device
 *
 * @param a Minimal redshift
 * @param b Maximal redshift
 * @param m mass [h^-1 Msun]
 * @param l1 ell1 [1/rad]
 * @param l2 ell2 [1/rad]
 * @param l3 ell3 [1/rad]
 * @param l4 ell4 [1/rad]
 * @param l5 ell5 [1/rad]
 * @param l6 ell6 [1/rad]
 * @return Integrand for Projected Tetraspectrum (1-halo term only)
 */
__device__ double tetraspectrum_limber_integrated(double a, double b, double m, double l1, double l2, double l3, double l4, double l5);


/**
 * @brief 1-Halo Term integrand for 2D-Pentaspectrum. Needs to be integrated over mass m and redshift z to give total 2D-Pentaspectrum
 *
 * @param m mass [h^-1 Msun]
 * @param z redshift
 * @param l1 ell1 [1/rad]
 * @param l2 ell2 [1/rad]
 * @param l3 ell3 [1/rad]
 * @param l4 ell4 [1/rad]
 * @param l5 ell5 [1/rad]
 * @param l6 ell6 [1/rad]
 * @return Integrand for Projected Pentaspectrum (1-halo term only)
 */
__device__ double pentaspectrum_integrand(double m, double z, double l1, double l2, double l3, double l4, double l5, double l6);

__device__ double pentaspectrum_integrand_ssc(double mmin, double mmax, double z, double l1, double l2, double l3, double l4, double l5, double l6);



/**
 * @brief 1-Halo Term of projected Pentaspectrum with performed Limber Integration. Needs to be integrated over mass m for total 2D-Pentaspectrum
 * Integrates over redshift from a to b using GQ on device
 *
 * @param a Minimal redshift
 * @param b Maximal redshift
 * @param m mass [h^-1 Msun]
 * @param l1 ell1 [1/rad]
 * @param l2 ell2 [1/rad]
 * @param l3 ell3 [1/rad]
 * @param l4 ell4 [1/rad]
 * @param l5 ell5 [1/rad]
 * @param l6 ell6 [1/rad]
 * @return Integrand for Projected Pentaspectrum (1-halo term only)
 */
__device__ double pentaspectrum_limber_integrated(double a, double b, double m, double l1, double l2, double l3, double l4, double l5, double l6);


__device__ double pentaspectrum_limber_integrated_ssc(double zmin, double zmax, double mmin, double mmax, double l1, double l2, double l3, double l4, double l5, double l6);


/**
 * @brief 1-Halo Term of projected Pentaspectrum with performed Limber and Mass Integration.
 * Integrates over redshift from zmin to zmax and over log(M) from logMmin to logMmax using GQ on device
 *
 * @param zmin Minimal redshift
 * @param zmax Maximal redshift
 * @param logMmin log(minimal mass) [log(h^-1 Msun)]
 * @param logMmax log(maximal mass) [log(h^-1 Msun)]
 * @param l1 ell1 [1/rad]
 * @param l2 ell2 [1/rad]
 * @param l3 ell3 [1/rad]
 * @param l4 ell4 [1/rad]
 * @param l5 ell5 [1/rad]
 * @param l6 ell6 [1/rad]
 * @return Integrand for Projected Pentaspectrum (1-halo term only)
 */
__device__ double pentaspectrum_limber_mass_integrated(double zmin, double zmax, double logMmin, double logMmax, double l1, double l2, double l3, double l4, double l5, double l6);

/**
 * @brief 1-Halo Term integrand for 2D-Powerspectrum. Needs to be integrated over mass m and redshift z to give total 2D-Pentaspectrum
 *
 * @param m mass [h^-1 Msun]
 * @param z redshift
 * @param l ell [1/rad]
 * @return Integrand for Projected Powerspectrum (1-halo term only)
 */
__device__ double powerspectrum_integrand(double m, double z, double l);


/**
 * @brief 1-Halo Term of projected Powerspectrum with performed Limber Integration. Needs to be integrated over mass m for total 2D-Pentaspectrum
 * Integrates over redshift from a to b using GQ on device
 *
 * @param a Minimal redshift
 * @param b Maximal redshift
 * @param m mass [h^-1 Msun]
 * @param l ell [1/rad]
 * @return Integrand for Projected Powerspectrum (1-halo term only)
 */
__device__ double powerspectrum_limber_integrated(double a, double b, double m, double l);


/**
 * @brief Wrapper for integrand_powerspectrum for cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (here: 1, automatically assigned by integration)
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param thisPtr Pointer to PowerspecContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_Powerspectrum(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

/**
 * @brief Integrand of powerspectrum, interface to GPU
 * @param vars Integration parameters (m)
 * @param ndim Number of dimensions of integral (here: 1)
 * @param npts Number of integration points
 * @param l ell-value[1/rad]
 * @param value value of integrand
 */
__global__ void integrand_Powerspectrum_kernel(const double *vars, unsigned ndim, int npts, double l, double *value);

/**
 * @brief 1-halo term of projected Powerspectrum
 * 
 * @param l ell-value [1/rad]
 */
double Powerspectrum(const double &l);


/**
 * @brief Wrapper for integrand_trispectrum for cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (here: 6, automatically assigned by integration)
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param thisPtr Pointer to TrispecContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_Trispectrum(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

/**
 * @brief Integrand of Trispectrum, interface to GPU
 * @param vars Integration parameters (m)
 * @param ndim Number of dimensions of integral (here: 1)
 * @param npts Number of integration points
 * @param l ell-value[1/rad]
 * @param value value of integrand
 */
__global__ void integrand_Trispectrum_kernel(const double *vars, unsigned ndim, int npts, double l1, double l2, double l3, double l4, double *value);

/**
 * @brief 1-halo term of projected Trispectrum
 * 
 * @param l1 ell1-value [1/rad]
 * @param l2 ell1-value [1/rad]
 * @param l3 ell1-value [1/rad]
 * @param l4 ell1-value [1/rad]
 * 
 */
double Trispectrum(const double &l1, const double &l2, const double &l3, const double &l4);


/**
 * @brief 1-halo term of projected Pentaspectrum
 * 
 * @param l1 ell1-value [1/rad]
 * @param l2 ell1-value [1/rad]
 * @param l3 ell1-value [1/rad]
 * @param l4 ell1-value [1/rad]
 * @param l5 ell1-value [1/rad]
 * @param l6 ell1-value [1/rad]
 
 */
double Pentaspectrum(const double &l1, const double &l2, const double &l3, const double &l4, const double &l5, const double &l6);


/**
 * @brief Wrapper for integrand_pentaspectrum for cubature library
 * See https://github.com/stevengj/cubature for documentation
 * @param ndim Number of dimensions of integral (here: 8, automatically assigned by integration)
 * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
 * @param vars Array containing integration variables
 * @param thisPtr Pointer to PentaspecContainer instance
 * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
 * @param value Value of integral
 * @return 0 on success
 */
int integrand_Pentaspectrum(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

/**
 * @brief Integrand of Pentaspectrum, interface to GPU
 * @param vars Integration parameters (m)
 * @param ndim Number of dimensions of integral (here: 1)
 * @param npts Number of integration points
 * @param l ell-value[1/rad]
 * @param value value of integrand
 */
__global__ void integrand_Pentaspectrum_kernel(const double *vars, unsigned ndim, int npts, double l1, double l2, double l3, double l4, double l5, double l6, double *value);

/**
 * @brief Container for \sigma²(m) Integration
 * 
 */
struct SigmaContainer
{
  double R;
  double dR;
};

/**
 * @brief Container for powerspec Integration
 * 
 */

struct PowerspecContainer
{
  double l;
};

/**
 * @brief Container for Trispec Integration
 * 
 */
struct TrispecContainer
{
  double l1;
  double l2;
  double l3;
  double l4;
};

// struct TrispecContainer3D
// {
//   double k1;
//   double k2;
//   double k3;
//   double k4;
//   double z;
// };

/**
 * @brief Container for Pentaspecspec Integration
 * 
 */
struct PentaspecContainer
{
  double l1;
  double l2;
  double l3;
  double l4;
  double l5;
  double l6;
};




__device__ double integrand_I_31(const double& k1, const double& k2, const double& k3, const double& m, const double& z);

__device__ double I_31(const double& k1, const double& k2, const double& k3, const double& a, const double& b, const double& z);


#endif // HALOMODEL_CUH