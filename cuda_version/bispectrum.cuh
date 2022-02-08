#ifndef BISPECTRUM_CUH
#define BISPECTRUM_CUH

#define slics true
//#define CONSTANT_POWERSPECTRUM false

#include "cosmology.cuh"

#include <vector>

/**
 * @file bispectrum.cuh
 * This file declares routines needed for the bispectrum calculation
 * Routines are defined in bispectrum.cu
 * These functions mirror the methods of class bispectrumCalculator in the pure c++ version
 * @author Laila Linke
 */


// Declarations of  constant variables
// Extern keyword is needed, so that the actual definition can happen in bispectrum.cu!
extern bool constant_powerspectrum;
extern __constant__ bool dev_constant_powerspectrum;

// Cosmological Parameters
extern __constant__ double dev_h,dev_sigma8,dev_omb,dev_omc,dev_ns,dev_w,dev_om,dev_ow,dev_norm;
extern cosmology cosmo;
extern double norm_P;

extern __constant__ double dev_sigma, dev_n;
extern double sigma, n;

extern __constant__ double dev_eps; //< Integration accuracy
extern const double eps;

extern __constant__ int dev_n_redshift_bins; // Number of redshift bins
const int n_redshift_bins = 256;
extern __constant__ int dev_n_kbins;
const int n_kbins = 256;


extern __constant__ double dev_f_K_array[n_redshift_bins]; // Array for comoving distance
extern __constant__ double dev_g_array[n_redshift_bins]; // Array for lensing efficacy g

extern __constant__ bool dev_Pk_given;
extern __constant__ double dev_Pk[n_kbins];
extern double Pk[n_kbins];
extern bool Pk_given;

extern __constant__ double dev_D1_array[n_redshift_bins]; // Array for growth factor
extern __constant__ double dev_r_sigma_array[n_redshift_bins]; // Array for r(sigma)
extern __constant__ double dev_n_eff_array[n_redshift_bins]; // Array for n_eff
extern __constant__ double dev_ncur_array[n_redshift_bins]; //Array for C in Halofit

const double H0_over_c= 100./299792.;
extern __constant__ double dev_H0_over_c ; //Hubble constant/speed of light [h s/m]
const double c_over_H0 = 2997.92 ;
extern __constant__ double dev_c_over_H0; //Speed of light / Hubble constant [h^-1 m/s]

extern double A96[48];
extern double W96[48];

extern __constant__ double dev_A96[48];// Abscissas for Gauss-Quadrature
extern __constant__ double dev_W96[48];// Weights for Gauss-Quadrature


extern __constant__ double dev_dz,dev_z_max; //Redshift bin and maximal redshift
extern double dz, z_max;

extern __constant__ double dev_dk,dev_k_min,dev_k_max; //k bin and maximal k
extern double dk, k_min, k_max;

void copyConstants();

/**
* Set cosmology and initialize Bispectrum.
* Assigns cosmology to internal variables, calculates f_k, g, D1, r_sigma and n_eff on grid
* All values are copied to the devices constant memory
* @param cosmo cosmology that is to be used
* @param dz_ redshiftbinsize
* @param nz_from_file If true: Uses lookup table for n(z), if false: uses analytical formula (Optional, default: False)
* @param nz Vector containing values of n(z) for the redshiftbins used for all functions (Optional, but needs to be provided if nz_from_file==True)
*/
void set_cosmology(cosmology cosmo, std::vector<double>* nz=NULL, std::vector<double>* P_k=NULL, double dk = 0, double kmin=0, 
   double kmax=1e4);

  /**
   * Non-linear Bispectrum w/o baryons according to Takahashi et al (2019)
   * @param k1 Absolute value of first mode [h/Mpc]
   * @param k2 Absolute value of second mode [h/Mpc]
   * @param k3 Absolute value of third mode [h/Mpc]
   * @param z Redshift
   * @param idx index of redshift bin (lower border), int(z/z_max*(nbins-1))
   * @param didx distance between redshift and redshift bin, (z/z_max*(nbins-1))-idx
   * @return Bispectrum at (k1, k2, k3, z) [(Mpc/h)^6]
   */
   __device__ double bispec(double k1, double k2, double k3, double z, int idx, double didx);

   
/**
 * 2D Bispectrum B_kappa, integrated from bispec 
 * @param ell1 l-mode 1
 * @param ell2 l_mode 2
 * @param ell3 l-mode 3
 */
__device__ double bkappa(double ell1, double ell2, double ell3);

  /**
   * 96 pt Gaussian Quadrature of integrand_bkappa along redshift
   * Defined here, to avoid overhead, since otherwise the function needs to be casted to a static function every time before use
   * @param a lower border of redshift integral
   * @param b upper border of redshift integral
   * @param ell1 lmode 1
   * @param ell2 lmode 2
   * @param ell3 lmode 3
   */
   __device__ double GQ96_of_bdelta(double a,double b, double ell1, double ell2, double ell3);

  /**
   * Integrand of two-dimensional bispectrum in Limber equation
   * @param z redshift
   * @param ell1 lmode 1
   * @param ell2 lmode 2
   * @param ell3 lmode 3
   */
   __device__ double integrand_bkappa(double z, double ell1, double ell2, double ell3);


  /**
   * Gives the lens efficiency g at redshift corresponding to idx+didx.
   * Uses interpolation of grid
   * @param idx index of redshift bin (lower border), int(z/z_max*(nbins-1))
   * @param didx distance between redshift and redshift bin, (z/z_max*(nbins-1))-idx
   */
   __device__ double g_interpolated(int idx, double didx);


   /**
   * Gives f_k at redshift corresponding to idx+didx.
   * Uses interpolation of grid
   * @param idx index of redshift bin (lower border), int(z/z_max*(nbins-1))
   * @param didx distance between redshift and redshift bin, (z/z_max*(nbins-1))-idx
   * @return f_K [h/Mpc]
   */
__device__ double f_K_interpolated(int idx, double didx);



  /**
   * Computes coefficients by interpolating on a grid. Grid was computed during initialization
   * @param idx index of redshift bin (lower border), int(z/z_max*(nbins-1))
   * @param didx distance between redshift and redshift bin, (z/z_max*(nbins-1))-idx
   * @param D1 will store Growth factor at redshift corresponding to idx+didx
   * @param r_sigma will store r_sigma at redshift corresponding to idx+didx
   * @param n_eff will store n_eff at redshift corresponding to idx+didx
   */
   __device__  void compute_coefficients(int idx, double didx, double *D1, double *r_sigma, double *n_eff);

   __device__ double om_m_of_z(double z);

   __device__ double om_v_of_z(double z);

   __device__ double limber_integrand(double ell, double z);

   __global__ void limber_integrand_wrapper(const double* vars, unsigned ndim, size_t npts, double ell, double* value);

   int limber_integrand_wrapper(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);

   double Pell(double ell);

   __device__ double limber_integrand_triple_power_spectrum(double ell1, double ell2, double ell3, double z, double shapenoise_contribution);

   double get_P_k_nonlinear(double* k, double* z, double* value, int npts);

   __global__ void global_get_P_k_nonlinear(double* k, double* z, double* values);

   /**
 * Non-linear power spectrum from the revised Halofit formula (Takahashi et al. 2012)
 * @param k scale mode [h/Mpc]
 * @param z redshift
 * @return non-linear power spectrum at k [Mpc^3/h^3]
 */
__device__ double P_k_nonlinear(double k, double z);


  /**
   * If dev_Pk_given==True, this gives the value of the precomputed powerspectrum at k (linearly interpolated between bins)
   * Else: gives Eisenstein & Hu Powerspectrum (without wiggle). Based on Eisenstein & Hu (1998).
   * @param k Mode [h/Mpc]
   * @return Power spectrum at k [(Mpc/h)^3]
   */
   __device__ double dev_linear_pk(double k);

   /**
   * If Pk_given==True, this gives the value of the precomputed powerspectrum at k (linearly interpolated between bins)
   * Else: gives Eisenstein & Hu Powerspectrum (without wiggle). Based on Eisenstein & Hu (1998).
   * @param k Mode [h/Mpc]
   * @return Power spectrum at k [(Mpc/h)^3]
   */
   double linear_pk(double k);

  /**
   * Gives tree-level bispectrum
   * Uses   \f$B(\vec{k}_1, \vec{k}_2, \vec{k}_3)= 2 D(z)^4 (F(k_1, k_2, k_3) P(k_1) P(k_2) + F(k_2, k_3, k_1) P(k_2) P(k_3) + F(k_3, k_1, k_2) P(k_3) P(k_1))\f$ 
   * @param k1 Absolute value of first mode [h/Mpc]
   * @param k2 Absolute value of second mode [h/Mpc]
   * @param k3 Absolute value of third mode [h/Mpc]
   * @param D1 Growth factor at current redshift [unitless]
   * @return Tree-level bispectrum at k1, k2, k3 and redshift corresponding to D1 [(Mpc/h)^6]
   */
   __device__ double bispec_tree(double k1, double k2, double k3, double z, double D1);  // tree-level BS [(Mpc/h)^6]

   __device__ double F2(double k1, double k2, double k3, double z, double D1, double r_sigma);

   /**
    * Mode coupling function for tree-level Bispectrum.
    *  \f$F_2(k_1, k_2, k_3) =  \frac{5}{7} +  \frac{2}{7} \frac{(\vec{k}_1\cdot\vec{k}_2)^2}{k_1^2k_2^2} + \frac{1}{2}\frac{\vec{k}_1\cdot\vec{k}_2}{k_1k_2}(k_1/k_2 + k_2/k_1)\f$
       * @param k1 Absolute value of first mode [h/Mpc]
    * @param k2 Absolute value of second mode [h/Mpc]
    * @param k3 Absolute value of third mode [h/Mpc]
    * @param D1 Growth factor at current redshift [unitless]
    * @return Mode coupling function [unitless]
    */
 __device__ double F2_tree(double k1, double k2, double k3);  // F2 kernel in tree level
 
 /**
 * Comoving angular diameter distance  \f$f_k\f$ 
 * Uses Gaussian quadrature of E_inv
 * @param z redshift
 * @return  \f$f_k\f$  [h/Mpc]
 */
double f_K_at_z(double z);



/**
 * Source redshift distribution.
 * Currently includes only SLICS distribution, or all sources at z=1.
 * Uses slics distribution, if slics=True
 * @param z redshift
 * @return n(z), normalized sich that  \f$\int_0^{z_{max}} dz n(z) = 1\f$ 
   */
double n_of_z(double z);

  /**
   * Linear growth function.
   * @warning Not normalized at z=0!
   * @param z redshift
   */
double lgr(double z);

double lgr_func(int j, double x, double y[2]);

double sigmam(double r, int j);

  /**
   * Windowfunction.
   * Gives either Fourier transformed Top Hat, Gaussian or 1st Derivative of Gaussian Window function
   * @param x dimensionless position in windowfunction
   * @param i Switch between different functions, i=0: FT Top hat, i=1: Gaussian, i=2: 1st derivative of Gaussian
   * @return Windowfunction at x [dimensionless]
   */
   double window(double x, int i);

double calc_r_sigma(double D1);


  /**
   * 96 pt Gaussian Quadrature of inverse expansion function.
   * Defined here, to avoid overhead, since otherwise the function needs to be casted to a static function every time before use
   * @param a lower border of redshift integral
   * @param b upper border of redshift integral
   * @return  \f$\int_a^b dz \frac{1}{E(z)}\f$
   */
double GQ96_of_Einv(double a,double b);





  /**
   * Expansion function, for flat Universe
   *  \f$ E(z) = \sqrt{\Omega_m (1+z)^3 + \Omega_\Lambda}\f$ 
   * @param z redshift
   */
double E(double z);

  /**
   * Expansion function, for flat Universe for device
   *  \f$ E(z) = \sqrt{\Omega_m (1+z)^3 + \Omega_\Lambda}\f$ 
   * @param z redshift
   */
__device__ double dev_E(double z);

   /**
   * Inverse of Expansion function, for flat Universe
   * @param z redshift
   */
    double E_inv(double z);

__device__ double dev_GQ96_of_Pk(double a, double b, double ell);


__device__ double dev_Pell(double ell);

__device__ double dev_limber_integrand_power_spectrum(double ell, double z);

__device__ double dev_limber_integrand_prefactor(double z, double g_value);
double limber_integrand_prefactor(double z, double g_value);

#endif //BISPECTRUM_CUH
