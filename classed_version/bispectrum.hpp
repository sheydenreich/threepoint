#ifndef BISPECTRUMHEADERDEF
#define BISPECTRUMHEADERDEF

#include <iostream>
#include <cmath>
#include <cassert>
#include <memory>
#include <functional>
#include <gsl/gsl_integration.h>
#include <vector>
#include <map>

// use slics cosmology (false -> millennium cosmology)
// TODO: Implenment redshift distribution for Euclid-like SLICS
#define slics true //false
#define test_analytical false

#define CONSTANT_POWERSPECTRUM false
#define ANALYTICAL_POWERSPECTRUM false
#define ANALYTICAL_POWERSPECTRUM_V2 false

#define TREAT_DEGENERATE_TRIANGLES true

#if ANALYTICAL_POWERSPECTRUM
// Powerspectrum: P(l)=p1*l*l*exp(-p2*l*l)
const double p1 = 1e-8;
const double p2 = 1e-8;
#elif ANALYTICAL_POWERSPECTRUM_V2
// Powerspectrum: P(l)=p1*l*exp(-p2*l)
const double p1 = 1e-4;
const double p2 = 1e-4;
#endif

/**
   * @brief one permutation of Eq.(21) in Joachimi et al. (2009)
   * @param ell1 first sidelength of triangle
   * @warning this function will probably be mitigated to a utility-module
   * @return 1 if ell_{1+i}=ell_{4+i}, else 0
   */
double delta_distrib(double ell1, double ell2, double ell3, double ell4, double ell5, double ell6);

/**
   * @brief D defined in in Eq.(21) in Joachimi et al. (2009)
   * @param ell1 first sidelength of triangle
   * @warning this function will probably be mitigated to a utility-module
   * @return sum of all delta distributions with permutations
   */
double delta_distrib_permutations(double ell1, double ell2, double ell3, double ell4, double ell5, double ell6);

/**
   * @brief returns a number proportional to the number of triangles of a certain configuration
   * Eq.(8) in Joachimi et al. (2009)
   * @param ell1 first sidelength of triangle
   * @param ell2 second sidelength of triangle
   * @param ell3 third sidelength of triangle
   * @warning this function will probably be mitigated to a utility-module
   * @return Value of function
   */

double number_of_triangles(double ell1, double ell2, double ell3);

/**
   * @brief checks if the three input ell can form a triangle
   * @param ell1 first sidelength of triangle
   * @param ell2 second sidelength of triangle
   * @param ell3 third sidelength of triangle
   * @warning this function will probably be mitigated to a utility-module
   */
bool is_triangle(double ell1, double ell2, double ell3);

/**
 *  @brief This structure contains parameters of a wCDM cosmology (should be in a utility.h)
 */
struct cosmology
{
  double h;      /**< dimensionless Hubble constant*/
  double sigma8; /**< Powerspectrum normalisation \f$\sigma_8\f$*/
  double omb;    /**< dimensionless baryon density parameter \f$\Omega_b\f$*/
  double omc;    /**< dimensionless density parameter of CDM*/
  double ns;     /**< Power spectrum spectral index*/
  double w;      /**< Eq. of state of Dark Energy*/
  double om;     /**< dimensionless matter density parameter*/
  double ow;     /**< dimensionless density parameter of Dark Energy*/
};

/**
 * @brief This structure contains the arguments of the projected bispectrum
 */
struct ell_params
{
  double ell1, ell2, ell3; /**< l-modes, in h/Mpc*/
};

/**
 * @brief Needed for GSL integrations
 */
template <typename F>
class gsl_function_pp : public gsl_function
{
public:
  gsl_function_pp(const F &func) : _func(func)
  {
    function = &gsl_function_pp::invoke;
    params = this;
  }

private:
  const F &_func;
  static double invoke(double x, void *params)
  {
    return static_cast<gsl_function_pp *>(params)->_func(x);
  }
};

/**
 * Class computing Bispectrums.  This class takes a given cosmology as an 
 * input and can compute the  matter- and convergence-bispectrum for a given 
 * redshift distribution. Based on Takahashi et al. (2019)
 * @warning Assumes flat cosmology
 */
class BispectrumCalculator
{
public:
  cosmology * cosmo;
//  double h, sigma8, omb, omc, ns, w, om, ow; /**< Cosmology parameters*/
  double norm;                                                           //Normalization

private:
  double eps = 1.e-4; /**< Integration precision*/
  int n_redshift_bins; /**< Number of redshift bins*/

  bool n_of_z_fromFile;            //Whether or not n(z) from file is given
  std::vector<double> n_of_z_file; //Values for n_of_z from file, is allocated and filled in constructor, if n_of_z_fromFile=True

  std::vector<double> f_K_array;       // Array of f_k for interpolating, allocated in initialize()
  std::vector<double> g_array;         //Array of lens efficiency g for interpolating, allocated in initialize()
  //std::vector<double> n_z_array_z;     // Array for n(z) for interpolating, allocated in initialize()
 // int len_n_z_array = 100; // Length of n(z) array

  bool Pk_given;                           // Whether or not linear matter power spectrum is given by constructor
  std::map<double, double> linearPk_given; // Given linear matter powerspectrum, key are wavevectors in h/Mpc, values are P(k) at z=0 in Mpc³/h³

  std::vector<double> D1_array;      /**< Array for Growthfactor on a grid, is initialized in initialize()*/
  std::vector<double> r_sigma_array; /**< Array for r_sigma on a grid, is initialized in initialize()*/
  std::vector<double> n_eff_array;   /**< Array for n_eff on a grid, is initialized in initialize()*/
  std::vector<double> ncur_array;

public:
  double z_max; /**< Maximal redshift to which D1, r_sigma and n_eff are calculated*/
private:
  double dz; /**< Binsize in redshift*/

  // These functions are required for the limber-integration (Shouldn't these be const in a utility.h?)
  double H0_over_c = 100. / 299792.; /**< Hubble constant divided by speed of light [h/Mpc]*/
  double c_over_H0 = 2997.92;        /**< speed of light divided by Hubble constant [Mpc/h] */

  gsl_integration_workspace *w_bkappa = gsl_integration_workspace_alloc(1000); /**< Workspace to integrate bispec to bkappa*/

  bool initialized = false; /**< Switch to decide if initalize was already run*/
  bool fast_calculations;   // Switch to decide if calculations should be sped up (Doesn't do anything right now!!!)

  // These are the necessary integration routines. Avoids overhead, since otherwise the function needs to be casted to a static function every time before use.
  double A96[48] = {
      0.016276744849603, 0.048812985136050, 0.081297495464426, 0.113695850110666,
      0.145973714654897, 0.178096882367619, 0.210031310460567, 0.241743156163840,
      0.273198812591049, 0.304364944354496, 0.335208522892625, 0.365696861472314,
      0.395797649828909, 0.425478988407301, 0.454709422167743, 0.483457973920596,
      0.511694177154668, 0.539388108324358, 0.566510418561397, 0.593032364777572,
      0.618925840125469, 0.644163403784967, 0.668718310043916, 0.692564536642172,
      0.715676812348968, 0.738030643744400, 0.759602341176648, 0.780369043867433,
      0.800308744139141, 0.819400310737932, 0.837623511228187, 0.854959033434602,
      0.871388505909297, 0.886894517402421, 0.901460635315852, 0.915071423120898,
      0.927712456722309, 0.939370339752755, 0.950032717784438, 0.959688291448743,
      0.968326828463264, 0.975939174585137, 0.982517263563015, 0.988054126329624,
      0.992543900323763, 0.995981842987209, 0.998364375863182, 0.999689503883231}; /**<abscissas for 96-point Gauss quadrature */

  double W96[48] = {
      0.032550614492363, 0.032516118713869, 0.032447163714064, 0.032343822568576,
      0.032206204794030, 0.032034456231993, 0.031828758894411, 0.031589330770727,
      0.031316425596861, 0.031010332586314, 0.030671376123669, 0.030299915420828,
      0.029896344136328, 0.029461089958168, 0.028994614150555, 0.028497411065085,
      0.027970007616848, 0.027412962726029, 0.026826866725592, 0.026212340735672,
      0.025570036005349, 0.024900633222484, 0.024204841792365, 0.023483399085926,
      0.022737069658329, 0.021966644438744, 0.021172939892191, 0.020356797154333,
      0.019519081140145, 0.018660679627411, 0.017782502316045, 0.016885479864245,
      0.015970562902562, 0.015038721026995, 0.014090941772315, 0.013128229566962,
      0.012151604671088, 0.011162102099839, 0.010160770535008, 0.009148671230783,
      0.008126876925699, 0.007096470791154, 0.006058545504236, 0.005014202742928,
      0.003964554338445, 0.002910731817935, 0.001853960788947, 0.000796792065552}; /**< weights for 96-point Gauss quadrature*/

public:
  /**
   * Empty constructor
   */
  BispectrumCalculator();

  /**
   * Constructor. Runs initialize and set_cosmology
   * @param cosmo cosmology that is to be used
   * @param n_z number of redshift bins for grids
   * @param z_max maximal redshift for grids
   * @param fast_calculations switch deciding if calculations should be sped up (doesn't do anything right now)
   */
  BispectrumCalculator(cosmology* cosmo, int n_z, double z_max, bool fast_calculations);

  /**
   * Constructor from n(z). Runs initialize and set_cosmology
   * @param cosmo cosmology that is to be used
   * @param n_of_z Array containing n(z) from file
   * @param n_z number of redshift bins for grids
   * @param z_max maximal redshift for grids
   */
  BispectrumCalculator(cosmology* cosmo, const std::vector<double> &n_of_z, int n_z, double z_max);

  /**
 * @brief Constructor from n(z) and linear P(k). Runs initialize and set_cosmology
 * 
 * @param cosmo cosmology that is to be used
 * @param n_of_z Array containing n(z) from file
 * @param n_z number of redshift bins for grids
 * @param z_max maximal redshift for grids
 * @param P_k linear matter power spectrum, keys are wavevectors k [h/Mpc], values are P(k) [Mpc^3/h^3]
 */
  BispectrumCalculator(cosmology* cosmo, const std::vector<double> &n_of_z, int n_z, double z_max,
                       const std::map<double, double> &P_k);

  /**
   * Initializes class. Allocates memory for f_k, g, D1, r_sigma and n_eff grids and sets cosmology
   * @param cosmo cosmology that is to be used
   * @param n_z number of redshift bins for grids
   * @param z_max maximal redshift for grid
   * @param fast_calculations switch deciding if calculations should be sped up (doesn't do anything right now)
   */
  void initialize(cosmology* cosmo, int n_z, double z_max, bool fast_calculations);

  /**
   * Set cosmology and initialize Bispectrum.
   * Assigns cosmology to internal variables, calculates f_k, g, D1, r_sigma and n_eff on grid
   * @param cosmo cosmology that is to be used
   */
  void set_cosmology(cosmology* cosmo);

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
  double bispec(double k1, double k2, double k3, double z, int idx, double didx);

  /**
   * 2D Bispectrum B_kappa, integrated from bispec 
   * @param ell1 l-mode 1
   * @param ell2 l_mode 2
   * @param ell3 l-mode 3
   */
  double bkappa(double ell1, double ell2, double ell3);


  /**
   * Integrand of two-dimensional bispectrum in Limber equation
   * @param z redshift
   * @param p ell- values of bispectrum
   */
  double integrand_bkappa(double z, ell_params p);

  /**
   * Non-linear Powerspectrum according to Takahashi et al. halofit model
   * @param k Mode [h/Mpc]
   * @param z redshift
   * @return Power spectrum at k [(Mpc/h)^3]
   */

  double P_k_nonlinear(double k, double z);

  /**
   * Linear Powerspectrum scaled with the growth factor
   * @param k Mode [h/Mpc]
   * @param z redshift
   * @return Power spectrum at k [(Mpc/h)^3]
   */

  double linear_pk_at_z(double k, double z);

  /**
   * limber-integrated power spectrum
   * @param ell desired ell value of P_kappa
   */
  double convergence_power_spectrum(double ell);

  /**
   * Wrapper for limber integration of power spectrum via cubature library
   */
  static int limber_integrand_power_spectrum(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

  /**
   * Integrand of limber equation for the power spectrum
   * @param ell desired ell value of P_kappa
   * @param z redshift
   */
  double limber_integrand_power_spectrum(double ell, double z, bool nonlinear);

  /**
   * Integrand of limber equation for a product of three power spectra
   * @param ell1 desired ell value of the first P_kappa
   * @param ell2 desired ell value of the second P_kappa
   * @param ell3 desired ell value of the third P_kappa
   * @param z redshift
   */
  double limber_integrand_triple_power_spectrum(double ell1, double ell2, double ell3, double z);

  /**
 * @brief Gaussian covariance of Bispectrum according to Joachimi+(2009)
 * 
 * @param ell1 l-mode 1 [1/rad]
 * @param ell2 l-mode 2 [1/rad]
 * @param ell3 l-mode 3 [1/rad]
 * @param ell4 l-mode 4 [1/rad]
 * @param ell5 l-mode 5 [1/rad]
 * @param ell6 l-mode 6 [1/rad]
 * @param delta_ell1 ell-binsize [1/rad]
 * @param delta_ell2 ell-binsize [1/rad]
 * @param delta_ell3 ell-binsize [1/rad]
 * @param delta_ell4 ell-binsize [1/rad]
 * @param delta_ell5 ell-binsize [1/rad]
 * @param delta_ell6 ell-binsize [1/rad]
 * @param survey_area Survey Area [rad²]
 * @return double Covariance
 */
  double bispectrumCovariance(double ell1, double ell2, double ell3,
                              double ell4, double ell5, double ell6,
                              double delta_ell1, double delta_ell2, double delta_ell3,
                              double delta_ell4, double delta_ell5, double delta_ell6,
                              double survey_area);

private:
  /**
   * Prefactor for the limber integration of the Power spectrum
   * @param z Redshift
   * @param g_value value of the lens efficiency function g(z)
   */
  double limber_integrand_prefactor(double z, double g_value);

  /**
   * Gives tree-level bispectrum
   * Uses   \f$B(\vec{k}_1, \vec{k}_2, \vec{k}_3)= 2 D(z)^4 (F(k_1, k_2, k_3) P(k_1) P(k_2) + F(k_2, k_3, k_1) P(k_2) P(k_3) + F(k_3, k_1, k_2) P(k_3) P(k_1))\f$ 
   * @param k1 Absolute value of first mode [h/Mpc]
   * @param k2 Absolute value of second mode [h/Mpc]
   * @param k3 Absolute value of third mode [h/Mpc]
   * @param D1 Growth factor at current redshift [unitless]
   * @return Tree-level bispectrum at k1, k2, k3 and redshift corresponding to D1 [(Mpc/h)^6]
   */
  double bispec_tree(double k1, double k2, double k3, double D1);

  /**
   * ??
   */
  double F2(double k1, double k2, double k3, double z, double D1, double r_sigma);

  /**
   * Mode coupling function for tree-level Bispectrum.
   *  \f$F_2(k_1, k_2, k_3) =  \frac{5}{7} +  \frac{2}{7} \frac{(\vec{k}_1\cdot\vec{k}_2)^2}{k_1^2k_2^2} + \frac{1}{2}\frac{\vec{k}_1\cdot\vec{k}_2}{k_1k_2}(k_1/k_2 + k_2/k_1)\f$
      * @param k1 Absolute value of first mode [h/Mpc]
   * @param k2 Absolute value of second mode [h/Mpc]
   * @param k3 Absolute value of third mode [h/Mpc]
   * @param D1 Growth factor at current redshift [unitless]
   * @return Mode coupling function [unitless]
   */
  double F2_tree(double k1, double k2, double k3);

  /**
   * Ratio of Bispectrum with baryons to Bispectrum without baryons
   * @param k1 Absolute value of first mode [h/Mpc]
   * @param k2 Absolute value of second mode [h/Mpc]
   * @param k3 Absolute value of third mode [h/Mpc]
   * @param z redshift [unitless]
   * @return Ratio of bispectrums [unitless]
   */
  double baryon_ratio(double k1, double k2, double k3, double z);

  double calc_r_sigma(double D1);

  double sigmam(double r, int j);

  /**
   * Windowfunction.
   * Gives either Fourier transformed Top Hat, Gaussian or 1st Derivative of Gaussian Window function
   * @param x dimensionless position in windowfunction
   * @param i Switch between different functions, i=0: FT Top hat, i=1: Gaussian, i=2: 1st derivative of Gaussian
   * @return Windowfunction at x [dimensionless]
   */
  double window(double x, int i);

  /**
   * Linear Powerspectrum. Either uses the powerspectrum given in constructor (with linear interpolation) or Eisenstein & Hu (without wiggle)
   * @param k Mode [h/Mpc]
   * @return Power spectrum at k [(Mpc/h)^3]
   */
  double linear_pk(double k);

  /**
   * Eisenstein & Hu Powerspectrum (without wiggle). Based on Eisenstein & Hu (1998).
  * @param k Mode [h/Mpc]
   * @return Power spectrum at k [(Mpc/h)^3]
   */
  double linear_pk_eh(double k);

  /**
   * Linear growth function.
   * @warning Not normalized at z=0!
   * @param z redshift
   */
  double lgr(double z);

  double lgr_func(int j, double x, double y[2]);

  /**
   * Computes coefficients by interpolating on a grid. Grid was computed during initialization
   * @param idx index of redshift bin (lower border), int(z/z_max*(nbins-1))
   * @param didx distance between redshift and redshift bin, (z/z_max*(nbins-1))-idx
   * @param D1 will store Growth factor at redshift corresponding to idx+didx
   * @param r_sigma will store r_sigma at redshift corresponding to idx+didx
   * @param n_eff will store n_eff at redshift corresponding to idx+didx
   */
  void compute_coefficients(int idx, double didx, double *D1, double *r_sigma, double *n_eff);

  /**
   * Source redshift distribution.
   * Currently includes only SLICS distribution, or all sources at z=1.
   * Uses slics distribution, if slics=True
   * @param z redshift
   * @return n(z), normalized sich that  \f$\int_0^{z_{max}} dz n(z) = 1\f$ 
   */
  double n_of_z(double z);

  /**
   * Comoving angular diameter distance  \f$f_k\f$ 
   * Uses Gaussian quadrature of E_inv
   * @param z redshift
   * @return  \f$f_k\f$  [h/Mpc]
   */
  double f_K_at_z(double z);

  /**
   * Expansion function, for flat Universe
   *  \f$ E(z) = \sqrt{\Omega_m (1+z)^3 + \Omega_\Lambda}\f$ 
   * @param z redshift
   */
  double E(double z);

  /**
   * Inverse of Expansion function, for flat Universe
   * @param z redshift
   */
  double E_inv(double z);

  /**
   * Gives the lens efficiency g at redshift corresponding to idx+didx.
   * Uses interpolation of grid
   * @param idx index of redshift bin (lower border), int(z/z_max*(nbins-1))
   * @param didx distance between redshift and redshift bin, (z/z_max*(nbins-1))-idx
   */
  double g_interpolated(int idx, double didx);

  /**
   * Gives f_k at redshift corresponding to idx+didx.
   * Uses interpolation of grid
   * @param idx index of redshift bin (lower border), int(z/z_max*(nbins-1))
   * @param didx distance between redshift and redshift bin, (z/z_max*(nbins-1))-idx
   * @return f_K [h/Mpc]
   */
  double f_K_interpolated(int idx, double didx);

  // For testing the gamma-integration
  double bispectrum_analytic_single_a(double l1, double l2, double phi, double a);

  /**
   * 96 pt Gaussian Quadrature of inverse expansion function.
   * Defined here, to avoid overhead, since otherwise the function needs to be casted to a static function every time before use
   * @param a lower border of redshift integral
   * @param b upper border of redshift integral
   * @return  \f$\int_a^b dz \frac{1}{E(z)}\f$
   */
  double GQ96_of_Einv(double a, double b);

  /**
   * 96 pt Gaussian Quadrature of integrand_bkappa along redshift
   * Defined here, to avoid overhead, since otherwise the function needs to be casted to a static function every time before use
   * @param a lower border of redshift integral
   * @param b upper border of redshift integral
   */
  double GQ96_of_bdelta(double a, double b, ell_params ells);

  /**
   * Matter density parameter at redshift z
   * @param z redshift
   */
  double om_m_of_z(double z);

  /**
   * Dark Energy density parameter at redshift z
   * @param z redshift
   */
  double om_v_of_z(double z);


  double GQ96_of_Pk(double a, double b, double ell);

public:
  double Pell(double ell);
};

struct BispectrumContainer
{
  /** Bispectrum to be calculated*/
  BispectrumCalculator *bispectrum;

  /** l-mode [1/rad]*/
  double ell;
};

#endif
