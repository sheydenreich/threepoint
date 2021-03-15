#ifndef GAMMAHEADERDEF
#define GAMMAHEADERDEF

#include "bispectrum.hpp"
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_math.h>
#include <complex>
#include <boost/math/quadrature/trapezoidal.hpp>
#include "../cubature/cubature.h"
// #include "Levin.h"

const std::complex<double> i_complex(0,1);

class GammaCalculator
{
private:
  int initialize_bessel(double prec_h_arg, double prec_k_arg);
  double w_function(unsigned int k, double *bessel_zeros);
  double psi(double t);
  double psip(double t);
  double angle(double x,double y);
  double bispectrum(double l1, double l2, double phi);
  double alpha(double psi, double x1, double x2, double phi, double varpsi);
  double betabar(double psi, double phi);
  double integrated_bispec(double psi, double phi, double A3);
  // std::complex<double> integrand_phi(double psi, double x1, double x2, double x3, double phi);
  // std::complex<double> integrand_psi(double psi, double x1, double x2, double x3);
  std::complex<double> integrand_phi_psi(double phi, double psi, double x1, double x2, double x3);
  std::complex<double> exponential(double x1, double x2, double x3, double psi, double phi, double varpsi);
  std::complex<double> prefactor_phi(double psi, double phi);
  
  // try: integrating bkappa by gaussian quadrature
  double GQ96_bkappa(double psi, double phi, double A3);
  double bkappa_rcubed_j6(double r, double psi, double phi, double A3);
  // try: integration via levin's method
  // Levin lp;

  // integration precision for bessel integral
  double prec_h;
  int prec_k;
  // this avoids integration of degenerate triangles
  double integral_border = 1.0e-14;
  double* bessel_zeros;
  double* pi_bessel_zeros;
  double* array_psi;
  double* array_bessel;
  double* array_psip;
  double* array_w;
  double* array_product;
  BispectrumCalculator Bispectrum_Class;

  bool fast_calculations;

  gsl_integration_workspace * w_phi = gsl_integration_workspace_alloc(1000);
  gsl_integration_workspace * w_psi = gsl_integration_workspace_alloc(1000);


  // These are the necessary integration routines. Avoids overhead, since otherwise the function needs to be casted to a static function every time before use.
  double A96[96]={                   /* abscissas for 96-point Gauss quadrature */
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

  double W96[96]={                     /* weights for 96-point Gauss quadrature */
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

  std::complex<double> GQ962D_phi_psi(double x1,double x2, double x3);



public:
  std::complex<double> gamma0(double x1, double x2, double x3); //x1,x2,x3 in rad
  // For testing: analytical ggg correlation
  std::complex<double> ggg(std::complex<double> x, std::complex<double> y); // FOR TESTING
  std::complex<double> ggg_single_a(std::complex<double> x, std::complex<double> y, double a);
  std::complex<double> integrand_phi_psi(double phi, double psi, double x1, double x2, double x3);
  double A(double psi, double x1, double x2, double phi, double varpsi);
  double varpsifunc(double an1, double an2, double opp);

  GammaCalculator(cosmology cosmo, double prec_h, double prec_k, bool fast_calculations_arg, int n_z, double z_max);


  std::complex<double> Trapz2D_phi_psi(double x1,double x2, double x3);
  std::complex<double> integrand_psi(double psi, double x1, double x2, double x3);
  std::complex<double> integrand_x_psi(double x, double psi, double x1, double x2, double x3);
  double r_integral(double phi, double psi, double x1, double x2, double x3);
  std::complex<double> integrand_r_phi_psi_one_x(double r, double phi, double psi, double x1, double x2, double x3);
  double integrand_imag(double r, double phi, double psi, double z, double x1, double x2, double x3);
  double integrand_real(double r, double phi, double psi, double z, double x1, double x2, double x3);

  static int integrand_imag(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value);
  static int integrand_real(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value);
  std::complex<double> gamma0_from_cubature(double x1, double x2, double x3);


};

struct integration_parameter
{
    
  double x1,x2,x3;
  GammaCalculator* gammaCalculator;
};


#endif
