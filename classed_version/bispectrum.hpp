#ifndef BISPECTRUMHEADERDEF
#define BISPECTRUMHEADERDEF

#include <iostream>
#include <cmath>
#include <cassert>
#include <memory>
#include <functional>
#include <gsl/gsl_integration.h>

// use slics cosmology (false -> millennium cosmology)
// TODO: Implenment redshift distribution for Euclid-like SLICS
#define slics true
#define test_analytical false

struct cosmology
{
    double h,sigma8,omb,omc,ns,w,om,ow;
};

struct ell_params
{
    double ell1,ell2,ell3;
};

template< typename F >
  class gsl_function_pp : public gsl_function {
  public:
  gsl_function_pp(const F& func) : _func(func) {
    function = &gsl_function_pp::invoke;
    params=this;
  }
  private:
  const F& _func;
  static double invoke(double x, void *params) {
    return static_cast<gsl_function_pp*>(params)->_func(x);
  }
};

class BispectrumCalculator
{
    /*
    This class takes a given cosmology as an input and can compute the
    matter- and convergence-bispectrum for a given redshift distribution
    */
private:
// These functions are necessary to compute the delta-Bispectrum via Takahashi et al.
    double h,sigma8,omb,omc,ns,w,om,ow,norm;
    double eps=1.e-4;
    double bispec_tree(double k1, double k2, double k3, double z, double D1);
    double F2(double k1, double k2, double k3, double z, double D1, double r_sigma);
    double F2_tree(double k1, double k2, double k3);
    double baryon_ratio(double k1, double k2, double k3, double z);
    double calc_r_sigma(double z, double D1);
    double sigmam(double r, int j); 
    double window(double x, int i);
    double linear_pk(double k); 
    double linear_pk_data(double k); 
    double linear_pk_eh(double k);   
    double lgr(double z);
    double lgr_func(int j, double x, double y[2]);
    int compute_coefficients(int idx, double didx, double *D1, double *r_sigma, double *n_eff);
    double* D1_array;
    double* r_sigma_array;
    double* n_eff_array;
    int n_redshift_bins;
    double z_max,dz;
// These functions are required for the limber-integration
    double H0_over_c = 100./299792.;
    double c_over_H0 = 2997.92;
    bool fast_calculations;
    double n_of_z(double z);
    double f_K_at_z(double z);
    double E(double z);
    double E_inv(double z);
    double g_interpolated(int idx,double didx);
    double f_K_interpolated(int idx,double didx);
    void read_nofz(char filename[255]);
    void normalize_nofz();
    double* f_K_array;
    double* g_array;
    double* n_z_array_data;
    double* n_z_array_z;
    int len_n_z_array = 100;


// For testing the gamma-integration
    double bispectrum_analytic_single_a(double l1, double l2, double phi, double a);


// This is the workspace to integrate bispec to bkappa
    gsl_integration_workspace * w_bkappa = gsl_integration_workspace_alloc(1000);

// This ensures that memory gets allocated
    bool initialized = false;

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
    double GQ96_of_Einv(double a,double b);
    double GQ96_of_bdelta(double a,double b,ell_params ells);


public:
    void set_cosmology(cosmology cosmo);
    void initialize(cosmology cosmo, int n_z, double z_max, bool fast_calculations);
    double integrand_bkappa(double z, ell_params p);

    double bispec(double k1, double k2, double k3, double z, int idx, double didx);
    double bkappa(double ell1,double ell2, double ell3);
    BispectrumCalculator(cosmology cosmo, int n_z, double z_max, bool fast_calculations);
// These functions are for optimized calculations (omitting prefactors)
    double bkappa_without_prefactor(double ell1,double ell2, double ell3);
    BispectrumCalculator();
};

#endif