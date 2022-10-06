#include "halomodel.cuh"
#include "cuda_helpers.cuh"
#include "bispectrum.cuh"
#include "apertureStatisticsCovariance.cuh"
#include "cubature.h"

#include <iostream>
#include <math.h>

__constant__ double devLogMmin, devLogMmax;
int __constant__ dev_n_mbins;

__constant__ double dev_sigma2_array[n_mbins];
__constant__ double dev_dSigma2dm_array[n_mbins];
double sigma2_array[n_mbins];
double dSigma2dm_array[n_mbins];

void initHalomodel()
{
  copyConstants();
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(devLogMmin, &logMmin, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(devLogMmax, &logMmax, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_mbins, &n_mbins, sizeof(int)));
  setSigma2();
  std::cerr << "Finished calculating sigma²(m)" << std::endl;
  setdSigma2dm();
  std::cerr << "Finished calculating dSigma²dm" << std::endl;
  std::cerr << "Finished all initializations" << std::endl;
}

__host__ __device__ double hmf(const double &m, const double &z)
{
  double A = 0.322;
  double q = 0.707;
  double p = 0.3;

  // Get sigma^2(m, z)
  double sigma2 = get_sigma2(m, z);
  // printf("%f %f\n", m, sigma2);

  // Get dsigma^2/dm
  double dsigma2 = get_dSigma2dm(m, z);

  // Get critical density contrast and mean density
#ifdef __CUDA_ARCH__
  double deltac = 1.686 * (1 + 0.0123 * log10(dev_om_m_of_z(z)));
  double rho_mean = 2.7754e11 * dev_om;
#else
  double deltac = 1.686 * (1 + 0.0123 * log10(om_m_of_z(z)));
  double rho_mean = 2.7754e11 * cosmo.om;
#endif

  double nu = deltac * deltac / sigma2;

  double result = -rho_mean / m / sigma2 * dsigma2 * A * (1 + pow(q * nu, -p)) * sqrt(q * nu / 2 / M_PI) * exp(-0.5 * q * nu);

  /*
   * \f $n(m,z)=-\frac{\bar{\rho}}{m \sigma} \dv{\sigma}{m} A \sqrt{2q/pi} (1+(\frac{\sigma^2}{q\delta_c^2})^p) \frac{\delta_c}{\sigma} \exp(-\frac{q\delta_c^2}{2\sigma^2})$ \f
   */
  return result;
}

__host__ __device__ void SiCi(double x, double &si, double &ci)
{
  double x2 = x * x;
  double x4 = x2 * x2;
  double x6 = x2 * x4;
  double x8 = x4 * x4;
  double x10 = x8 * x2;
  double x12 = x6 * x6;
  double x14 = x12 * x2;

  if (x < 4)
  {

    double a = 1 - 4.54393409816329991e-2 * x2 + 1.15457225751016682e-3 * x4 - 1.41018536821330254e-5 * x6 + 9.43280809438713025e-8 * x8 - 3.53201978997168357e-10 * x10 + 7.08240282274875911e-13 * x12 - 6.05338212010422477e-16 * x14;

    double b = 1 + 1.01162145739225565e-2 * x2 + 4.99175116169755106e-5 * x4 + 1.55654986308745614e-7 * x6 + 3.28067571055789734e-10 * x8 + 4.5049097575386581e-13 * x10 + 3.21107051193712168e-16 * x12;

    si = x * a / b;

    double gamma = 0.5772156649;
    a = -0.25 + 7.51851524438898291e-3 * x2 - 1.27528342240267686e-4 * x4 + 1.05297363846239184e-6 * x6 - 4.68889508144848019e-9 * x8 + 1.06480802891189243e-11 * x10 - 9.93728488857585407e-15 * x12;

    b = 1 + 1.1592605689110735e-2 * x2 + 6.72126800814254432e-5 * x4 + 2.55533277086129636e-7 * x6 + 6.97071295760958946e-10 * x8 + 1.38536352772778619e-12 * x10 + 1.89106054713059759e-15 * x12 + 1.39759616731376855e-18 * x14;

    ci = gamma + std::log(x) + x2 * a / b;
  }
  else
  {
    double x16 = x8 * x8;
    double x18 = x16 * x2;
    double x20 = x10 * x10;
    double cos_x = cos(x);
    double sin_x = sin(x);

    double f = (1 + 7.44437068161936700618e2 / x2 + 1.96396372895146869801e5 / x4 + 2.37750310125431834034e7 / x6 + 1.43073403821274636888e9 / x8 + 4.33736238870432522765e10 / x10 + 6.40533830574022022911e11 / x12 + 4.20968180571076940208e12 / x14 + 1.00795182980368574617e13 / x16 + 4.94816688199951963482e12 / x18 - 4.94701168645415959931e11 / x20) / (1 + 7.46437068161927678031e2 / x2 + 1.97865247031583951450e5 / x4 + 2.41535670165126845144e7 / x6 + 1.47478952192985464958e9 / x8 + 4.58595115847765779830e10 / x10 + 7.08501308149515401563e11 / x12 + 5.06084464593475076774e12 / x14 + 1.43468549171581016479e13 / x16 + 1.11535493509914254097e13 / x18) / x;

    double g = (1 + 8.1359520115168615e2 / x2 + 2.35239181626478200e5 / x4 + 3.12557570795778731e7 / x6 + 2.06297595146763354e9 / x8 + 6.83052205423625007e10 / x10 + 1.09049528450362786e12 / x12 + 7.57664583257834349e12 / x14 + 1.81004487464664575e13 / x16 + 6.43291613143049485e12 / x18 - 1.36517137670871689e12 / x20) /
               (1 + 8.19595201151451564e2 / x2 + 2.40036752835578777e5 / x4 + 3.26026661647090822e7 / x6 + 2.23355543278099360e9 / x8 + 7.87465017341829930e10 / x10 + 1.39866710696414565e12 / x12 + 1.17164723371736605e13 / x14 + 4.01839087307656620e13 / x16 + 3.99653257887490811e13 / x18) / x2;

    si = 0.5 * M_PI - f * cos_x - g * sin_x;
    ci = f * sin_x - g * cos_x;
  };
  return;
}

__host__ __device__ double r_200(const double &m, const double &z)
{
  double rhobar = 2.7754e11; // critical density[Msun*h²/Mpc³]

#ifdef __CUDA_ARCH__
  rhobar *= dev_om;
#else
  rhobar *= cosmo.om;
#endif

  return pow(0.75 / M_PI * m / rhobar / 200, 1. / 3.);
}

__host__ __device__ double u_NFW(const double &k, const double &m, const double &z)
{
  // Get concentration
  double c = concentration(m, z);

  double arg1 = k * r_200(m, z) / c;
  double arg2 = arg1 * (1 + c);

  double si1, ci1, si2, ci2;
  SiCi(arg1, si1, ci1);
  SiCi(arg2, si2, ci2);

  double term1 = sin(arg1) * (si2 - si1);
  double term2 = cos(arg1) * (ci2 - ci1);
  double term3 = -sin(arg1 * c) / arg2;
  double F = std::log(1. + c) - c / (1. + c);

  double result = (term1 + term2 + term3) / F;
  return result;
}

__host__ __device__ double concentration(const double &m, const double &z)
{
  // Using Duffy+ 08 (second mass definition, all halos)
  // To Do: Bullock01 might be better!
  double A = 10.14;
  double Mpiv = 2e12; // Msun h⁻1
  double B = -0.081;
  double C = -1.01;
  double result = A * pow(m / Mpiv, B) * pow(1 + z, C);
  return result;
}

double sigma2(const double &m)
{
  double rhobar = 2.7754e11 * cosmo.om; // critical density[Msun*h²/Mpc³]

  double R = pow(0.75 / M_PI * m / rhobar, 1. / 3.); //[Mpc/h]

  SigmaContainer container;
  container.R = R;

  double k_min[1] = {0};
  double k_max[1] = {1e12};
  double result, error;

  hcubature_v(1, integrand_sigma2, &container, 1, k_min, k_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

  return result / 2 / M_PI / M_PI;
}

int integrand_sigma2(unsigned ndim, size_t npts, const double *k, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "integrand_sigma2: Wrong fdim" << std::endl;
    exit(1);
  };
  SigmaContainer *container = (SigmaContainer *)thisPtr;

  double R = container->R;

#pragma omp parallel for
  for (unsigned int i = 0; i < npts; i++)
  {
    double k_ = k[i * ndim];
    double W = window(k_ * R, 0);

    value[i] = k_ * k_ * W * W * linear_pk(k_);
  };

  return 0;
}

void setSigma2()
{
  double rhobar = 2.7754e11 * cosmo.om; // critical density[Msun*h²/Mpc³]

  double deltaM = (logMmax - logMmin) / n_mbins;

  for (int i = 0; i < n_mbins; i++)
  {

    double m = pow(10, logMmin + i * deltaM);
    double R = pow(0.75 / M_PI * m / rhobar, 1. / 3.);
    double sigma = sigma2(m);
    sigma2_array[i] = sigma;
  }
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma2_array, &sigma2_array, n_mbins * sizeof(double)));
  std::cerr << "Finished precalculating sigma2(m)" << std::endl;
}

double dSigma2dm(const double &m)
{
  double rhobar = 2.7754e11; // critical density[Msun*h²/Mpc³]

#ifdef __CUDA_ARCH__
  rhobar *= dev_om;
#else
  rhobar *= cosmo.om;
#endif

  double R = pow(0.75 / M_PI * m / rhobar, 1. / 3.);                      //[Mpc/h]
  double dR = pow(0.75 / M_PI / rhobar, 1. / 3.) * pow(m, -2. / 3.) / 3.; //[Mpc/h]

  SigmaContainer container;
  container.R = R;
  container.dR = dR;

  double k_min[1] = {0};
  double k_max[1] = {1e12};
  double result, error;

  hcubature_v(1, integrand_dSigma2dm, &container, 1, k_min, k_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

  return result / M_PI / M_PI;
}

int integrand_dSigma2dm(unsigned ndim, size_t npts, const double *k, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "integrand_dSigma2dm: Wrong fdim" << std::endl;
    exit(1);
  };
  SigmaContainer *container = (SigmaContainer *)thisPtr;

  double R = container->R;
  double dR = container->dR;

#pragma omp parallel for
  for (unsigned int i = 0; i < npts; i++)
  {
    double k_ = k[i * ndim];
    double W = window(k_ * R, 0);
    double Wprime = window(k_ * R, 4);

    value[i] = k_ * k_ * W * Wprime * dR * linear_pk(k_) * k_;
  };

  return 0;
}

void setdSigma2dm()
{

  double deltaM = (logMmax - logMmin) / n_mbins;

  for (int i = 0; i < n_mbins; i++)
  {

    double m = pow(10, logMmin + i * deltaM);

    dSigma2dm_array[i] = dSigma2dm(m);
  }
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_dSigma2dm_array, &dSigma2dm_array, n_mbins * sizeof(double)));
  std::cerr << "Finished precalculating dsigma2/dm" << std::endl;
}

__host__ __device__ double get_sigma2(const double &m, const double &z)
{
  double logM = log10(m);

#ifdef __CUDA_ARCH__
  double didx = (logM - devLogMmin) / (devLogMmax - devLogMmin) * (dev_n_mbins - 1);
  int idx = didx;
  didx = didx - idx;
  if (idx == dev_n_mbins - 1)
  {
    idx = dev_n_mbins - 2;
    didx = 1.;
  }
  double sigma = dev_sigma2_array[idx] * (1 - didx) + dev_sigma2_array[idx + 1] * didx;

  didx = z / dev_z_max * (dev_n_redshift_bins - 1);
  idx = didx;
  didx = didx - idx;
  if (idx == dev_n_redshift_bins - 1)
  {
    idx = dev_n_redshift_bins - 2;
    didx = 1.;
  }

  double D1 = dev_D1_array[idx] * (1 - didx) + dev_D1_array[idx + 1] * didx;
#else
  double didx = (logM - logMmin) / (logMmax - logMmin) * (n_mbins - 1);
  int idx = didx;
  didx = didx - idx;
  if (idx == n_mbins - 1)
  {
    idx = n_mbins - 2;
    didx = 1.;
  }
  double sigma = sigma2_array[idx] * (1 - didx) + sigma2_array[idx + 1] * didx;

  didx = z / z_max * (n_redshift_bins - 1);
  idx = didx;
  didx = didx - idx;
  if (idx == n_redshift_bins - 1)
  {
    idx = n_redshift_bins - 2;
    didx = 1.;
  }

  double D1 = D1_array[idx] * (1 - didx) + D1_array[idx + 1] * didx;
#endif

  sigma *= D1 * D1;

  return sigma;
}

__host__ __device__ double get_dSigma2dm(const double &m, const double &z)
{
  double logM = log10(m);

#ifdef __CUDA_ARCH__
  double didx = (logM - devLogMmin) / (devLogMmax - devLogMmin) * (dev_n_mbins - 1);
  int idx = didx;
  didx = didx - idx;
  if (idx == dev_n_mbins - 1)
  {
    idx = dev_n_mbins - 2;
    didx = 1.;
  }
  double dsigma = dev_dSigma2dm_array[idx] * (1 - didx) + dev_dSigma2dm_array[idx + 1] * didx;

  didx = z / dev_z_max * (dev_n_redshift_bins - 1);
  idx = didx;
  didx = didx - idx;
  if (idx == dev_n_redshift_bins - 1)
  {
    idx = dev_n_redshift_bins - 2;
    didx = 1.;
  }

  double D1 = dev_D1_array[idx] * (1 - didx) + dev_D1_array[idx + 1] * didx;
#else
  double didx = (logM - logMmin) / (logMmax - logMmin) * (n_mbins - 1);
  int idx = didx;
  didx = didx - idx;
  if (idx == n_mbins - 1)
  {
    idx = n_mbins - 2;
    didx = 1.;
  }
  double dsigma = dSigma2dm_array[idx] * (1 - didx) + dSigma2dm_array[idx + 1] * didx;

  didx = z / z_max * (n_redshift_bins - 1);
  idx = didx;
  didx = didx - idx;
  if (idx == n_redshift_bins - 1)
  {
    idx = n_redshift_bins - 2;
    didx = 1.;
  }

  double D1 = D1_array[idx] * (1 - didx) + D1_array[idx + 1] * didx;
#endif
  dsigma *= D1 * D1;
  return dsigma;
}


__device__ __host__ double halo_bias(const double& m, const double& z)
{
  double q=0.707;
  double p=0.3;

  double sig2=get_sigma2(m, z);
#ifdef __CUDA_ARCH__
  double om_z=dev_om_m_of_z(z);
#else
  double om_z=om_m_of_z(z);
#endif

  double delta_c=1.686*(1+0.0123*log10(om_z));
  double factor=1;
  // if( m> 1e14)
  // {
  //   factor=3.3*pow(m, -0.04);
  // }

  return( 1+1./delta_c*(q*delta_c*delta_c/sig2 - 1 + 2*p/(1+pow(q*delta_c*delta_c/sig2, p))))*factor;

}

__device__ double trispectrum_integrand(double m, double z, double l1, double l2, double l3, double l4)
{
  double didx = z / dev_z_max * (n_redshift_bins - 1);
  int idx = didx;
  didx = didx - idx;
  if (idx == n_redshift_bins - 1)
  {
    idx = n_redshift_bins - 2;
    didx = 1.;
  }
  double g = dev_g_array[idx] * (1 - didx) + dev_g_array[idx + 1] * didx;
  double chi = dev_f_K_array[idx] * (1 - didx) + dev_f_K_array[idx + 1] * didx;
  double rhobar = 2.7754e11; // critical density[Msun*h²/Mpc³]
  rhobar *= dev_om;

  double result = hmf(m, z) * g * g * g * g / chi / chi;
  result *= dev_c_over_H0 / dev_E(z);
  result *= u_NFW(l1 / chi, m, z) * u_NFW(l2 / chi, m, z) * u_NFW(l3 / chi, m, z) * u_NFW(l4 / chi, m, z);
  result *= m * m * m * m / rhobar / rhobar / rhobar / rhobar;
  result *= pow(1.5 * dev_om / dev_c_over_H0 / dev_c_over_H0, 4); // Prefactor in h^8
  result *= pow(1 + z, 4);
  return result;
}

__device__ double trispectrum_limber_integrated(double a, double b, double m, double l1, double l2, double l3, double l4)
{
  int i;
  double cx, dx, q;
  cx = (a + b) / 2;
  dx = (b - a) / 2;
  q = 0;
  for (i = 0; i < 48; i++)
    q += dev_W96[i] * (trispectrum_integrand(m, cx - dx * dev_A96[i], l1, l2, l3, l4) + trispectrum_integrand(m, cx + dx * dev_A96[i], l1, l2, l3, l4));
  return (q * dx);
}

__device__ double pentaspectrum_integrand(double m, double z, double l1, double l2, double l3, double l4, double l5, double l6)
{
  double didx = z / dev_z_max * (n_redshift_bins - 1);
  int idx = didx;
  didx = didx - idx;
  if (idx == n_redshift_bins - 1)
  {
    idx = n_redshift_bins - 2;
    didx = 1.;
  }
  double g = dev_g_array[idx] * (1 - didx) + dev_g_array[idx + 1] * didx;
  double chi = dev_f_K_array[idx] * (1 - didx) + dev_f_K_array[idx + 1] * didx;
  double rhobar = 2.7754e11; // critical density[Msun*h²/Mpc³]
  rhobar *= dev_om;

  double result = 1;

  result *= pow(g, 6);
  result /= pow(chi, 4);

  result *= dev_c_over_H0 / dev_E(z);
  result *= u_NFW(l1 / chi, m, z) * u_NFW(l2 / chi, m, z) * u_NFW(l3 / chi, m, z);
  result *= u_NFW(l4 / chi, m, z) * u_NFW(l5 / chi, m, z) * u_NFW(l6 / chi, m, z);
  result *= pow(m / rhobar, 6) * hmf(m, z);
  result *= pow(1.5 * dev_om / dev_c_over_H0 / dev_c_over_H0, 6);
  result *= pow(1 + z, 6);

  return result;
}

__device__ double pentaspectrum_integrand_ssc(double mmin, double mmax, double z, double l1, double l2, double l3, double l4, double l5, double l6)
{
  double didx = z / dev_z_max * (n_redshift_bins - 1);
  int idx = didx;
  didx = didx - idx;
  if (idx == n_redshift_bins - 1) 
  {
    idx = n_redshift_bins - 2;
    didx = 1.;
  }
  double g = dev_g_array[idx] * (1 - didx) + dev_g_array[idx + 1] * didx;
  double chi = dev_f_K_array[idx] * (1 - didx) + dev_f_K_array[idx + 1] * didx;
  double sigma2 = dev_sigma2_from_windowfunction_array[idx] * (1-didx) + dev_sigma2_from_windowfunction_array[idx+1]*didx;
  double rhobar = 2.7754e11; // critical density[Msun*h²/Mpc³]
  rhobar *= dev_om;

  double result = 1;

  result *= pow(g, 6);
  result /= pow(chi, 4);

  result *= dev_c_over_H0 / dev_E(z);
  result *= I_31(l1/chi, l2/chi, l3/chi, mmin, mmax, z);
  result *= I_31(l4/chi, l5/chi, l6/chi, mmin, mmax, z);
  result *= sigma2;
  result *= pow(1.5 * dev_om / dev_c_over_H0 / dev_c_over_H0, 6);
  result *= pow(1 + z, 6);

  return result;
}

__device__ double pentaspectrum_limber_integrated(double a, double b, double m, double l1, double l2, double l3, double l4, double l5, double l6)
{
  double cx, dx, q;
  cx = (a + b) / 2;
  dx = (b - a) / 2;
  q = 0;
  for (int i = 0; i < 48; i++)
  {
    q += dev_W96[i] * (pentaspectrum_integrand(m, cx - dx * dev_A96[i], l1, l2, l3, l4, l5, l6) + pentaspectrum_integrand(m, cx + dx * dev_A96[i], l1, l2, l3, l4, l5, l6));
  }

  return (q * dx);
}

__device__ double pentaspectrum_limber_mass_integrated(double zmin, double zmax, double logMmin, double logMmax, double l1, double l2, double l3, double l4, double l5, double l6)
{
  double cx, dx, q;
  cx = (logMmin + logMmax) / 2;
  dx = (logMmax - logMmin) / 2;
  q = 0;
  for (int i = 0; i < 48; i++)
  {
    double m1 = exp(cx - dx * dev_A96[i]);
    double m2 = exp(cx + dx * dev_A96[i]);

    q += dev_W96[i] * (m1 * pentaspectrum_limber_integrated(zmin, zmax, m1, l1, l2, l3, l4, l5, l6) + m2 * pentaspectrum_limber_integrated(zmin, zmax, m2, l1, l2, l3, l4, l5, l6));
  }

  return (q * dx);
}


__device__ double pentaspectrum_limber_integrated_ssc(double zmin, double zmax, double mmin, double mmax, double l1, double l2, double l3, double l4, double l5, double l6)
{
  double cx, dx, q;
  cx = (zmin + zmax) / 2;
  dx = (zmax - zmin) / 2;
  q = 0;
  for (int i = 0; i < 48; i++)
  {
    q += dev_W96[i] * (pentaspectrum_integrand_ssc(mmin, mmax, cx - dx * dev_A96[i], l1, l2, l3, l4, l5, l6) + pentaspectrum_integrand_ssc(mmin, mmax, cx + dx * dev_A96[i], l1, l2, l3, l4, l5, l6));
  }

  return (q * dx);
}

__device__ double powerspectrum_integrand(double m, double z, double l)
{
  double didx = z / dev_z_max * (n_redshift_bins - 1);
  int idx = didx;
  didx = didx - idx;
  if (idx == n_redshift_bins - 1)
  {
    idx = n_redshift_bins - 2;
    didx = 1.;
  }
  double g = dev_g_array[idx] * (1 - didx) + dev_g_array[idx + 1] * didx;
  double chi = dev_f_K_array[idx] * (1 - didx) + dev_f_K_array[idx + 1] * didx;
  double rhobar = 2.7754e11; // critical density[Msun*h²/Mpc³]
  rhobar *= dev_om;

  double result = hmf(m, z) * g * g;
  result *= dev_c_over_H0 / dev_E(z);
  result *= pow(u_NFW(l / chi, m, z), 2);
  result *= m * m / rhobar / rhobar;
  result *= pow(1.5 * dev_om / dev_c_over_H0 / dev_c_over_H0, 2); // Prefactor in h^4
  result *= pow(1 + z, 2);
  return result;
}

__device__ double powerspectrum_limber_integrated(double a, double b, double m, double l)
{
  int i;
  double cx, dx, q;
  cx = (a + b) / 2;
  dx = (b - a) / 2;
  q = 0;
  for (i = 0; i < 48; i++)
    q += dev_W96[i] * (powerspectrum_integrand(m, cx - dx * dev_A96[i], l) + powerspectrum_integrand(m, cx + dx * dev_A96[i], l));
  return (q * dx);
}

int integrand_Powerspectrum(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "integrand: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
  // Read data for integration
  PowerspecContainer *container = (PowerspecContainer *)thisPtr;

  if (npts > 1e8)
  {
    std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    return 1;
  };

  double l = container->l;
  // Allocate memory on device for integrand values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

  // Calculate values
  integrand_Powerspectrum_kernel<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, l, dev_value);

  cudaFree(dev_vars); // Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Free values

  return 0; // Success :)
}

__global__ void integrand_Powerspectrum_kernel(const double *vars, unsigned ndim, int npts, double l, double *value)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double m = vars[i * ndim];
    value[i] = powerspectrum_limber_integrated(0, dev_z_max, m, l);
  }
}

double Powerspectrum(const double &l)
{
  PowerspecContainer container;
  container.l = l;
  double result, error;

  double mmin = pow(10, logMmin);
  double mmax = pow(10, logMmax);
  double vals_min[1] = {mmin};
  double vals_max[1] = {mmax};

  hcubature_v(1, integrand_Powerspectrum, &container, 1, vals_min, vals_max, 0, 0, 1e-1, ERROR_L1, &result, &error);

  return result;
}

int integrand_Trispectrum(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "integrand: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
  // Read data for integration
  TrispecContainer *container = (TrispecContainer *)thisPtr;

  if (npts > 1e8)
  {
    std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    return 1;
  };

  double l1 = container->l1;
  double l2 = container->l2;
  double l3 = container->l3;
  double l4 = container->l4;

  // Allocate memory on device for integrand values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

  // Calculate values
  integrand_Trispectrum_kernel<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, l1, l2, l3, l4, dev_value);

  cudaFree(dev_vars); // Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Free values

  return 0; // Success :)
}

__global__ void integrand_Trispectrum_kernel(const double *vars, unsigned ndim, int npts, double l1, double l2, double l3, double l4, double *value)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double m = vars[i * ndim];
    value[i] = trispectrum_limber_integrated(0, dev_z_max, m, l1, l2, l3, l4);
  }
}

double Trispectrum(const double &l1, const double &l2, const double &l3, const double &l4)
{
  TrispecContainer container;
  container.l1 = l1;
  container.l2 = l2;
  container.l3 = l3;
  container.l4 = l4;
  double result, error;

  double mmin = pow(10, logMmin);
  double mmax = pow(10, logMmax);
  double vals_min[1] = {mmin};
  double vals_max[1] = {mmax};

  hcubature_v(1, integrand_Trispectrum, &container, 1, vals_min, vals_max, 0, 0, 1e-1, ERROR_L1, &result, &error);

  return result;
}

double Pentaspectrum(const double &l1, const double &l2, const double &l3, const double &l4, const double &l5, const double &l6)
{
  PentaspecContainer container;
  container.l1 = l1;
  container.l2 = l2;
  container.l3 = l3;
  container.l4 = l4;
  container.l5 = l5;
  container.l6 = l6;

  double result, error;

  double mmin = log(pow(10, logMmin));
  double mmax = log(pow(10, logMmax));
  double zmin = 0;
  double zmax = z_max;
  double vals_min[2] = {mmin, zmin};
  double vals_max[2] = {mmax, zmax};

  hcubature_v(1, integrand_Pentaspectrum, &container, 2, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);

  return result;
}

int integrand_Pentaspectrum(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "integrand: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
  // Read data for integration
  PentaspecContainer *container = (PentaspecContainer *)thisPtr;

  if (npts > 1e8)
  {
    std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    return 1;
  };

  double l1 = container->l1;
  double l2 = container->l2;
  double l3 = container->l3;
  double l4 = container->l4;
  double l5 = container->l5;
  double l6 = container->l6;

  // Allocate memory on device for integrand values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

  // Calculate values
  integrand_Pentaspectrum_kernel<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, l1, l2, l3, l4, l5, l6, dev_value);

  cudaFree(dev_vars); // Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Free values

  return 0; // Success :)
}

__global__ void integrand_Pentaspectrum_kernel(const double *vars, unsigned ndim, int npts, double l1, double l2, double l3,
                                               double l4, double l5, double l6, double *value)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double m = exp(vars[i * ndim]);
    double z = vars[i * ndim + 1];
    value[i] = m * pentaspectrum_integrand(m, z, l1, l2, l3, l4, l5, l6);
  }
}

__device__ double tetraspectrum_integrand(double m, double z, double l1, double l2, double l3, double l4, double l5)
{
  double didx = z / dev_z_max * (n_redshift_bins - 1);
  int idx = didx;
  didx = didx - idx;
  if (idx == n_redshift_bins - 1)
  {
    idx = n_redshift_bins - 2;
    didx = 1.;
  }
  double g = dev_g_array[idx] * (1 - didx) + dev_g_array[idx + 1] * didx;
  double chi = dev_f_K_array[idx] * (1 - didx) + dev_f_K_array[idx + 1] * didx;
  double rhobar = 2.7754e11; // critical density[Msun*h²/Mpc³]
  rhobar *= dev_om;

  double result = 1;

  result *= pow(g, 5);
  result /= pow(chi, 3);

  result *= dev_c_over_H0 / dev_E(z);
  result *= u_NFW(l1 / chi, m, z) * u_NFW(l2 / chi, m, z) * u_NFW(l3 / chi, m, z);
  result *= u_NFW(l4 / chi, m, z) * u_NFW(l5 / chi, m, z);
  result *= pow(m / rhobar, 5) * hmf(m, z);
  result *= pow(1.5 * dev_om / dev_c_over_H0 / dev_c_over_H0, 5);
  result *= pow(1 + z, 5);

  return result;
}

__device__ double tetraspectrum_limber_integrated(double a, double b, double m, double l1, double l2, double l3, double l4, double l5)
{
  double cx, dx, q;
  cx = (a + b) / 2;
  dx = (b - a) / 2;
  q = 0;
  for (int i = 0; i < 48; i++)
  {
    q += dev_W96[i] * (tetraspectrum_integrand(m, cx - dx * dev_A96[i], l1, l2, l3, l4, l5) + tetraspectrum_integrand(m, cx + dx * dev_A96[i], l1, l2, l3, l4, l5));
  }

  return (q * dx);
}

__device__ double integrand_I_31(const double& k1, const double& k2, const double& k3, const double& m, const double& z)
{
  double rhobar = 2.7754e11; // critical density[Msun*h²/Mpc³]
  rhobar *= dev_om;


  double result = u_NFW(k1, m, z) * u_NFW(k2, m, z) * u_NFW(k3, m, z);
  result *= pow(m/rhobar, 3)*hmf(m, z);
  
  result *= halo_bias(m, z);

  return result;

}


__device__ double I_31(const double& k1, const double& k2, const double& k3, const double& a, const double& b, const double& z)
{
  double cx, dx, q;
  double logMmin=log(a);
  double logMmax=log(b);
    cx = (logMmin + logMmax) / 2;
  dx = (logMmax - logMmin) / 2;

  q = 0;
  for (int i = 0; i < 48; i++)
  {
        double m1 = exp(cx - dx * dev_A96[i]);
    double m2 = exp(cx + dx * dev_A96[i]);
    q += dev_W96[i] * (m1* integrand_I_31(k1, k2, k3, m1, z) + m2*integrand_I_31(k1, k2, k3, m2,z));
  }

  return (q * dx);
}
