#include "apertureStatistics.hpp"
#include "bispectrum.hpp"
#include "cubature.h"
#include "helper.hpp"
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <gsl/gsl_sf_bessel.h>

double ApertureStatistics::uHat(const double &eta)
{
  double temp = 0.5 * eta * eta;
  return temp * exp(-temp);
}

double ApertureStatistics::integrand(const double &l1, const double &l2, const double &phi, std::vector<double> thetas)
{
  double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));

  if (!isfinite(l1) || !isfinite(l2) || !isfinite(l3))
  {
    std::cerr << "ApertureStatistics::integrand: One of the l's not finite" << std::endl;
    std::cerr << l1 << " " << l2 << " " << l3 << " " << phi << std::endl;
    exit(1);
  };

  return l1 * l2 * Bispectrum_->bkappa(l1, l2, l3) * uHat(l1 * thetas[0]) * uHat(l2 * thetas[1]) * uHat(l3 * thetas[2]);
}

double ApertureStatistics::integrand_4d(const double &l1, const double &l2, const double &phi, const double &z, std::vector<double> thetas)
{
  double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));

  if (!isfinite(l1) || !isfinite(l2) || !isfinite(l3))
  {
    std::cerr << "ApertureStatistics::integrand: One of the l's not finite" << std::endl;
    std::cerr << l1 << " " << l2 << " " << l3 << " " << phi << std::endl;
    exit(1);
  };

  struct ell_params ells;
  ells.ell1 = l1;
  ells.ell2 = l2;
  ells.ell3 = l3;

  return l1 * l2 * Bispectrum_->integrand_bkappa(z, ells) * uHat(l1 * thetas[0]) * uHat(l2 * thetas[1]) * uHat(l3 * thetas[2]);
}

#if DO_CYCLIC_PERMUTATIONS
double ApertureStatistics::_uHat_product(const double &l1, const double &l2, const double &l3, double *thetas)
{
  return uHat(l1 * thetas[0]) * uHat(l2 * thetas[1]) * uHat(l3 * thetas[2]);
}
double ApertureStatistics::uHat_product(const double &l1, const double &l2, const double &l3, double *thetas)
{
  std::cout << "test" << std::endl;
  return (_uHat_product(l1, l2, l3, thetas) + _uHat_product(l2, l3, l1, thetas) + _uHat_product(l3, l1, l2, thetas)) / 3.;
}
#else
double ApertureStatistics::uHat_product(const double &l1, const double &l2, const double &l3, std::vector<double> thetas)
{
  return uHat(l1 * thetas[0]) * uHat(l2 * thetas[1]) * uHat(l3 * thetas[2]);
}
#endif

double ApertureStatistics::uHat_product_permutations(const double &l1, const double &l2, const double &l3, std::vector<double> thetas)
{
  double _result;
  _result = uHat_product(l1, l2, l3, thetas);
  _result += uHat_product(l2, l3, l1, thetas);
  _result += uHat_product(l3, l1, l2, thetas);
  _result += uHat_product(l1, l3, l2, thetas);
  _result += uHat_product(l3, l2, l1, thetas);
  _result += uHat_product(l2, l1, l3, thetas);
  return _result;
}

double ApertureStatistics::integrand_Gaussian_Aperture_Covariance(const double &l1, const double &l2, const double &phi, const double &z,
                                                                  std::vector<double> thetas_123, std::vector<double> thetas_456)
{
  double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));
  double result = uHat_product(l1, l2, l3, thetas_123);
  result *= uHat_product_permutations(l1, l2, l3, thetas_456);
#if CONSTANT_POWERSPECTRUM
#elif ANALYTICAL_POWERSPECTRUM
  double P1 = p1 * l1 * l1 * exp(-p2 * l1 * l1);
  double P2 = p1 * l2 * l2 * exp(-p2 * l2 * l2);
  double P3 = p1 * l3 * l3 * exp(-p2 * l3 * l3);

  result *= P1 * P2 * P3;
#elif ANALYTICAL_POWERSPECTRUM_V2
  double P1 = p1 * l1 * exp(-p2 * l1);
  double P2 = p1 * l2 * exp(-p2 * l2);
  double P3 = p1 * l3 * exp(-p2 * l3);
  result *= P1 * P2 * P3;
#else
  result *= Bispectrum_->limber_integrand_triple_power_spectrum(l1, l2, l3, z);
#endif
  result *= l1 * l2;

  return result;
}

int ApertureStatistics::integrand_Gaussian_Aperture_Covariance(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "ApertureStatistics::integrand_Aperture_Covariance: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
#if CONSTANT_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM_V2
  if (ndim != 3)
  {
    std::cerr << "ApertureStatistics::integrand_Aperture_Covariance: Wrong number of variable dimensions" << std::endl;
    exit(1);
  };
#else
  if (ndim != 4)
  {
    std::cerr << "ApertureStatistics::integrand_Aperture_Covariance: Wrong number of variable dimensions" << std::endl;
    exit(1);
  };
#endif

  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  ApertureStatistics *apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas_123 = container->thetas;
  std::vector<double> thetas_456 = container->thetas2;

  if (npts > 1e8)
    std::cerr << "Npts: " << npts << " at thetas " << convert_rad_to_angle(thetas_123[0]) << ", " << convert_rad_to_angle(thetas_123[1]) << ", " << convert_rad_to_angle(thetas_123[2]) << ", " << convert_rad_to_angle(thetas_456[0]) << ", " << convert_rad_to_angle(thetas_456[1]) << ", " << convert_rad_to_angle(thetas_456[2]) << ", " << std::endl;

#if PARALLEL_INTEGRATION
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i < npts; i++)
  {
    double ell1 = vars[i * ndim];
    double ell2 = vars[i * ndim + 1];
    double phi = vars[i * ndim + 2];
#if CONSTANT_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM_V2
    double z = 0;
#else
    double z = vars[i * ndim + 3];
#endif

    value[i] = apertureStatistics->integrand_Gaussian_Aperture_Covariance(ell1, ell2, phi, z, thetas_123, thetas_456);
    if(!isfinite(value[i]))
    {
      std::cerr<<value[i]<<" "<<ell1<<" "<<ell2<<" "<<phi<<" "<<z<<std::endl;
    }
  }
  return 0; // Success :)
}

double ApertureStatistics::integrand_phi(double phi, void *thisPtr)
{
  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  ApertureStatistics *apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas = container->thetas;

  return apertureStatistics->integrand(apertureStatistics->l1_, apertureStatistics->l2_, phi, thetas);
}

double ApertureStatistics::integral_phi(double l1, double l2, std::vector<double> thetas)
{
  l1_ = l1;
  l2_ = l2;

  // variables for result and integration error
  double result, error;

  // cast kernel to gsl_function
  gsl_function F;
  F.function = &ApertureStatistics::integrand_phi;
  ApertureStatisticsContainer container;
  container.aperturestatistics = this;
  container.thetas = thetas;
  F.params = &container;

  // perform integration
  gsl_integration_qags(&F, phiMin, phiMax, 0.0, 1e-4, 1000, w_phi, &result, &error);

  return result;
}

double ApertureStatistics::integrand_l2(double l2, void *thisPtr)
{
  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  ApertureStatistics *apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas = container->thetas;

  return apertureStatistics->integral_phi(apertureStatistics->l1_, l2, thetas);
}

double ApertureStatistics::integral_l2(double l1, std::vector<double> thetas)
{
  l1_ = l1;

  // variables for result and integration error
  double result, error;

  // cast kernel to gsl_function
  gsl_function F;
  F.function = &ApertureStatistics::integrand_l2;
  ApertureStatisticsContainer container;
  container.aperturestatistics = this;
  container.thetas = thetas;
  F.params = &container;

  // perform integration
  gsl_integration_qags(&F, lMin, lMax, 0.0, 1e-4, 1000, w_l2, &result, &error);

  return result;
}

double ApertureStatistics::integrand_l1(double l1, void *thisPtr)
{
  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  ApertureStatistics *apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas = container->thetas;

  return apertureStatistics->integral_l2(l1, thetas);
}

double ApertureStatistics::integral_l1(std::vector<double> thetas)
{

  // variables for result and integration error
  double result, error;

  // cast kernel to gsl_function
  gsl_function F;
  F.function = &ApertureStatistics::integrand_l1;
  ApertureStatisticsContainer container;
  container.aperturestatistics = this;
  container.thetas = thetas;
  F.params = &container;

  // perform integration
  gsl_integration_qags(&F, lMin, lMax, 0.0, 1e-4, 1000, w_l1, &result, &error);

  return result;
}

int ApertureStatistics::integrand(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "ApertureStatistics::integrand: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  ApertureStatistics *apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas = container->thetas;

#if PARALLEL_INTEGRATION
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i < npts; i++)
  {
    double ell1 = vars[i * ndim];
    double ell2 = vars[i * ndim + 1];
    double phi = vars[i * ndim + 2];

    value[i] = apertureStatistics->integrand(ell1, ell2, phi, thetas);
  }

  return 0; // Success :)
}

int ApertureStatistics::integrand_4d(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "ApertureStatistics::integrand: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  ApertureStatistics *apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas = container->thetas;

#if PARALLEL_INTEGRATION
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i < npts; i++)
  {
    double ell1 = vars[i * ndim];
    double ell2 = vars[i * ndim + 1];
    double phi = vars[i * ndim + 2];
    double z = vars[i * ndim + 3];

    value[i] = apertureStatistics->integrand_4d(ell1, ell2, phi, z, thetas);
  }

  return 0; // Success :)
}

ApertureStatistics::ApertureStatistics(BispectrumCalculator *Bispectrum)
{

  // Set bispectrum
  Bispectrum_ = Bispectrum;

  // Allocate GSL workspaces
  w_l1 = gsl_integration_workspace_alloc(1000);
  w_l2 = gsl_integration_workspace_alloc(1000);
  w_phi = gsl_integration_workspace_alloc(1000);
}

double ApertureStatistics::MapMapMap(const std::vector<double> &thetas)
{

  // Set maximal l value such, that theta*l <= 10
  double thetaMin = *std::min_element(std::begin(thetas), std::end(thetas));
  lMax = 10. / thetaMin;

  ApertureStatisticsContainer container;
  container.aperturestatistics = this;
  container.thetas = thetas;
  double result, error;

#if CUBATURE // Do cubature integration

#if INTEGRATE4D // do limber integration via cubature
  double vals_min[4] = {lMin, lMin, phiMin, 0};
  double vals_max[4] = {lMax, lMax, phiMax, Bispectrum_->z_max};

  pcubature_v(1, integrand_4d, &container, 4, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);
  result *= 27. / 8. * pow(Bispectrum_->cosmo->om, 3) * pow(100. / 299792., 5); // account for prefactor of limber integration

#else // do limber integration separately
  double vals_min[3] = {lMin, lMin, phiMin};
  double vals_max[3] = {lMax, lMax, phiMax};

  hcubature_v(1, integrand, &container, 3, vals_min, vals_max, 0, 0, 1e-6, ERROR_L1, &result, &error);
#endif
#else // Do standard GSL integration
  result = integral_l1();
#endif

  return result / 8 / M_PI / M_PI / M_PI; // Divided by (2*pi)³
}

double ApertureStatistics::MapMapMap_covariance_Gauss(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456, double survey_area)
{

  // Set maximal l value such, that theta*l <= 10
  double thetaMin_123 = *std::min_element(std::begin(thetas_123), std::end(thetas_123));
  double thetaMin_456 = *std::min_element(std::begin(thetas_456), std::end(thetas_456));
  double thetaMin = std::max({thetaMin_123, thetaMin_456}); // should increase runtime, if either theta_123 or theta_456 is zero, so is their product
  lMax = 10. / thetaMin;

  ApertureStatisticsContainer container;
  container.aperturestatistics = this;
  container.thetas = thetas_123;
  container.thetas2 = thetas_456;
  double result, error;

#if CUBATURE // Do cubature integration

#if INTEGRATE4D // do limber integration via cubature
#if CONSTANT_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM_V2
  double vals_min[3] = {lMin, lMin, phiMin};
  double vals_max[3] = {lMax, lMax, phiMax / 2.}; // use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

  pcubature_v(1, integrand_Gaussian_Aperture_Covariance, &container, 3, vals_min, vals_max, 0, 0, 1e-6, ERROR_L1, &result, &error);

#else
  double vals_min[4] = {lMin, lMin, phiMin, 0};
  double vals_max[4] = {lMax, lMax, phiMax / 2., Bispectrum_->z_max}; // use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

  hcubature_v(1, integrand_Gaussian_Aperture_Covariance, &container, 4, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);
#endif

#else // do limber integration separately
  std::cerr << "MapMapMap_covariance_Gauss: 3-dimensional cubature integration not implemented!" << std::endl;
  exit(-1);
#endif
#else // Do standard GSL integration
  std::cerr << "MapMapMap_covariance_Gauss: GSL integration not implemented!" << std::endl;
  exit(-1);
#endif

  return 2. * result / survey_area / 8 / M_PI / M_PI / M_PI;
}

double ApertureStatistics::G(double ellX, double ellY, double thetaMax)
{
  double tmp1 = 0.5 * ellX * thetaMax;
  double tmp2 = 0.5 * ellY * thetaMax;

  double j01, j02;
  if (abs(tmp1) <= 1e-6)
  {
    j01 = 1;
  }
  else
  {
    j01 = sin(tmp1) / tmp1;
  }

  if (abs(tmp2) <= 1e-6)
  {
    j02 = 1;
  }
  else
  {
    j02 = sin(tmp2) / tmp2;
  }

  return j01 * j01 * j02 * j02;
};

double ApertureStatistics::G_circular(double ell, double thetaMax)
{
  double tmp = thetaMax * ell;
  double result = gsl_sf_bessel_J1(tmp);
  result *= result;
  result *= 4 / tmp / tmp;
  return result;
}

double ApertureStatistics::integrand_L1(double a, double b, double c, double d, double e, double f, double thetaMax,
                                        double theta1, double theta2, double theta3, double theta4, double theta5, double theta6)
{
  double Gfactor = G(a, b, thetaMax);

  double ell1 = sqrt((a - c - e) * (a - c - e) + (b - d - f) * (b - d - f));
  double ell2 = sqrt(c * c + d * d);
  double ell3 = sqrt(e * e + f * f);

  if (ell1 == 0 || ell2 == 0 || ell3 == 0)
    return 0;

  double result = Bispectrum_->Pell(ell1) * Bispectrum_->Pell(ell2) * Bispectrum_->Pell(ell3);
  result *= uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3 * theta3);
  result *= uHat(ell1 * theta4) * uHat(ell2 * theta5) * uHat(ell3 * theta6);
  result *= Gfactor;

  return result;
}

double ApertureStatistics::integrand_L1_circular(double a, double b, double c, double d, double e, double f, double thetaMax,
                                                 double theta1, double theta2, double theta3, double theta4, double theta5, double theta6)
{
  double ell = sqrt(a * a + b * b);
  double Gfactor = G_circular(ell, thetaMax);

  double ell1 = sqrt((a - c - e) * (a - c - e) + (b - d - f) * (b - d - f));
  double ell2 = sqrt(c * c + d * d);
  double ell3 = sqrt(e * e + f * f);

  if (ell1 == 0 || ell2 == 0 || ell3 == 0)
    return 0;

  double result = Bispectrum_->Pell(ell1) * Bispectrum_->Pell(ell2) * Bispectrum_->Pell(ell3);
  result *= uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3 * theta3);
  result *= uHat(ell1 * theta4) * uHat(ell2 * theta5) * uHat(ell3 * theta6);
  result *= Gfactor;

  return result;
}

double ApertureStatistics::integrand_L2_A(double ell, double theta1, double theta2)
{
  double result = ell * Bispectrum_->Pell(ell);
  result *= uHat(ell * theta1) * uHat(ell * theta2);
  return result;
}

double ApertureStatistics::integrand_L2_B(double ellX, double ellY, double thetaMax, double theta1, double theta2)
{
  double Gfactor = G(ellX, ellY, thetaMax);
  double ell = sqrt(ellX * ellX + ellY * ellY);
  double result = Bispectrum_->Pell(ell);
  result *= uHat(ell * theta1) * uHat(ell * theta2);
  result *= Gfactor;

  return result;
}

double ApertureStatistics::integrand_L2_B_circular(double ell, double thetaMax, double theta1, double theta2)
{
  double Gfactor = G_circular(ell, thetaMax);
  double result = Bispectrum_->Pell(ell);
  result *= ell * uHat(ell * theta1) * uHat(ell * theta2);
  result *= Gfactor;

  return result;
}

double ApertureStatistics::integrand_L4_circular(double v, double ell2, double ell4, double ell5, double alphaV, double alpha2, double alpha4,
                              double alpha5, double thetaMax, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6)
{
  double ell1=sqrt(v*v+ell4*ell4-2*v*ell4*cos(alpha4-alphaV));
  double ell3=sqrt(ell1*ell1+ell2*ell2+2*ell1*ell2*cos(alpha2-alpha4+alphaV));
  double ell6=sqrt(ell4*ell4+ell5*ell5+2*ell4*ell5*cos(alpha5-alpha4));


  double Gfactor = G_circular(v, thetaMax);


  double result = Bispectrum_->bkappa(ell4, ell2, ell3);
  result *= Bispectrum_->bkappa(ell1, ell5, ell6);
  result *= uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3*theta3)*uHat(ell4*theta4)*uHat(ell5*theta5)*uHat(ell6*theta6);
  result *= v*ell2*ell4*ell5;
  result *= Gfactor;

  return result;
}

int ApertureStatistics::integrand_L1(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{

  if (fdim != 1)
  {
    std::cerr << "ApertureStatistics::integrand_L1: Wrong number of function dimensions" << std::endl;
    exit(1);
  };

  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  ApertureStatistics *apertureStatistics = container->aperturestatistics;

  if (npts > 1e8)
  {
    std::cerr << "WARNING: Large Number of points:" << npts << " "
              << std::endl;
  };

#pragma omp parallel for
  for (unsigned int i = 0; i < npts; i++)
  {
    double ell1X = vars[i * ndim];
    double ell1Y = vars[i * ndim + 1];
    double ell2X = vars[i * ndim + 2];
    double ell2Y = vars[i * ndim + 3];
    double ell3X = vars[i * ndim + 4];
    double ell3Y = vars[i * ndim + 5];
#if CIRCULAR_SURVEY
    value[i] = apertureStatistics->integrand_L1_circular(ell1X, ell1Y, ell2X, ell2Y, ell3X, ell3Y, container->thetaMax,
                                                         container->thetas.at(0), container->thetas.at(1), container->thetas.at(2),
                                                         container->thetas2.at(0), container->thetas2.at(1), container->thetas2.at(2));
#else
    value[i] = apertureStatistics->integrand_L1(ell1X, ell1Y, ell2X, ell2Y, ell3X, ell3Y, container->thetaMax,
                                                container->thetas.at(0), container->thetas.at(1), container->thetas.at(2),
                                                container->thetas2.at(0), container->thetas2.at(1), container->thetas2.at(2));
#endif
  }
  return 0;
}

int ApertureStatistics::integrand_L2_A(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "ApertureStatistics::integrand_L1: Wrong number of function dimensions" << std::endl;
    exit(1);
  };

  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  ApertureStatistics *apertureStatistics = container->aperturestatistics;

#pragma omp parallel for
  for (unsigned int i = 0; i < npts; i++)
  {
    double ell = vars[i * ndim];

    value[i] = apertureStatistics->integrand_L2_A(ell, container->thetas.at(0), container->thetas.at(1));
  }
  return 0;
}

int ApertureStatistics::integrand_L2_B(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "ApertureStatistics::integrand_L1: Wrong number of function dimensions" << std::endl;
    exit(1);
  };

  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  ApertureStatistics *apertureStatistics = container->aperturestatistics;

#pragma omp parallel for
  for (unsigned int i = 0; i < npts; i++)
  {
#if CIRCULAR_SURVEY
    double ell = vars[i * ndim];
    value[i] = 2 * M_PI * apertureStatistics->integrand_L2_B_circular(ell, container->thetaMax, container->thetas.at(0), container->thetas.at(1));
#else
    double ellX = vars[i * ndim];
    double ellY = vars[i * ndim + 1];

    value[i] = apertureStatistics->integrand_L2_B(ellX, ellY, container->thetaMax,
                                                  container->thetas.at(0), container->thetas.at(1));
#endif
  }
  return 0;
}


int ApertureStatistics::integrand_L4(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "ApertureStatistics::integrand_L1: Wrong number of function dimensions" << std::endl;
    exit(1);
  };

  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  ApertureStatistics *apertureStatistics = container->aperturestatistics;

#pragma omp parallel for
  for (unsigned int i = 0; i < npts; i++)
  {
#if CIRCULAR_SURVEY
    double v = vars[i * ndim];
    double ell2 = vars[i*ndim+1];
    double ell4 = vars[i*ndim+2];
    double ell5 = vars[i*ndim+3];
    double alphaV = vars[i*ndim+4];
    double alpha2 = vars[i*ndim+5];
    double alpha4 = vars[i*ndim+6];
    double alpha5 = vars[i*ndim+7];


    value[i] = apertureStatistics->integrand_L4_circular(v, ell2, ell4, ell5, alphaV, alpha2, alpha4, alpha5,
                                                         container->thetaMax, container->thetas.at(0), container->thetas.at(1),
                                                         container->thetas.at(2), container->thetas2.at(0), container->thetas2.at(1),
                                                         container->thetas2.at(2));
#else
  std::cerr<<"L4 only coded for circular survey!"<<std::endl;
  exit(1);
#endif
  }
  return 0;
}

double ApertureStatistics::L1(double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double thetaMax)
{
  std::vector<double> thetas1{theta1, theta2, theta3};
  std::vector<double> thetas2{theta4, theta5, theta6};

  ApertureStatisticsContainer container;
  container.aperturestatistics = this;
  container.thetas = thetas1;
  container.thetas2 = thetas2;
  container.thetaMax = thetaMax;
  double result, error;

  double vals_min[6] = {-1.1e4, -1.1e4, -1.1e4, -1.1e4, -1.1e4, -1.1e4};
  double vals_max[6] = {1e4, 1e4, 1e4, 1e4, 1e4, 1e4};
  int errcode = hcubature_v(1, integrand_L1, &container, 6, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
  if (errcode != 0)
  {
    std::cerr << "errcode in hcubature:" << errcode << std::endl;
  };
  std::cerr << "res L1:" << result << " " << error << std::endl;
  return result;
}

double ApertureStatistics::L2(double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double thetaMax)
{

  // Integral over ell 1
  std::vector<double> thetas1{theta1, theta2};
  ApertureStatisticsContainer container;
  container.aperturestatistics = this;
  container.thetas = thetas1;
  container.thetaMax = thetaMax;

  double result_A1, error_A1;

  double vals_min1[1] = {1e-1};
  double vals_max1[1] = {1e4};
  hcubature_v(1, integrand_L2_A, &container, 1, vals_min1, vals_max1, 0, 0, 1e-3, ERROR_L1, &result_A1, &error_A1);

  // Integral over ell 3
  std::vector<double> thetas2{theta5, theta6};
  container.thetas = thetas2;

  double result_A2, error_A2;

  hcubature_v(1, integrand_L2_A, &container, 1, vals_min1, vals_max1, 0, 0, 1e-3, ERROR_L1, &result_A2, &error_A2);

  // Integral over ell 2
  std::vector<double> thetas3{theta3, theta4};
  container.thetas = thetas3;

  double result_B, error_B;

#if CIRCULAR_SURVEY
  hcubature_v(1, integrand_L2_B, &container, 1, vals_min1, vals_max1, 0, 0, 1e-3, ERROR_L1, &result_B, &error_B);
#else
  double vals_min2[2] = {-1e4, -1e4};
  double vals_max2[2] = {1e4, 1e4};
  hcubature_v(1, integrand_L2_B, &container, 2, vals_min2, vals_max2, 0, 0, 1e-3, ERROR_L1, &result_B, &error_B);
#endif

  double result = result_A1 * result_A2 * result_B;
  return result;
}


double ApertureStatistics::L4(double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double thetaMax)
{

  std::vector<double> thetas1{theta1, theta2, theta3};
  std::vector<double> thetas2{theta4, theta5, theta6};
  ApertureStatisticsContainer container;
  container.aperturestatistics = this;
  container.thetas = thetas1;
  container.thetas2 = thetas2;
  container.thetaMax = thetaMax;

  double result, error;

#if CIRCULAR_SURVEY
  double vals_min[8] = {1e-1, 1e-1, 1e-1, 1e-1, phiMin, phiMin, phiMin, phiMin};
  double vals_max[8] = {1e4, 1e4, 1e4, 1e4, phiMax, phiMax, phiMax, phiMax};
  pcubature_v(1, integrand_L4, &container, 8, vals_min, vals_max, 0, 0, 0.2, ERROR_L1, &result, &error);
#else
  std::cerr<<"L4 only coded for circular survey"<<std::endl;
  exit(1),
#endif

  return result;
}

double ApertureStatistics::L1_total(const std::vector<double> &thetas123, const std::vector<double> &thetas456, double thetaMax)
{
  double term1 = L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(0), thetas456.at(1), thetas456.at(2), thetaMax);
  term1 += L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(0), thetas456.at(2), thetas456.at(1), thetaMax);
  term1 += L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(1), thetas456.at(0), thetas456.at(2), thetaMax);
  term1 += L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(1), thetas456.at(2), thetas456.at(0), thetaMax);
  term1 += L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(2), thetas456.at(0), thetas456.at(1), thetaMax);
  term1 += L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(2), thetas456.at(1), thetas456.at(0), thetaMax);

  term1 /= pow(2 * M_PI, 6);
  return term1;
}

double ApertureStatistics::L2_total(const std::vector<double> &thetas123, const std::vector<double> &thetas456, double thetaMax)
{
  double term2 = L2(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(0), thetas456.at(1), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(1), thetas456.at(0), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(2), thetas456.at(1), thetas456.at(0), thetaMax);
  term2 += L2(thetas123.at(0), thetas123.at(2), thetas123.at(1), thetas456.at(0), thetas456.at(1), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(0), thetas123.at(2), thetas123.at(1), thetas456.at(1), thetas456.at(0), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(0), thetas123.at(2), thetas123.at(1), thetas456.at(2), thetas456.at(1), thetas456.at(0), thetaMax);
  term2 += L2(thetas123.at(1), thetas123.at(2), thetas123.at(0), thetas456.at(0), thetas456.at(1), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(1), thetas123.at(2), thetas123.at(0), thetas456.at(1), thetas456.at(0), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(1), thetas123.at(2), thetas123.at(0), thetas456.at(2), thetas456.at(1), thetas456.at(0), thetaMax);

  term2 /= pow(2 * M_PI, 4);

  return term2;
}


double ApertureStatistics::L4_total(const std::vector<double> &thetas123, const std::vector<double> &thetas456, double thetaMax)
{
  double term4 = L4(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(0), thetas456.at(1), thetas456.at(2), thetaMax);
  term4+=L4(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(1), thetas456.at(0), thetas456.at(2), thetaMax);
  term4+=L4(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(2), thetas456.at(1), thetas456.at(1), thetaMax);
  term4+=L4(thetas123.at(1), thetas123.at(0), thetas123.at(2), thetas456.at(0), thetas456.at(1), thetas456.at(2), thetaMax);
  term4+=L4(thetas123.at(1), thetas123.at(0), thetas123.at(2), thetas456.at(1), thetas456.at(0), thetas456.at(2), thetaMax);
  term4+=L4(thetas123.at(1), thetas123.at(0), thetas123.at(2), thetas456.at(2), thetas456.at(1), thetas456.at(0), thetaMax);
  term4+=L4(thetas123.at(2), thetas123.at(0), thetas123.at(1), thetas456.at(0), thetas456.at(1), thetas456.at(2), thetaMax);
  term4+=L4(thetas123.at(2), thetas123.at(0), thetas123.at(1), thetas456.at(1), thetas456.at(0), thetas456.at(2), thetaMax);
  term4+=L4(thetas123.at(2), thetas123.at(0), thetas123.at(1), thetas456.at(2), thetas456.at(1), thetas456.at(0), thetaMax);
  
  term4 /= pow(2*M_PI, 8);
  return term4;
}

double ApertureStatistics::Cov(const std::vector<double> &thetas123, const std::vector<double> &thetas456, double thetaMax)
{

  double term1 = L1_total(thetas123, thetas456, thetaMax);

  double term2 = L1_total(thetas123, thetas456, thetaMax);

  return term1 + term2;
}

double ApertureStatistics::Cov_NG(const std::vector<double> &thetas123, const std::vector<double> &thetas456, double thetaMax)
{

  double term = L4_total(thetas123, thetas456, thetaMax);

  return term;
}

double ApertureStatistics::integrand_NonGaussian_Aperture_Covariance(const double &l1, const double &l2, const double &l5,
                                                                     const double &phi1, const double &phi2, const double &z,
                                                                     const double &theta1, const double &theta2, const double &theta3,
                                                                     const double &theta4, const double &theta5, const double &theta6)
{
#if CONSTANT_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM_V2
  std::cerr << "Non Gaussian Cov doesn't make sense for GRFs!" << std::endl;
  exit(1);
#endif
  double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi1));
  double l6 = sqrt(l1 * l1 + l5 * l5 + 2 * l1 * l5 * cos(phi2));

  double result = uHat(l1 * theta4) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l1 * theta1) * uHat(l5 * theta5) * uHat(l6 * theta6);
  result += uHat(l1 * theta5) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l1 * theta1) * uHat(l5 * theta4) * uHat(l6 * theta6);
  result += uHat(l1 * theta6) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l1 * theta1) * uHat(l5 * theta4) * uHat(l6 * theta5);
  result += uHat(l1 * theta4) * uHat(l2 * theta1) * uHat(l3 * theta3) * uHat(l1 * theta2) * uHat(l5 * theta5) * uHat(l6 * theta6);
  result += uHat(l1 * theta5) * uHat(l2 * theta1) * uHat(l3 * theta3) * uHat(l1 * theta2) * uHat(l5 * theta4) * uHat(l6 * theta6);
  result += uHat(l1 * theta6) * uHat(l2 * theta1) * uHat(l3 * theta3) * uHat(l1 * theta2) * uHat(l5 * theta4) * uHat(l6 * theta5);
  result += uHat(l1 * theta4) * uHat(l2 * theta1) * uHat(l3 * theta2) * uHat(l1 * theta3) * uHat(l5 * theta5) * uHat(l6 * theta6);
  result += uHat(l1 * theta5) * uHat(l2 * theta1) * uHat(l3 * theta2) * uHat(l1 * theta3) * uHat(l5 * theta4) * uHat(l6 * theta6);
  result += uHat(l1 * theta6) * uHat(l2 * theta1) * uHat(l3 * theta2) * uHat(l1 * theta3) * uHat(l5 * theta4) * uHat(l6 * theta5);

  struct ell_params ells_123;
  ells_123.ell1 = l1;
  ells_123.ell2 = l2;
  ells_123.ell3 = l3;

  struct ell_params ells_156;
  ells_156.ell1 = l1;
  ells_156.ell2 = l5;
  ells_156.ell3 = l6;

  result *= Bispectrum_->integrand_bkappa(z, ells_123);
  result *= Bispectrum_->integrand_bkappa(z, ells_156);
  result *= l1 * l2 * l5;

  if(!isfinite(result))
  {
    std::cerr<<result<<" "
            <<l1 << " " << l2 << " " << l5 << " " << phi1 <<" " <<phi2<<std::endl;
  };
  return result;
}


int ApertureStatistics::integrand_NonGaussian_Aperture_Covariance(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "ApertureStatistics::integrand_NonGaussian_Aperture_Covariance: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
#if CONSTANT_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM_V2
    std::cerr << "NonGaussian Covariance is zero for GRF" << std::endl;
    exit(1); 
#endif

  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  ApertureStatistics *apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas_123 = container->thetas;
  std::vector<double> thetas_456 = container->thetas2;

  if(npts > 1e8)
  {
    std::cerr << "Ran out of memory"<<std::endl;
    return 1;
  };
#if PARALLEL_INTEGRATION
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i < npts; i++)
  {
    double ell1 = vars[i * ndim];
    double ell2 = vars[i * ndim + 1];
    double ell5 = vars[i*ndim +2];
    double phi1 = vars[i*ndim +3];
    double phi2 = vars[i * ndim + 4];
    double z = vars[i * ndim + 5];

    value[i] = apertureStatistics->integrand_NonGaussian_Aperture_Covariance(ell1, ell2, ell5, phi1,phi2,z, 
                                                                              thetas_123.at(0), thetas_123.at(1), thetas_123.at(2),
                                                                              thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));

    
  }
  return 0; // Success :)
}


double ApertureStatistics::MapMapMap_covariance_NonGauss(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456, double survey_area)
{

  // Set maximal l value such, that theta*l <= 10
  double thetaMin_123 = *std::min_element(std::begin(thetas_123), std::end(thetas_123));
  double thetaMin_456 = *std::min_element(std::begin(thetas_456), std::end(thetas_456));
  double thetaMin = std::max({thetaMin_123, thetaMin_456}); // should increase runtime, if either theta_123 or theta_456 is zero, so is their product
  lMax = 10. / thetaMin;

  ApertureStatisticsContainer container;
  container.aperturestatistics = this;
  container.thetas = thetas_123;
  container.thetas2 = thetas_456;
  double result, error;

#if CUBATURE // Do cubature integration

#if INTEGRATE4D // do limber integration via cubature
#if CONSTANT_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM_V2
    std::cerr << "NonGaussian Covariance is zero for GRF" << std::endl;
    std::cerr << "No calculations performed" <<std::endl;
    exit(1); 
#else
  double vals_min[6] = {lMin, lMin, lMin, phiMin, phiMin, 0};
  double vals_max[6] = {lMax, lMax, lMax, M_PI, M_PI, Bispectrum_->z_max}; // use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

  hcubature_v(1, integrand_NonGaussian_Aperture_Covariance, &container, 6, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
#endif

#else // do limber integration separately
  std::cerr << "MapMapMap_covariance_NonGauss: 3-dimensional cubature integration not implemented!" << std::endl;
  exit(-1);
#endif
#else // Do standard GSL integration
  std::cerr << "MapMapMap_covariance_NonGauss: GSL integration not implemented!" << std::endl;
  exit(-1);
#endif
  result= 4. * result / survey_area / pow(2*M_PI, 5);
  result *= 27. / 8. * pow(Bispectrum_->cosmo->om, 3) * pow(100. / 299792., 5); // account for prefactor of limber integration
  result *= 27. / 8. * pow(Bispectrum_->cosmo->om, 3) * pow(100. / 299792., 5); // account for prefactor of limber integration
  std::cerr<<result<<std::endl;
  return result;
}