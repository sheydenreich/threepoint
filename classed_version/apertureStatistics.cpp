#include "apertureStatistics.hpp"
#include "bispectrum.hpp"
#include "cubature.h"
#include "helper.hpp"
#include <cmath>
#include <omp.h>
#include <algorithm>

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
    std::cout << "Npts: " << npts << " at thetas " << convert_rad_to_angle(thetas_123[0]) << ", " << convert_rad_to_angle(thetas_123[1]) << ", " << convert_rad_to_angle(thetas_123[2]) << ", " << convert_rad_to_angle(thetas_456[0]) << ", " << convert_rad_to_angle(thetas_456[1]) << ", " << convert_rad_to_angle(thetas_456[2]) << ", " << std::endl;

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

  pcubature_v(1, integrand_Gaussian_Aperture_Covariance, &container, 4, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);
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
  return 2.0 / (ellX * ellX * ellY * ellY) * (1 - cos(ellX * thetaMax)) * (1 - cos(ellY * thetaMax));
};

double ApertureStatistics::integrand_L1(double ell1X, double ell1Y, double ell2X, double ell2Y, double ell3X, double ell3Y, double thetaMax,
                                        double theta1, double theta2, double theta3, double theta4, double theta5, double theta6)
{
  double Gfactor = G(ell1X + ell2X + ell3X, ell1Y + ell2Y + ell3Y, thetaMax);

  double ell1 = sqrt(ell1X * ell1X + ell1Y * ell1Y);
  double ell2 = sqrt(ell2X * ell2X + ell2Y * ell2Y);
  double ell3 = sqrt(ell3X * ell3X + ell3Y * ell3Y);

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

int ApertureStatistics::integrand_L1(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  std::cerr<<npts<<std::endl;
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
    double ell1X = vars[i * ndim];
    double ell1Y = vars[i * ndim + 1];
    double ell2X = vars[i * ndim + 2];
    double ell2Y = vars[i * ndim + 3];
    double ell3X = vars[i * ndim + 4];
    double ell3Y = vars[i * ndim + 5];

    value[i] = apertureStatistics->integrand_L1(ell1X, ell1Y, ell2X, ell2Y, ell3X, ell3Y, container->thetaMax,
                                                container->thetas.at(0), container->thetas.at(1), container->thetas.at(2),
                                                container->thetas2.at(0), container->thetas2.at(1), container->thetas2.at(2));
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
    double ellX = vars[i * ndim];
    double ellY = vars[i * ndim + 1];
    
    value[i] = apertureStatistics->integrand_L2_B(ellX, ellY, container->thetaMax,
                                                container->thetas.at(0), container->thetas.at(1));
  }
  return 0;
}

double ApertureStatistics::L1(double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double thetaMax)
{
  std::vector<double> thetas1 {theta1, theta2, theta3};
  std::vector<double> thetas2 {theta4, theta5, theta6};

  ApertureStatisticsContainer container;
  container.aperturestatistics = this;
  container.thetas = thetas1;
  container.thetas2 = thetas2;
  container.thetaMax = thetaMax;
  double result, error;

  double vals_min[6] = {lMin, lMin, lMin, lMin, lMin, lMin};
  double vals_max[6] = {lMax, lMax, lMax, lMax, lMax, lMax};
  hcubature_v(1, integrand_L1, &container, 6, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);
  std::cerr<<result<<std::endl;
  return 2*result;
}

double ApertureStatistics::L2(double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double thetaMax)
{


// Integral over ell 1
  std::vector<double> thetas1 {theta1, theta2};
  ApertureStatisticsContainer container;
  container.aperturestatistics = this;
  container.thetas = thetas1;
  container.thetaMax = thetaMax;
  
  double result_A1, error_A1;

  double vals_min1[1] = {lMin};
  double vals_max1[1] = {lMax};
  hcubature_v(1, integrand_L2_A, &container, 1, vals_min1, vals_max1, 0, 0, 1e-3, ERROR_L1, &result_A1, &error_A1);

// Integral over ell 3
  std::vector<double> thetas2 {theta5, theta6};
  container.thetas = thetas2;
  
  double result_A2, error_A2;

  hcubature_v(1, integrand_L2_A, &container, 1, vals_min1, vals_max1, 0, 0, 1e-3, ERROR_L1, &result_A2, &error_A2);

// Integral over ell 2
  std::vector<double> thetas3{theta3, theta4};
  container.thetas = thetas3;
  
  double result_B, error_B;

  double vals_min2[2] = {lMin, lMin};
  double vals_max2[2] = {lMax, lMax};
  hcubature_v(1, integrand_L2_B, &container, 2, vals_min2, vals_max2, 0, 0, 1e-3, ERROR_L1, &result_B, &error_B);

  double result=2*result_A1*result_A2*result_B;
  //std::cerr<<result<<std::endl;
  return result;
}

double ApertureStatistics::Cov(const std::vector<double>& thetas123, const std::vector<double>& thetas456, double thetaMax)
{
  double term1 = L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(0), thetas456.at(1), thetas456.at(2), thetaMax);
  term1 += L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(0), thetas456.at(2), thetas456.at(1), thetaMax);
  term1 += L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(1), thetas456.at(0), thetas456.at(2), thetaMax);
  term1 += L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(1), thetas456.at(2), thetas456.at(0), thetaMax);
  term1 += L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(2), thetas456.at(0), thetas456.at(1), thetaMax);
  term1 += L1(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(2), thetas456.at(1), thetas456.at(0), thetaMax);

  term1/=pow(2*M_PI, 6)*thetaMax*thetaMax;

  double term2 = L2(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(0), thetas456.at(1), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(1), thetas456.at(0), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(0), thetas123.at(1), thetas123.at(2), thetas456.at(2), thetas456.at(1), thetas456.at(0), thetaMax);
  term2 += L2(thetas123.at(0), thetas123.at(2), thetas123.at(1), thetas456.at(0), thetas456.at(1), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(0), thetas123.at(2), thetas123.at(1), thetas456.at(1), thetas456.at(0), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(0), thetas123.at(2), thetas123.at(1), thetas456.at(2), thetas456.at(1), thetas456.at(0), thetaMax);
  term2 += L2(thetas123.at(1), thetas123.at(2), thetas123.at(0), thetas456.at(0), thetas456.at(1), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(1), thetas123.at(2), thetas123.at(0), thetas456.at(1), thetas456.at(0), thetas456.at(2), thetaMax);
  term2 += L2(thetas123.at(1), thetas123.at(2), thetas123.at(0), thetas456.at(2), thetas456.at(1), thetas456.at(0), thetaMax);

  term2/=pow(2*M_PI, 4)*thetaMax*thetaMax;

  return term1+term2;
}