#include "apertureStatistics.hpp"
#include "bispectrum.hpp"
#include "cubature.h"
#include "helper.hpp"
#include <cmath>
#include <omp.h>
#include <algorithm>


double ApertureStatistics::uHat(const double& eta)
{
  double temp=0.5*eta*eta;
  return temp*exp(-temp);
}

double ApertureStatistics::integrand(const double& l1, const double& l2, const double& phi, std::vector<double> thetas)
{
  double l3=sqrt(l1*l1+l2*l2+2*l1*l2*cos(phi));

  if(!isfinite(l1) || !isfinite(l2) || !isfinite(l3))
    {
      std::cerr<<"ApertureStatistics::integrand: One of the l's not finite"<<std::endl;
      std::cerr<<l1<<" "<<l2<<" "<<l3<<" "<<phi<<std::endl;
      exit(1);
    };

  return l1*l2*Bispectrum_->bkappa(l1, l2, l3)*uHat(l1*thetas[0])*uHat(l2*thetas[1])*uHat(l3*thetas[2]);
}

double ApertureStatistics::integrand_4d(const double& l1, const double& l2, const double& phi, const double& z, std::vector<double> thetas)
{
  double l3=sqrt(l1*l1+l2*l2+2*l1*l2*cos(phi));

  if(!isfinite(l1) || !isfinite(l2) || !isfinite(l3))
    {
      std::cerr<<"ApertureStatistics::integrand: One of the l's not finite"<<std::endl;
      std::cerr<<l1<<" "<<l2<<" "<<l3<<" "<<phi<<std::endl;
      exit(1);
    };

  struct ell_params ells;
  ells.ell1 = l1;
  ells.ell2 = l2;
  ells.ell3 = l3;

  return l1*l2*Bispectrum_->integrand_bkappa(z,ells)*uHat(l1*thetas[0])*uHat(l2*thetas[1])*uHat(l3*thetas[2]);
}

#if DO_CYCLIC_PERMUTATIONS
  double ApertureStatistics::_uHat_product(const double& l1, const double& l2, const double& l3, double* thetas)
  {
    return uHat(l1*thetas[0])*uHat(l2*thetas[1])*uHat(l3*thetas[2]);
  }
  double ApertureStatistics::uHat_product(const double& l1, const double& l2, const double& l3, double* thetas)
  {
    std::cout << "test" << std::endl;
    return (_uHat_product(l1, l2, l3, thetas)+_uHat_product(l2, l3, l1, thetas)+_uHat_product(l3, l1, l2, thetas))/3.;
  }
#else
  double ApertureStatistics::uHat_product(const double& l1, const double& l2, const double& l3, std::vector<double> thetas)
  {
    return uHat(l1*thetas[0])*uHat(l2*thetas[1])*uHat(l3*thetas[2]);
  }
#endif

double ApertureStatistics::uHat_product_permutations(const double& l1, const double& l2, const double& l3, std::vector<double> thetas)
{
  double _result;
  _result = uHat_product(l1,l2,l3,thetas);
  _result += uHat_product(l2,l3,l1,thetas);
  _result += uHat_product(l3,l1,l2,thetas);
  _result += uHat_product(l1,l3,l2,thetas);
  _result += uHat_product(l3,l2,l1,thetas);
  _result += uHat_product(l2,l1,l3,thetas);
  return _result;
}

double ApertureStatistics::integrand_Gaussian_Aperture_Covariance(const double& l1, const double& l2, const double& phi, const double& z, 
                                                          std::vector<double> thetas_123, std::vector<double> thetas_456)
{
  double l3 = sqrt(l1*l1+l2*l2+2*l1*l2*cos(phi));
  double result = uHat_product(l1, l2, l3, thetas_123);
  result *= uHat_product_permutations(l1, l2, l3, thetas_456);
#if CONSTANT_POWERSPECTRUM
#elif ANALYTICAL_POWERSPECTRUM
  double P1=p1*l1*l1*exp(-p2*l1*l1);
  double P2=p1*l2*l2*exp(-p2*l2*l2);
  double P3=p1*l3*l3*exp(-p2*l3*l3);

  result *= P1*P2*P3;
#elif ANALYTICAL_POWERSPECTRUM_V2
  double P1=p1*l1*exp(-p2*l1);
  double P2=p1*l2*exp(-p2*l2);
  double P3=p1*l3*exp(-p2*l3);
  result *= P1*P2*P3;
#else
  result *= Bispectrum_->limber_integrand_triple_power_spectrum(l1,l2,l3,z);
#endif
  result *= l1*l2;

  return result;
}

int ApertureStatistics::integrand_Gaussian_Aperture_Covariance(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim != 1)
    {
      std::cerr<<"ApertureStatistics::integrand_Aperture_Covariance: Wrong number of function dimensions"<<std::endl;
      exit(1);
    };
#if CONSTANT_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM_V2
    if(ndim != 3)
    {
      std::cerr<<"ApertureStatistics::integrand_Aperture_Covariance: Wrong number of variable dimensions"<<std::endl;
      exit(1);
    };
#else
  if(ndim != 4)
    {
      std::cerr<<"ApertureStatistics::integrand_Aperture_Covariance: Wrong number of variable dimensions"<<std::endl;
      exit(1);
    };
#endif
  
  ApertureStatisticsContainer* container = (ApertureStatisticsContainer*) thisPtr;

  ApertureStatistics* apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas_123 = container->thetas;
  std::vector<double> thetas_456 = container->thetas2;

  if(npts>1e8)
  std::cout << "Npts: " << npts << " at thetas " << 
   convert_rad_to_angle(thetas_123[0]) << ", " <<
   convert_rad_to_angle(thetas_123[1]) << ", " <<
   convert_rad_to_angle(thetas_123[2]) << ", " <<
   convert_rad_to_angle(thetas_456[0]) << ", " <<
   convert_rad_to_angle(thetas_456[1]) << ", " <<
   convert_rad_to_angle(thetas_456[2]) << ", " <<
   std::endl;

#if PARALLEL_INTEGRATION
#pragma omp parallel for
#endif
  for( unsigned int i=0; i<npts; i++)
    {
      double ell1=vars[i*ndim];
      double ell2=vars[i*ndim+1];
      double phi=vars[i*ndim+2];
#if CONSTANT_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM_V2
      double z=0;
#else
      double z=vars[i*ndim+3];
#endif
  
      value[i]=apertureStatistics->integrand_Gaussian_Aperture_Covariance(ell1, ell2, phi, z, thetas_123, thetas_456);
    }
  return 0; //Success :)
}



double ApertureStatistics::integrand_phi(double phi, void * thisPtr)
{
  ApertureStatisticsContainer* container = (ApertureStatisticsContainer*) thisPtr;

  ApertureStatistics* apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas = container->thetas;
  
  return apertureStatistics->integrand(apertureStatistics->l1_, apertureStatistics->l2_, phi, thetas);
}

double ApertureStatistics::integral_phi(double l1, double l2, std::vector<double> thetas)
{
  l1_=l1;
  l2_=l2;

  //variables for result and integration error
  double result, error;

  //cast kernel to gsl_function
  gsl_function F;
  F.function = &ApertureStatistics::integrand_phi;
  ApertureStatisticsContainer container;
  container.aperturestatistics=this;
  container.thetas=thetas;
  F.params = &container;

  //perform integration
  gsl_integration_qags(&F, phiMin, phiMax, 0.0, 1e-4, 1000, w_phi, &result, &error);

  return result;
}


double ApertureStatistics::integrand_l2(double l2, void * thisPtr)
{
  ApertureStatisticsContainer* container = (ApertureStatisticsContainer*) thisPtr;

  ApertureStatistics* apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas = container->thetas;
  
  return apertureStatistics->integral_phi(apertureStatistics->l1_, l2, thetas);
}


double ApertureStatistics::integral_l2(double l1, std::vector<double> thetas)
{
  l1_=l1;

  //variables for result and integration error
  double result, error;

  //cast kernel to gsl_function
  gsl_function F;
  F.function = &ApertureStatistics::integrand_l2;
  ApertureStatisticsContainer container;
  container.aperturestatistics=this;
  container.thetas=thetas;
  F.params = &container;

  //perform integration
  gsl_integration_qags(&F, lMin, lMax, 0.0, 1e-4, 1000, w_l2, &result, &error);

  return result;
}

double ApertureStatistics::integrand_l1(double l1, void* thisPtr)
{
   ApertureStatisticsContainer* container = (ApertureStatisticsContainer*) thisPtr;

  ApertureStatistics* apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas = container->thetas;

  return apertureStatistics->integral_l2(l1, thetas);
}

double ApertureStatistics::integral_l1(std::vector<double> thetas)
{

  //variables for result and integration error
  double result, error;
  
  //cast kernel to gsl_function
  gsl_function F;
  F.function = &ApertureStatistics::integrand_l1;
  ApertureStatisticsContainer container;
  container.aperturestatistics=this;
  container.thetas=thetas;
  F.params = &container;

  //perform integration
  gsl_integration_qags(&F, lMin, lMax, 0.0, 1e-4, 1000, w_l1, &result, &error);

  return result;
}

int ApertureStatistics::integrand(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim != 1)
    {
      std::cerr<<"ApertureStatistics::integrand: Wrong number of function dimensions"<<std::endl;
      exit(1);
    };
  ApertureStatisticsContainer* container = (ApertureStatisticsContainer*) thisPtr;

  ApertureStatistics* apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas = container->thetas;

#if PARALLEL_INTEGRATION
#pragma omp parallel for
#endif
  for( unsigned int i=0; i<npts; i++)
    {
      double ell1=vars[i*ndim];
      double ell2=vars[i*ndim+1];
      double phi=vars[i*ndim+2];
  
      value[i]=apertureStatistics->integrand(ell1, ell2, phi, thetas);
    }
  
  return 0; //Success :)
}

int ApertureStatistics::integrand_4d(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim != 1)
    {
      std::cerr<<"ApertureStatistics::integrand: Wrong number of function dimensions"<<std::endl;
      exit(1);
    };
  ApertureStatisticsContainer* container = (ApertureStatisticsContainer*) thisPtr;

  ApertureStatistics* apertureStatistics = container->aperturestatistics;
  std::vector<double> thetas = container->thetas;

#if PARALLEL_INTEGRATION
#pragma omp parallel for
#endif
  for( unsigned int i=0; i<npts; i++)
    {
      double ell1=vars[i*ndim];
      double ell2=vars[i*ndim+1];
      double phi=vars[i*ndim+2];
      double z=vars[i*ndim+3];
  
      value[i]=apertureStatistics->integrand_4d(ell1, ell2, phi, z, thetas);
    }

  
  return 0; //Success :)
}




ApertureStatistics::ApertureStatistics(BispectrumCalculator* Bispectrum)
{

  // Set bispectrum
  Bispectrum_=Bispectrum;

  // Allocate GSL workspaces
  w_l1=gsl_integration_workspace_alloc(1000);
  w_l2=gsl_integration_workspace_alloc(1000);
  w_phi=gsl_integration_workspace_alloc(1000);
}



double ApertureStatistics::MapMapMap(const std::vector<double>& thetas)
{

  //Set maximal l value such, that theta*l <= 10
  double thetaMin=*std::min_element(std::begin(thetas), std::end(thetas));
  lMax=10./thetaMin;


  ApertureStatisticsContainer container;
  container.aperturestatistics=this;
  container.thetas=thetas;
  double result,error;

#if CUBATURE //Do cubature integration

  #if INTEGRATE4D //do limber integration via cubature
    double vals_min[4]={lMin, lMin, phiMin, 0};
    double vals_max[4]={lMax, lMax, phiMax, Bispectrum_->z_max};

    pcubature_v(1, integrand_4d, &container, 4, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);
    result *= 27./8.*pow(Bispectrum_->cosmo->om,3)*pow(100./299792.,5); //account for prefactor of limber integration

  #else //do limber integration separately
    double vals_min[3]={lMin, lMin, phiMin};
    double vals_max[3]={lMax, lMax, phiMax};

    hcubature_v(1, integrand, &container, 3, vals_min, vals_max, 0, 0, 1e-6, ERROR_L1, &result, &error);
  #endif
#else //Do standard GSL integration
    result= integral_l1(); 
#endif
  
  return result/8/M_PI/M_PI/M_PI;//Divided by (2*pi)³
}

double ApertureStatistics::MapMapMap_covariance_Gauss(const std::vector<double>& thetas_123, const std::vector<double>& thetas_456, double survey_area)
{

  //Set maximal l value such, that theta*l <= 10
  double thetaMin_123=*std::min_element(std::begin(thetas_123), std::end(thetas_123));
  double thetaMin_456=*std::min_element(std::begin(thetas_456), std::end(thetas_456));
  double thetaMin=std::max({thetaMin_123,thetaMin_456}); //should increase runtime, if either theta_123 or theta_456 is zero, so is their product
  lMax=10./thetaMin;


  ApertureStatisticsContainer container;
  container.aperturestatistics=this;
  container.thetas=thetas_123;
  container.thetas2=thetas_456;
  double result,error;

#if CUBATURE //Do cubature integration

#if INTEGRATE4D //do limber integration via cubature
#if CONSTANT_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM_V2
  double vals_min[3]={lMin, lMin, phiMin};
  double vals_max[3]={lMax, lMax, phiMax/2.}; //use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

  pcubature_v(1, integrand_Gaussian_Aperture_Covariance, &container, 3, vals_min, vals_max, 0, 0, 1e-6, ERROR_L1, &result, &error);

#else
  double vals_min[4]={lMin, lMin, phiMin, 0};
  double vals_max[4]={lMax, lMax, phiMax/2., Bispectrum_->z_max}; //use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

  pcubature_v(1, integrand_Gaussian_Aperture_Covariance, &container, 4, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);
#endif
  
#else //do limber integration separately
  std::cerr << "MapMapMap_covariance_Gauss: 3-dimensional cubature integration not implemented!" << std::endl;
  exit(-1);
#endif
#else //Do standard GSL integration
  std::cerr << "MapMapMap_covariance_Gauss: GSL integration not implemented!" << std::endl;
  exit(-1);
#endif
  
  return 2.*result/survey_area/8/M_PI/M_PI/M_PI;
}
