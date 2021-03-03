#include "apertureStatistics.hpp"
#include "../cubature-1.0.3/cubature.h"
#include <cmath>
#include <omp.h>

double ApertureStatistics::uHat(const double& eta)
{
  double temp=0.5*eta*eta;
  return temp*exp(-temp);
}

double ApertureStatistics::integrand(const double& l1, const double& l2, const double& phi)
{
  double l3=sqrt(l1*l1+l2*l2+2*l1*l2*cos(phi));

  if(!isfinite(l1) || !isfinite(l2) || !isfinite(l3))
    {
      std::cerr<<"ApertureStatistics::integrand: One of the l's not finite"<<std::endl;
      std::cerr<<l1<<" "<<l2<<" "<<l3<<" "<<phi<<std::endl;
      exit(1);
    };

  return l1*l2*Bispectrum_->bkappa(l1, l2, l3)*uHat(l1*theta1_)*uHat(l2*theta2_)*uHat(l3*theta3_);
}


double ApertureStatistics::integrand_phi(double phi, void * thisPtr)
{
  ApertureStatistics* apertureStatistics = (ApertureStatistics*) thisPtr;

  return apertureStatistics->integrand(apertureStatistics->l1_, apertureStatistics->l2_, phi);
}

double ApertureStatistics::integral_phi(double l1, double l2)
{
  l1_=l1;
  l2_=l2;

  //variables for result and integration error
  double result, error;

  //cast kernel to gsl_function
  gsl_function F;
  F.function = &ApertureStatistics::integrand_phi;
  F.params = this;

  //perform integration
  gsl_integration_qags(&F, phiMin, phiMax, 0.0, 1e-4, 1000, w_phi, &result, &error);

  return result;
}


double ApertureStatistics::integrand_l2(double l2, void * thisPtr)
{
  ApertureStatistics* apertureStatistics = (ApertureStatistics*) thisPtr;
  return apertureStatistics->integral_phi(apertureStatistics->l1_, l2);
}


double ApertureStatistics::integral_l2(double l1)
{
  l1_=l1;

  //variables for result and integration error
  double result, error;

  //cast kernel to gsl_function
  gsl_function F;
  F.function = &ApertureStatistics::integrand_l2;
  F.params = this;

  //perform integration
  gsl_integration_qags(&F, lMin, lMax, 0.0, 1e-4, 1000, w_l2, &result, &error);

  return result;
}

double ApertureStatistics::integrand_l1(double l1, void* thisPtr)
{
  ApertureStatistics* apertureStatistics = (ApertureStatistics*) thisPtr;

  return apertureStatistics->integral_l2(l1);
}

double ApertureStatistics::integral_l1()
{

  //variables for result and integration error
  double result, error;
  
  //cast kernel to gsl_function
  gsl_function F;
  F.function = &ApertureStatistics::integrand_l1;
  F.params = this;

  //perform integration
  gsl_integration_qags(&F, lMin, lMax, 0.0, 1e-4, 1000, w_l1, &result, &error);

  return result;
}

int ApertureStatistics::integrand(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
{
  ApertureStatistics* apertureStatistics = (ApertureStatistics*) thisPtr;
  //  std::cout<<"Current Function Evaluations:"<<npts<<std::endl;
  //#pragma omp parallel for num_threads(12) //DONT PARALLELIZE! SOMETHING IS NOT THREAD SAFE!!!
  for( unsigned int i=0; i<npts; i++)
    {
      double ell1=vars[i*ndim];
      double ell2=vars[i*ndim+1];
      double phi=vars[i*ndim+2];
  
      value[i]=apertureStatistics->integrand(ell1, ell2, phi);
    }
  
  return 0; //Success :)
}




ApertureStatistics::ApertureStatistics(BispectrumCalculator* Bispectrum)
{
  std::cout<<"Started initializing Aperture Statistics"<<std::endl;

  // Set bispectrum
  Bispectrum_=Bispectrum;

  // Allocate GSL workspaces
  w_l1=gsl_integration_workspace_alloc(1000);
  w_l2=gsl_integration_workspace_alloc(1000);
  w_phi=gsl_integration_workspace_alloc(1000);
  std::cout<<"Finished initializing Aperture Statistics"<<std::endl;
}

double ApertureStatistics::MapMapMap(const double& theta1, const double& theta2, const double& theta3)
{
  // Set thetas
  theta1_=theta1;
  theta2_=theta2;
  theta3_=theta3;

  //Set maximal l value such, that theta*l <= 10
  double thetaMin=std::min({theta1, theta2, theta3});
  lMax=10./thetaMin;


  double result;

#if CUBATURE //Do cubature integration
  double vals_min[3]={lMin, lMin, phiMin};
  double vals_max[3]={lMax, lMax, phiMax};

  double error;

  hcubature_v(1, integrand, this, 3, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
  
#else //Do standard GSL integration
    result= integral_l1(); 
#endif
  
  return result/248.050213442;//Divided by (2*pi)³
}
