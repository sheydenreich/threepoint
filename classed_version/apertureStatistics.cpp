#include "apertureStatistics.hpp"

#include <cmath>


double ApertureStatistics::uHat(const double& eta)
{
  double temp=0.5*eta*eta;
  return temp*exp(-temp);
}

double ApertureStatistics::integrand(const double& l1, const double& l2, const double& phi)
{
  double l3=sqrt(l1*l1+l2*l2+2*l1*l2*cos(phi));
  return l1*l2*Bispectrum_->bkappa(l1, l2, l3)*(uHat(l1*theta1_)*uHat(l2*theta2_)*uHat(l3*theta3_)
					       +uHat(l1*theta2_)*uHat(l2*theta3_)*uHat(l3*theta1_)
					       +uHat(l1*theta3_)*uHat(l2*theta1_)*uHat(l3*theta2_));
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
  double thetaMax=std::min({theta1, theta2, theta3});
  lMax=10./thetaMax;


  return integral_l1()/248.050213442; //Divided by (2*pi)³
}
