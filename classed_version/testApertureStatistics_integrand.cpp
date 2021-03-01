#include "apertureStatistics.hpp"
#include <fstream>

/**
 * @file testApertureStatistics_integrand.cpp
 * This executable gives the integrand of Eq. 58 in Schneider, Kilbinger & Lomabrdi (2003)
 * used for debugging
 * @author Laila Linke
 * @warning output is written in file "../tests/TestIntegrand.dat"
 * @todo Outputfilename should be read from command line
 */

int main()
{
  // Set up Cosmology
  struct cosmology cosmo;

  if(slics)
    {
      printf("using SLICS cosmology...");
      cosmo.h=0.6898;     // Hubble parameter
      cosmo.sigma8=0.826; // sigma 8
      cosmo.omb=0.0473;   // Omega baryon
      cosmo.omc=0.2905-cosmo.omb;   // Omega CDM
      cosmo.ns=0.969;    // spectral index of linear P(k)
      cosmo.w=-1.0;
      cosmo.om = cosmo.omb+cosmo.omc;
      cosmo.ow = 1-cosmo.om;
    }
  else
    {
      printf("using Millennium cosmology...");
      cosmo.h = 0.73;
      cosmo.sigma8 = 0.9;
      cosmo.omb = 0.045;
      cosmo.omc = 0.25 - cosmo.omb;
      cosmo.ns = 1.;
      cosmo.w = -1.0;
      cosmo.om = cosmo.omc+cosmo.omb;
      cosmo.ow = 1.-cosmo.om;
    }

  if(test_analytical)
    {
      std::cout<<"Warning: Doing analytical test"<<std::endl;
    };

  //Initialize Bispectrum
  
  int n_z=200; //Number of redshift bins for grids
  double z_max=2; //maximal redshift
  bool fastCalc=false; //whether calculations should be sped up
  BispectrumCalculator bispectrum(cosmo, n_z, z_max, fastCalc);

  //Initialize Aperture Statistics
  
  ApertureStatistics apertureStatistics(&bispectrum);

  //Ls and phis for which integrand is calculated
  
  std::vector<double> ls{1.00000000e-03, 5.99484250e-03, 3.59381366e-02,
			 2.15443469e-01, 1.29154967e+00, 7.74263683e+00,
			 4.64158883e+01, 2.78255940e+02, 1.66810054e+03,
			 1.00000000e+04};
  std::vector<double> phis{1.00000000e-04, 6.98209479e-01, 1.39631896e+00,
			   2.09442844e+00,  2.79253791e+00, 3.49064739e+00,
			   4.18875687e+00, 4.88686635e+00, 5.58497583e+00,
			   6.28308531e+00} ; //<[rad]

  // Set output
  std::ofstream out;
  out.open("../tests/TestIntegrand.dat");

  //Set up thetas
  double theta=10./60./180.*2*3.1416; //10 arcmin in rad
  apertureStatistics.theta1_=theta;
  apertureStatistics.theta2_=theta;
  apertureStatistics.theta3_=theta;

  //Calculate integrand and print result
  for(unsigned int i=0; i<ls.size(); i++)
    {
      double l1=ls.at(i);
      for(unsigned int j=0; j<ls.size(); j++)
	{
	  double l2=ls.at(j);
	  for(unsigned int k=0; k<phis.size(); k++)
	    {
	      double phi=phis.at(k);
	      out<<l1<<" "
		 <<l2<<" "
		 <<phi<<" "
		 <<apertureStatistics.integrand(l1, l2, phi)<<std::endl;
	    };
	};
    };

  return 0;
}
