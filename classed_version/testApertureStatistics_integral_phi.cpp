#include "apertureStatistics.hpp"
#include <fstream>

/**
 * @file testApertureStatistics_phi.cpp
 * This executable gives the phi Integral of Eq. 58 in Schneider, Kilbinger & Lomabrdi (2003)
 * used for debugging
 * @author Laila Linke
 * @warning output is written in file "../tests/TestIntegralPhi.dat"
 * @todo Outputfilename should be read from command line
 */

int main()
{
  //Set up cosmology
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

  //Ls for which integral is calculated
  std::vector<double> ls{1.00000000e+00, 3.59381366e+00, 1.29154967e+01, 4.64158883e+01,
       1.66810054e+02, 5.99484250e+02, 2.15443469e+03, 7.74263683e+03,
       2.78255940e+04, 1.00000000e+05};

  //Set up output
  std::ofstream out;
  out.open("../tests/TestIntegralPhi.dat");


  // Set up thetas
  double theta=10./60./180.*3.1416; //10 arcmin in rad
  std::vector<double> thetas={theta, theta, theta};

  // Calculate integral and give the result
  for(unsigned int i=0; i<ls.size(); i++)
    {
      double l1=ls.at(i);
      for(unsigned int j=0; j<ls.size(); j++)
	{
	  double l2=ls.at(j);
	  out<<l1<<" "
	     <<l2<<" "
	     <<apertureStatistics.integral_phi(l1, l2, thetas)<<std::endl;

	};
    };

  return 0;
}
