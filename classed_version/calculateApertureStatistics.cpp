/**
 * @file calculateApertureStatistics.cpp
 * This executable calculates <MapMapMap> for predefined thetas from the
 * Takahashi+ Bispectrum
 * @author Laila Linke
 * @warning thetas currently hardcoded
 * @warning output is written in file "../tests/TestMapMapMap.dat"
 * @todo Thetas should be read from command line
 * @todo Outputfilename should be read from command line
 * @todo Parallelize calculation of MapMapMap for different points
 */


#include "apertureStatistics.hpp"
#include <fstream>


int main()
{
  // Set Up Cosmology
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


  //Initialize Bispectrum
  
  int n_z=200; //Number of redshift bins for grids
  double z_max=2; //maximal redshift
  bool fastCalc=false; //whether calculations should be sped up
  BispectrumCalculator bispectrum(cosmo, n_z, z_max, fastCalc);

  //Initialize Aperture Statistics
  
  ApertureStatistics apertureStatistics(&bispectrum);

  // Set up thetas for which ApertureStatistics are calculated
  std::vector<double> thetas{0.5, 1, 2, 4, 8, 16, 32}; //Thetas in arcmin

  //Set up output
  std::ofstream out;


#if CUBATURE
  std::cout<<"Using cubature for integration"<<std::endl;
  out.open("../tests/TestMapMapMapCubature.dat");
#else
  std::cout<<"Using GSL for integration"<<std::endl;
  out.open("../tests/TestMapMapMapGSL.dat");
 #endif
  //Calculate <MapMapMap>(theta1, theta2, theta3) and output
  //This could be parallelized (But take care of output!)
    for (unsigned int i=0; i<thetas.size(); i++)
    {
      double theta1=thetas[i]*3.1416/180./60; //Conversion to rad
      
      for (unsigned int j=0; j<thetas.size(); j++)
	{
	  double theta2=thetas[j]*3.1416/180./60.;

	  for(unsigned int k=0; k<thetas.size(); k++)
	    {
	      double theta3=thetas[k]*3.1416/180./60.;
	      std::cout<<"Calculating MapMapMap for "<<thetas[i]<<" "<<thetas[j]<<" "<<thetas[k]<<std::endl;
	      out<<thetas[i]<<" "
		 <<thetas[j]<<" "
		 <<thetas[k]<<" "
		 <<apertureStatistics.MapMapMap(theta1, theta2, theta3)
		 <<std::endl;	      
	    };
	};
    };
  



  return 0;
}
