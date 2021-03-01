/**
 * @file testApertureStatistics_bispectrum.cpp
 * This executable gives the projected bispectrum bkappa(ell1, ell2, ell3)
 * used for debugging
 * @author Laila Linke
 * @warning output is written in file "../tests/TestBispectrum.dat"
 * @todo Outputfilename should be read from command line
 */

#include "apertureStatistics.hpp"
#include <fstream>

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

  //INitialize Aperture Statistics
  ApertureStatistics apertureStatistics(&bispectrum);

  //Ls and phis for which bispectrum is calculated
  std::vector<double> ls{1.00000000e-03, 1.45634848e-03, 2.12095089e-03, 3.08884360e-03,
       4.49843267e-03, 6.55128557e-03, 9.54095476e-03, 1.38949549e-02,
       2.02358965e-02, 2.94705170e-02, 4.29193426e-02, 6.25055193e-02,
       9.10298178e-02, 1.32571137e-01, 1.93069773e-01, 2.81176870e-01,
       4.09491506e-01, 5.96362332e-01, 8.68511374e-01, 1.26485522e+00,
       1.84206997e+00, 2.68269580e+00, 3.90693994e+00, 5.68986603e+00,
       8.28642773e+00, 1.20679264e+01, 1.75751062e+01, 2.55954792e+01,
       3.72759372e+01, 5.42867544e+01, 7.90604321e+01, 1.15139540e+02,
       1.67683294e+02, 2.44205309e+02, 3.55648031e+02, 5.17947468e+02,
       7.54312006e+02, 1.09854114e+03, 1.59985872e+03, 2.32995181e+03,
       3.39322177e+03, 4.94171336e+03, 7.19685673e+03, 1.04811313e+04,
       1.52641797e+04, 2.22299648e+04, 3.23745754e+04, 4.71486636e+04,
       6.86648845e+04, 1.00000000e+05};
  std::vector<double> phis{1.00000000e-04, 6.98209479e-01, 1.39631896e+00,
			   2.09442844e+00,  2.79253791e+00, 3.49064739e+00,
			   4.18875687e+00, 4.88686635e+00, 5.58497583e+00,
			   6.28308531e+00} ;

  // Set output
  std::ofstream out;
  out.open("../tests/TestBispectrum.dat");

  //Set up thetas
  double theta=10./60./180.*2*3.1416; //10 arcmin in rad
  apertureStatistics.theta1_=theta;
  apertureStatistics.theta2_=theta;
  apertureStatistics.theta3_=theta;

  //Calculate bispectrum
  for(unsigned int i=0; i<ls.size(); i++)
    {
      double l1=ls.at(i);
      for(unsigned int j=0; j<ls.size(); j++)
	{
	  double l2=ls.at(j);
	  for(unsigned int k=0; k<phis.size(); k++)
	    {
	      double phi=phis.at(k);
	      double l3=sqrt(l1*l1+l2*l2+2*l1*l2*cos(phi));
	      
	      out<<l1<<" "
		 <<l2<<" "
		 <<phi<<" "
		 <<apertureStatistics.Bispectrum_->bkappa(l1, l2, l3)<<std::endl;
	    };
	  out<<std::endl;
	};
      out<<std::endl;
    };

  return 0;
}
