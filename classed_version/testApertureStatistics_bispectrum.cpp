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

  if (slics)
  {
    printf("using SLICS cosmology...");
    cosmo.h = 0.6898;               // Hubble parameter
    cosmo.sigma8 = 0.826;           // sigma 8
    cosmo.omb = 0.0473;             // Omega baryon
    cosmo.omc = 0.2905 - cosmo.omb; // Omega CDM
    cosmo.ns = 0.969;               // spectral index of linear P(k)
    cosmo.w = -1.0;
    cosmo.om = cosmo.omb + cosmo.omc;
    cosmo.ow = 1 - cosmo.om;
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
    cosmo.om = cosmo.omc + cosmo.omb;
    cosmo.ow = 1. - cosmo.om;
  }

  if (test_analytical)
  {
    std::cout << "Warning: Doing analytical test" << std::endl;
  };

  //Initialize Bispectrum

  int n_z = 200;         //Number of redshift bins for grids
  double z_max = 2;      //maximal redshift
  bool fastCalc = false; //whether calculations should be sped up
  BispectrumCalculator bispectrum(&cosmo, n_z, z_max, fastCalc);

  //INitialize Aperture Statistics
  ApertureStatistics apertureStatistics(&bispectrum);

  //Ls and phis for which bispectrum is calculated
  std::vector<double> ls{1.00000000e+00, 3.59381366e+00, 1.29154967e+01, 4.64158883e+01,
                         1.66810054e+02, 5.99484250e+02, 2.15443469e+03, 7.74263683e+03,
                         2.78255940e+04, 1.00000000e+05};
  std::vector<double> phis{1.00000000e-04, 6.98209479e-01, 1.39631896e+00,
                           2.09442844e+00, 2.79253791e+00, 3.49064739e+00,
                           4.18875687e+00, 4.88686635e+00, 5.58497583e+00,
                           6.28308531e+00};

  // Set output
  std::ofstream out;
  out.open("../tests/TestBispectrum.dat");

  //Calculate bispectrum
  for (unsigned int i = 0; i < ls.size(); i++)
  {
    double l1 = ls.at(i);
    for (unsigned int j = 0; j < ls.size(); j++)
    {
      double l2 = ls.at(j);
      for (unsigned int k = 0; k < phis.size(); k++)
      {
        double phi = phis.at(k);
        double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));

        out << l1 << " "
            << l2 << " "
            << phi << " "
            << apertureStatistics.Bispectrum_->bkappa(l1, l2, l3) << std::endl;
      };
      out << std::endl;
    };
    out << std::endl;
  };

  return 0;
}
