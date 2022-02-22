#include "apertureStatistics.hpp"
#include "helper.hpp"
#include <fstream>

/**
 * @file testApertureStatistics_integral_l1.cpp
 * This executable gives the l1 Integral of Eq. 58 in Schneider, Kilbinger & Lomabrdi (2003)
 * used for debugging
 * @author Laila Linke
 * @warning output is written in file "../tests/TestIntegralL1.dat"
 * @todo Outputfilename should be read from command line
 */

int main()
{
  // Set up cosmology
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

  //Initialize Aperture Statistics
  ApertureStatistics apertureStatistics(&bispectrum);

  //Set output
  std::ofstream out;
  out.open("../tests/TestIntegralL1.dat");

  // Set up thetas
  double theta = convert_angle_to_rad(10.); //10 arcmin in rad
  std::vector<double> thetas = {theta, theta, theta};
  apertureStatistics.lMax = 10. / theta;
  apertureStatistics.lMin = 1e-6;

  // Calculate integral
  out << apertureStatistics.integral_l1(thetas) << std::endl;

  return 0;
}
