/**
 * @file calculateApertureStatistics.cpp
 * This executable calculates <MapMapMap> for predefined thetas from the
 * Takahashi+ Bispectrum
 * If PARALLEL_INTEGRATION is true, the code is parallelized over the integration point calculation
 * If PARALLEL_RADII is true, the code is parallelized over the aperture radii
 * @author Laila Linke
 * @warning thetas currently hardcoded
 * @warning Output is hardcoded
 * @todo Thetas should be read from command line
 * @todo Outputfilename should be read from command line
 */

#include "apertureStatistics.hpp"
#include "helper.hpp"
#include <fstream>

int main()
{
  // Set Up Cosmology
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

  //Initialize Bispectrum

  int n_z = 400;      //Number of redshift bins for grids
  double z_max = 1.1; //maximal redshift

  bool fastCalc = false; //whether calculations should be sped up
  BispectrumCalculator bispectrum(&cosmo, n_z, z_max, fastCalc);

  //Initialize Aperture Statistics
  ApertureStatistics apertureStatistics(&bispectrum);

  // Set up thetas for which ApertureStatistics are calculated
  std::vector<double> thetas{0.5, 1, 2, 4, 8, 16, 32}; //Thetas in arcmin
  int N = thetas.size();

  // Set up vector for aperture statistics
  std::vector<double> MapMapMaps(N * N * N);

#if CUBATURE
  std::cout << "Using cubature for integration" << std::endl;
#else
  std::cout << "Using GSL for integration" << std::endl;
#endif

#if PARALLEL_RADII
  std::cout << "Parallelized over different aperture radii" << std::endl;
#pragma omp parallel for collapse(3)
  //Calculate <MapMapMap>(theta1, theta2, theta3)
  //This does the calculation only for theta1<=theta2<=theta3, but because of
  //the properties of omp collapse, the for-loops are defined starting from 0
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      for (int k = 0; k < N; k++)
      {
        if (j >= i && k >= j) //Only do calculation for theta1<=theta2<=theta3
        {
          //Thetas are defined here, because auf omp collapse
          double theta1 = convert_angle_to_rad(thetas.at(i)); //Conversion to rad
          double theta2 = convert_angle_to_rad(thetas.at(j));
          double theta3 = convert_angle_to_rad(thetas.at(k));
          double thetas_calc[3] = {theta1, theta2, theta3};

          double MapMapMap = apertureStatistics.MapMapMap(thetas_calc); //Do calculation
          // Do assigment (including permutations)
          MapMapMaps.at(i * N * N + j * N + k) = MapMapMap;
          MapMapMaps.at(i * N * N + k * N + j) = MapMapMap;
          MapMapMaps.at(j * N * N + i * N + k) = MapMapMap;
          MapMapMaps.at(j * N * N + k * N + i) = MapMapMap;
          MapMapMaps.at(k * N * N + i * N + j) = MapMapMap;
          MapMapMaps.at(k * N * N + j * N + i) = MapMapMap;
        };
      };
    };
  };
#else
#if PARALLEL_INTEGRATION
  std::cout << "Parallelization over Integration" << std::endl;
#else
  std::cout << "No Parallelizaion" << std::endl;
#endif
  //Needed for monitoring
  int Ntotal = N * (N + 1) * (N + 2) / 6.; //Total number of bins that need to be calculated, = (N+3+1) ncr 3
  int step = 0;

  //Calculate <MapMapMap>(theta1, theta2, theta3) in three loops
  // Calculation only for theta1<=theta2<=theta3, other combinations are assigned
  for (int i = 0; i < N; i++)
  {
    double theta1 = convert_angle_to_rad(thetas.at(i)); //Conversion to rad

    for (int j = i; j < N; j++)
    {
      double theta2 = convert_angle_to_rad(thetas.at(j));

      for (int k = j; k < N; k++)
      {

        double theta3 = convert_angle_to_rad(thetas.at(k));
        std::vector<double> thetas_calc = {theta1, theta2, theta3};

        //Progress for the impatient user (Thetas in arcmin)
        step += 1;
        std::cout << step << "/" << Ntotal << ": Thetas:" << thetas.at(i) << " " << thetas.at(j) << " " << thetas.at(k) << " \r";
        std::cout.flush();

        double MapMapMap = apertureStatistics.MapMapMap(thetas_calc); //Do calculation

        // Do assigment (including permutations)
        MapMapMaps.at(i * N * N + j * N + k) = MapMapMap;
        MapMapMaps.at(i * N * N + k * N + j) = MapMapMap;
        MapMapMaps.at(j * N * N + i * N + k) = MapMapMap;
        MapMapMaps.at(j * N * N + k * N + i) = MapMapMap;
        MapMapMaps.at(k * N * N + i * N + j) = MapMapMap;
        MapMapMaps.at(k * N * N + j * N + i) = MapMapMap;
      };
    };
  };
#endif

  //Output
  std::string outfn;
  std::ofstream out;

#if test_analytical
  outfn = "../results_analytical/MapMapMap_bispec.dat";
#elif slics
  outfn = "../results_SLICS/MapMapMap_bispec_4d.dat";
#else
  outfn = "../results_MR/MapMapMap_bispec.dat";
#endif
  std::cout << "Writing results to " << outfn << std::endl;
  out.open(outfn.c_str());

  //Print out ==> Should not be parallelized!!!
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      for (int k = 0; k < N; k++)
      {
        out << thetas[i] << " "
            << thetas[j] << " "
            << thetas[k] << " "
            << MapMapMaps.at(k * N * N + i * N + j) << " "
            << std::endl;
      };
    };
  };

  return 0;
}
