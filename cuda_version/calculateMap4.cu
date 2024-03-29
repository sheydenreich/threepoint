#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "helpers.cuh"
#include "halomodel.cuh"
#include "apertureStatisticsCovariance.cuh"
#include "cuba.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
/**
 * @file calculateMap4.cu
 * This executable calculates <Map⁴> from the 1-halo term of the Trispectrum
 * Aperture radii are read from file and <Map⁴> is only calculated for theta1<=theta2<=theta3<=theta4
 * Code uses CUDA and cubature library  (See
 * https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 */
int main(int argc, char *argv[])
{
  // Read in command line

  const char *message = R"( 
calculateMap4.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas [arcmin]
Argument 3: Outputfilename, directory needs to exist 
Argument 4: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateMap4.x ../necessary_files/MR_cosmo.dat ../necessary_files/HOWLS_thetas.dat ../../results_MR/Map4.dat ../necessary_files/nz_MR.dat
)";

  if (argc < 4) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile, thetasfn, outfn, nzfn;

  cosmo_paramfile = argv[1];
  thetasfn = argv[2];
  outfn = argv[3];
  nzfn = argv[4];

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  // Read in n_z
  std::vector<double> nz;
  read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);
 
  // Check if output file can be opened
  std::ofstream out;
  out.open(outfn.c_str());
  if (!out.is_open())
  {
    std::cerr << "Couldn't open " << outfn << std::endl;
    exit(1);
  };

  // Read in thetas
  std::vector<double> thetas;
  read_thetas(thetasfn, thetas);
  int N = thetas.size();

  // User output
  std::cerr << "Using cosmology from " << cosmo_paramfile << ":" << std::endl;
  std::cerr << cosmo;
  std::cerr << "Using thetas in " << thetasfn << std::endl;
  std::cerr << "Writing to:" << outfn << std::endl;

  int ncores = 0;
  int pcores = 0;
  cubacores(&ncores, &pcores);
  cubaaccel(&ncores, &pcores);

  // Initialize Bispectrum

  copyConstants();

  std::cerr << "Using n(z) from " << nzfn << std::endl;
  set_cosmology(cosmo, &nz);

  initHalomodel();

  //  Set up vector for aperture statistics
  int Ntotal = factorial(N + 3) / factorial(N - 1) / 24; // Total number of bins that need to be calculated, = (N+4+1) ncr 3

  std::vector<double> Map4s;

  // Needed for monitoring

  int step = 0;

  // Calculate <Map⁴>(theta1, theta2, theta3, theta4) in four loops
  // Calculation only for theta1<=theta2<=theta3<=theta4
  for (int i = 0; i < N; i++)
  {
    double theta1 = convert_angle_to_rad(thetas.at(i)); // Conversion to rad

    for (int j = i; j < N; j++)
    {
      double theta2 = convert_angle_to_rad(thetas.at(j));

      for (int k = j; k < N; k++)
      {

        double theta3 = convert_angle_to_rad(thetas.at(k));
        for (int l = k; l < N; l++)
        {

          double theta4 = convert_angle_to_rad(thetas.at(l));
          std::vector<double> thetas_calc = {theta1, theta2, theta3, theta4};
          // Progress for the impatient user (Thetas in arcmin)
          step += 1;
          std::cout << step << "/" << Ntotal << ": Thetas:" << thetas.at(i) << " "
                    << thetas.at(j) << " " << thetas.at(k) << " " << thetas.at(l) << std::endl;

          double Map4_ = Map4(thetas_calc); // Do calculation
          std::cerr << Map4_ << std::endl;

          Map4s.push_back(Map4_);
        };
      };
    };
  };

  // Output
  step = 0;
  for (int i = 0; i < N; i++)
  {
    for (int j = i; j < N; j++)
    {
      for (int k = j; k < N; k++)
      {
        for (int l = k; l < N; l++)
        {
          out << thetas[i] << " " << thetas[j] << " " << thetas[k] << " " << thetas[l] << " "
              << Map4s.at(step) << " " << std::endl;
          step++;
        };
      };
    };
  };

  return 0;
}
