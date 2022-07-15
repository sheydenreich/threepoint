#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "helpers.cuh"
#include "halomodel.cuh"
#include "cuba.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
/**
 * @file calculateMap6.cu
 * This executable calculates <Map⁶> from the 1-halo term of the Pentaspectrum
 * Aperture radii are read from file and <Map⁶> is only calculated for all thetas equal
 * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 */
int main(int argc, char *argv[])
{
  // Read in command line

  const char *message = R"( 
calculateMap6.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas [arcmin]
Argument 3: Outputfilename, directory needs to exist 
Argument 4: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateMap6.x ../necessary_files/MR_cosmo.dat ../necessary_files/HOWLS_thetas.dat ../../results_MR/Map6.dat 1 ../necessary_files/nz_MR.dat
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

  // Calculate <Map⁶>(theta)  and do output
  for (int i = 0; i < N; i++)
  {
    double theta = convert_angle_to_rad(thetas.at(i)); // Conversion to rad
    std::vector<double> thetas_calc = {theta, theta, theta, theta, theta, theta};
    // Progress for the impatient user (Thetas in arcmin)
    std::cout << i << "/" << N << ": Theta:" << thetas.at(i) << std::endl;
    double Map6_ = Map6(thetas_calc); // Do calculation

    out << thetas[i] << " "
        << Map6_ << " " << std::endl;
  };

  return 0;
}
