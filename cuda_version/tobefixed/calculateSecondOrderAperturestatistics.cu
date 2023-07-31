#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "helpers.cuh"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
/**
 * @file calculateSecondOrderAperturestatistics.cu
 * This executable calculates <Map²> from the revised Halofit Powerspectrum
 * Aperture radii are read from file
 * Code uses CUDA and cubature library  (See
 * https://github.com/stevengj/cubature for documentation)
 * @author Sven Heydenreich
 */
int main(int argc, char *argv[])
{
  // Read in command line

  const char *message = R"( 
calculateSecondOrderApertureStatistics.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename for covariance parameters (ASCII, see examples/exampleCovariance.param for an example)
Argument 3: Filename with thetas [arcmin]
Argument 4: Outputfilename, directory needs to exist
Argument 5: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateSecondOrderApertureStatistics.x ../necessary_files/MR_cosmo.dat ../necessary_files/HOWLS_thetas.dat ../../results_MR/Map2.dat ../necessary_files/nz_MR.dat
)";

  if (argc < 5) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile, covariance_paramfile, thetasfn, outfn, nzfn;

  cosmo_paramfile = argv[1];
  covariance_paramfile = argv[2];
  thetasfn = argv[3];
  outfn = argv[4];
  nzfn = argv[5];

  covarianceParameters covPar(covariance_paramfile);
  constant_powerspectrum = covPar.shapenoiseOnly;

  if (constant_powerspectrum)
  {
    std::cerr << "WARNING: Uses shape noise only powerspectrum" << std::endl;
  };
  sigma = covPar.shapenoise_sigma;
  n = covPar.galaxy_density;

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  // Read in n_z
  std::vector<double> nz;
  try
  {
    read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return -1;
  };

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
  std::cerr << "Covariance from " << covariance_paramfile << ":" << std::endl;
  std::cerr << covPar;
  if (sigma != 0)
    std::cerr << "Calculating Power Spectrum WITH shapenoise, sigma=" << sigma << std::endl;
  else
    std::cerr << "Calculating Power Spectrum WITHOUT shapenoise" << std::endl;
  std::cerr << "Writing to:" << outfn << std::endl;

  // Initialize Bispectrum

  copyConstants();
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma, &sigma, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n, &n, sizeof(double)));

  std::cerr << "Using n(z) from " << nzfn << std::endl;
  set_cosmology(cosmo, &nz);

  // Set up vector for aperture statistics
  std::vector<double> MapMaps;

  // Needed for monitoring

  int step = 0;

  // Calculate <MapMap>(theta)
  for (int i = 0; i < N; i++)
  {
    double theta = convert_angle_to_rad(thetas.at(i)); // Conversion to rad

    std::cout << step << "/" << N << ": Theta:" << thetas.at(i) << " \r";
    std::cout.flush();

    double Map2_here =
        Map2(theta); // Do calculation
    MapMaps.push_back(Map2_here);
  };

  // Output
  for (int i = 0; i < N; i++)
  {
    out << thetas[i] << " "
        << MapMaps.at(i) << " " << std::endl;
  }

  return 0;
}
