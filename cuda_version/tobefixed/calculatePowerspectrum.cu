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
 * @file calculatePowerspectrum.cu
 * This executable gives out the Limberintegrated Revised Halofit Powerspectrum
 * The cosmology is read from file
 * @warning ellMin and ellMax are hardcoded
 * @author Sven Heydenreich
 */
int main(int argc, char *argv[])
{
  const char *message = R"( 
calculatePowerspectrum.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Outputfilename, directory needs to exist 
Argument 3: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculatePowerspectrum.x ../necessary_files/MR_cosmo.dat ../../results_MR/powerspectrum_MR.dat ../necessary_files/nz_MR.dat
)";

  // Read in command line
  double lk_min = -5;
  double lk_max = 1;
  const int n_k = 100;
  double *k = new double[n_k];
  double *z = new double[n_k];
  for (int i = 0; i < n_k; i++)
  {
    double k_temp = lk_min + (lk_max - lk_min) * (i + 0.5) / n_k;
    k[i] = pow(10, k_temp);
    z[i] = 1.;
  }
  double *value = new double[n_k];

  if (argc < 4) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile, thetasfn, outfn, nzfn;

  cosmo_paramfile = argv[1];
  outfn = argv[2];
  nzfn = argv[3];

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  sigma = 0;
  n = 1;

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

  // User output
  std::cerr << "Using cosmology from " << cosmo_paramfile << ":" << std::endl;
  std::cerr << cosmo;
  std::cerr << "Writing to:" << outfn << std::endl;

  // Initialize Bispectrum

  copyConstants();

  std::cerr << "Using n(z) from " << nzfn << std::endl;
  set_cosmology(cosmo, &nz);

  for (double ell = 1; ell < 5. * pow(10, 4); ell *= 1.05)
  {
    // double ell = ells[i];
    printf("\b\b\b\b\b\b\b\b\b\b\b\b [%.3e]", ell);
    // Output
    out << ell << " " << Pell(ell) << " " << std::endl;
  }
  out.close();

  return 0;
}
