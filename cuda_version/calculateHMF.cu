#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "helpers.cuh"
#include "halomodel.cuh"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
/**
 * @file calculateHMF.cu
 * This executable gives out the Halo Mass Function used for the Tri- and Pentaspectrum for consistency tests
 * The Halo Mass Function is the Sheth-Tormen (2001) one
 * Cosmology is read from file
 * @author Laila Linke
 */
int main(int argc, char *argv[])
{
  // Read in command line

  const char *message = R"( 
calculateHMF.x : Wrong number of command line parameters (Needed: 4)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Outputfilename, directory needs to exist 
Argument 3: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateHMF.x ../necessary_files/MR_cosmo.dat ../../results_MR/HMF.dat ../necessary_files/nz_MR.dat
)";

  if (argc < 3) // Give out error message if too few CLI arguments
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

  // Read in n_z
  std::vector<double> nz;
  if (nz_from_file)
  {
    read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);
  };

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

  if (nz_from_file)
  {
    std::cerr << "Using n(z) from " << nzfn << std::endl;
    set_cosmology(cosmo, &nz);
  }
  else
  {
    set_cosmology(cosmo);
  };

  initHalomodel();

  double mmin = 10;
  double mmax = 16;
  int Nbins = 100;
  double mbin = (mmax - mmin) / Nbins;
  double z = 1;

  for (int i = 0; i < Nbins; i++)
  {
    double m = pow(10, mmin + (i + 0.5) * mbin);
    double u = hmf(m, z);
    out << m << " " << u << std::endl;
  }

  return 0;
}
