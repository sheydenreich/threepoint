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
#include <math.h>

int main(int argc, char *argv[])
{
  // Read in command line

  const char *message = R"( 
calculateTrispectrum_halomodel.x : Wrong number of command line parameters (Needed: 4)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Outputfilename, directory needs to exist 
Argument 3: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateTrispectrum_halomodel.x ../necessary_files/MR_cosmo.dat  ../../results_MR/Trispectrum.dat
)";

  if (argc < 3) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile, thetasfn, outfn, nzfn;

  cosmo_paramfile = argv[1];
  outfn = argv[2];

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);


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

  set_cosmology_noZ(cosmo);

  initHalomodel();

  double mmin=pow(10, logMmin);
  double mmax=pow(10, logMmax);
  double z=0;


  double kmin = log10(1e-1);
  double kmax = log10(1000);
  int Nbins = 20; // 13;//100;
  double kbin = (kmax - kmin) / Nbins;

  for (int i = 0; i < Nbins; i++)
  {
    double k = pow(10, kmin + i * kbin);
    double T = I_40(k, k, k, k, mmin, mmax, z);

    double Delta = pow(T, 0.33)*k*k*k/2/3.1416/3.1416;
          out << k << " "
              << k << " "
              << k << " "
              << k << " "
              << T << " "
              << Delta << std::endl;

  };

  return 0;
}
