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

/**
 * @file calculateTrispectrum_halomodel.cu
 * This executable gives out the 2D- Trispectrum based on the 1-halo term of the halomodel
 * The cosmology is read from file
 * @warning ellMin and ellMax are hardcoded
 * @author Laila Linke
 */
int main(int argc, char *argv[])
{
  // Read in command line

  const char *message = R"( 
calculateTrispectrum_halomodel.x : Wrong number of command line parameters (Needed: 4)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Outputfilename, directory needs to exist 
Argument 3: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateTrispectrum_halomodel.x ../necessary_files/MR_cosmo.dat  ../../results_MR/Trispectrum.dat ../necessary_files/nz_MR.dat
)";

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

  double lmin = log10(150);
  double lmax = log10(40000);
  int Nbins = 5; // 13;//100;
  double lbin = (lmax - lmin) / Nbins;

  for (int i = 0; i < Nbins; i++)
  {
    double l1 = pow(10, lmin + i * lbin);
    //    double l=pow(10, lmin+(i+0.5)*lbin);
    for (int j = 0; j < Nbins; j++)
    {
      double l2 = pow(10, lmin + j * lbin);
      for (int k = 0; k < Nbins; k++)
      {
        double l3 = pow(10, lmin + k * lbin);
        for (int l = 0; l < Nbins; l++)
        {
          double l4 = pow(10, lmin + l * lbin);
          double T = Trispectrum(l1, l2, l3, l4);
          out << l1 << " "
              << l2 << " "
              << l3 << " "
              << l4 << " "
              << T << std::endl;
        }
      }
    };
  }

  return 0;
}
