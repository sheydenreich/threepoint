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
 * @file calculatePowerspectrum_halomodel.cu
 * This executable gives out the 2D- Powerspectrum based on the 1-halo term of the halomodel
 * The cosmology is read from file
 * @warning ellMin and ellMax are hardcoded
 * @author Laila Linke
 */
int main(int argc, char *argv[]) {
  // Read in command line

  const char *message = R"( 
calculatePowerspectrum_halomodel.x : Wrong number of command line parameters (Needed: 4)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Outputfilename, directory needs to exist 
Argument 3: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculatePowerspectrum_halomodel.x ../necessary_files/MR_cosmo.dat  ../../results_MR/Powerspectrum.dat ../necessary_files/nz_MR.dat
)";

  if (argc < 4) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile, thetasfn, outfn, nzfn;

  
  cosmo_paramfile = argv[1];
  outfn = argv[2];
  nzfn = argv[4];

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  // Read in n_z
  std::vector<double> nz;
  read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);

  // Check if output file can be opened
  std::ofstream out;
  out.open(outfn.c_str());
  if (!out.is_open()) {
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

  initHalomodel();
  
  double lmin=1;
  double lmax=5;
  int Nbins=100;
  double lbin=(lmax-lmin)/Nbins;

  for( int i=0; i<Nbins; i++)
  {
    double l=pow(10, lmin+(i+0.5)*lbin);
    double P=Powerspectrum(l);
    out<<l<<" "<<P<<std::endl;
  }


  return 0;
}
