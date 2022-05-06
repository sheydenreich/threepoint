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
 * @file calculateApertureStatistics.cu
 * This executable calculates <MapMapMap> from the
 * Takahashi+ Bispectrum
 * Aperture radii are read from file and <MapMapMap> is only calculated for
 * independent combis of thetas Code uses CUDA and cubature library  (See
 * https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 */
int main(int argc, char *argv[]) {
  // Read in command line

  const char *message = R"( 
calculateApertureStatistics.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas [arcmin]
Argument 3: Outputfilename, directory needs to exist 
Argument 4: 0: use analytic n(z) (only works for MR and SLICS), or 1: use n(z) from file                  
Argument 5 (optional): Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateApertureStatistics.x ../necessary_files/MR_cosmo.dat ../necessary_files/HOWLS_thetas.dat ../../results_MR/MapMapMap_bispec_gpu_nz.dat 1 ../necessary_files/nz_MR.dat
)";

  if (argc < 5) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile, thetasfn, outfn, nzfn;
  bool nz_from_file = false;

  cosmo_paramfile = argv[1];
  outfn = argv[2];
  nz_from_file = std::stoi(argv[3]);
  if (nz_from_file) {
    nzfn = argv[4];
  };

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  // Read in n_z
  std::vector<double> nz;
  if (nz_from_file) {
    read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);
  };

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

  if (nz_from_file) {
    std::cerr << "Using n(z) from " << nzfn << std::endl;
    set_cosmology(cosmo, &nz);
  } 
  else 
  {
    set_cosmology(cosmo);
  };

  initHalomodel();
  
  double lmin=log10(150);
  double lmax=log10(40000);
  int Nbins=13;//100;
  double lbin=(lmax-lmin)/Nbins;

  for( int i=0; i<Nbins; i++)
  {
//    double l=pow(10, lmin+(i+0.5)*lbin);
    double l=pow(10, lmin+i*lbin);
    double T=Trispectrum(l,l,l,l);
    out<<l<<" "<<T<<std::endl;
  }


  return 0;
}
