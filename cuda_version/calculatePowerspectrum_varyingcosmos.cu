#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "helpers.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
/**
 * @file testPowerspectrum.cu
 * This executable calculates the nonlinear power spectrum from the
 * Takahashi+ halofit formula
 * @author Sven Heydenreich
 */
int main(int argc, char *argv[]) {
  // Read in command line
  double lk_min =-5;
  double lk_max = 1;
  const int n_k = 100;
  double* k = new double[n_k];
  double* z = new double[n_k];
  for(int i=0;i<n_k;i++)
  {
    double k_temp = lk_min + (lk_max-lk_min)*(i+0.5)/n_k;
    k[i] = pow(10,k_temp);
    z[i] = 1.;
  }
  double* value = new double[n_k];

  const char *message = R"( 
testPowerspectrum.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Outputfilename, directory needs to exist 
Argument 3: Stencil size for derivatives
Argument 4: 0: use analytic n(z) (only works for MR and SLICS), or 1: use n(z) from file                  
Argument 5 (optional): Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./testPowerspectrum.x ../necessary_files/MR_cosmo.dat ../../results_MR/powerspectrum_MR.dat 1 ../necessary_files/nz_MR.dat
)";

  if (argc < 5) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile, thetasfn, outfn, nzfn;
  double stencil_size;
  bool nz_from_file = false;

  cosmo_paramfile = argv[1];
  outfn = argv[2];
  stencil_size = std::atof(argv[3]);
  nz_from_file = std::stoi(argv[4]);
  if (nz_from_file) {
    nzfn = argv[5];
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


  std::vector<cosmology> cosmos;             ///<container for all cosmologies
  std::vector<std::string> cosmo_names;

  cosmology newCosmo = cosmo;
  
  cosmos.push_back(newCosmo);
  cosmo_names.push_back("fid");

  // Set om
  newCosmo = cosmo;
  newCosmo.om = cosmo.om*(1. + stencil_size);
  newCosmo.omc = newCosmo.om - newCosmo.omb;
  newCosmo.ow = 1.-newCosmo.om;
  cosmos.push_back(newCosmo);
  cosmo_names.push_back("om");


  // Set sigma_8
  newCosmo = cosmo;
  newCosmo.sigma8 = cosmo.sigma8*(1. + stencil_size);
  cosmos.push_back(newCosmo);
  cosmo_names.push_back("sig8");

  // Set w
  newCosmo = cosmo;
  newCosmo.w = cosmo.w*(1. + stencil_size);
  cosmos.push_back(newCosmo);
  cosmo_names.push_back("w0");

  // Set h
  newCosmo = cosmo;
  newCosmo.h = cosmo.h*(1. + stencil_size);
  cosmos.push_back(newCosmo);
  cosmo_names.push_back("h");


  // Initialize Bispectrum
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A96, &A96, 48 * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_W96, &W96, 48 * sizeof(double)));

  for(int i=0;i<5;i++)
  {
    if (nz_from_file) {
      std::cerr << "Using n(z) from " << nzfn << std::endl;
      set_cosmology(cosmos[i], &nz);
    } else {
      set_cosmology(cosmos[i]);
    };
    std::cerr << "Working on " << cosmo_names.at(i) << std::endl;
  

    std::string myOutfn = outfn;
    myOutfn.append(cosmo_names.at(i));
    myOutfn.append(".dat");


    std::ofstream out;
    out.open(myOutfn.c_str());
    // double ells[20] = {118.8502227437018, 167.88040181225605, 237.13737056616563, 334.9654391578279, 473.15125896148015, 668.3439175686144, 944.0608762859234, 1333.5214321633246, 1883.6490894898002, 2660.7250597988077, 3758.3740428844467, 5308.844442309884, 7498.942093324547, 10592.537251772896, 14962.356560944321, 21134.890398366497, 29853.826189179592, 42169.65034285816, 59566.21435290098, 84139.51416451944};
    // for(double ell=10;ell<5.*pow(10,4);ell*=1.05)
    double ells[10] = {243.20835813173147, 359.6462168791317, 531.8295896944992, 786.4470671456434, 1162.9646063455602, 1719.742792761936, 2543.0827878332184, 3760.6030930863967, 5561.02054222956, 8223.401594268895};
    for(int i=0;i<10;i++)
    {
      double ell = ells[i];
        printf("\b\b\b\b\b\b\b\b\b\b\b\b [%.3e]",ell);
        // Output
        out << ell << " " << Pell(ell) << " " << std::endl;
    }
    out.close();

  }

  return 0;

}
