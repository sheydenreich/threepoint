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
 * @author Pierre Burger
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

  if (argc < 4) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile = argv[1];
  int zbin1 = std::stoi(argv[2]);
  int zbin2 = std::stoi(argv[3]);
  std::string output_file = argv[4];

  int Ntomo = std::stoi(argv[5]);

  std::vector<std::string> nzfns;
  for (int i = 0; i < Ntomo; i++)
  {
    std::string nzfn = argv[6 + i];
    nzfns.push_back(nzfn);
  }

  std::string shape_noise_file = argv[6 + Ntomo];

  // Check if output file can be opened
  std::ofstream out;
  out.open(output_file.c_str());
  if (!out.is_open())
  {
    std::cerr << "Couldn't open " << output_file << std::endl;
    exit(1);
  };

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  // Read in n_z
  std::vector<std::vector<double>> nzs;
  for (int i = 0; i < Ntomo; i++)
  {
    std::vector<double> nz;
    read_n_of_z(nzfns.at(i), n_redshift_bins, cosmo.zmax, nz);
    nzs.push_back(nz);
  }

  std::vector<double> sigma_epsilon_per_bin;
  std::vector<double> ngal_per_bin;
  read_shapenoise(shape_noise_file, sigma_epsilon_per_bin, ngal_per_bin);

  copyConstants();
  double *dev_g_array, *dev_p_array;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_g_array, Ntomo * n_redshift_bins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_p_array, Ntomo * n_redshift_bins * sizeof(double)));

  double *dev_sigma_epsilon, *dev_ngal;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_sigma_epsilon, Ntomo * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_ngal, Ntomo * sizeof(double)));

  set_cosmology(cosmo, dev_g_array, dev_p_array, &nzs, &sigma_epsilon_per_bin, &ngal_per_bin, dev_sigma_epsilon, dev_ngal);

  for (double ell = 1; ell < 5. * pow(10, 4); ell *= 1.05)
  {

    double spectra = Pell(ell, zbin1, zbin2, dev_g_array, dev_p_array, Ntomo, &sigma_epsilon_per_bin, &ngal_per_bin);
    std::cout << ell << "\t" << spectra << std::endl;
    out << ell << " " << spectra << " " << std::endl;
  }
  out.close();

  cudaFree(dev_g_array);
  cudaFree(dev_p_array);

  return 0;
}
