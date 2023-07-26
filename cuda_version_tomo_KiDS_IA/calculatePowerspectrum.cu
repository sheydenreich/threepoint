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

  std::string nzfn_1 = argv[5];
  std::string nzfn_2 = argv[6];
  std::string nzfn_3 = argv[7];
  std::string nzfn_4 = argv[8];
  std::string nzfn_5 = argv[9];

  std::string shape_noise_file = argv[10];

  std::cerr << output_file << std::endl;
  std::cerr << nzfn_1 << std::endl;
  std::cerr << nzfn_2 << std::endl;
  std::cerr << nzfn_3 << std::endl;
  std::cerr << nzfn_4 << std::endl;
  std::cerr << nzfn_5 << std::endl;

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
  std::vector<double> nz_1;
  read_n_of_z(nzfn_1, n_redshift_bins, cosmo.zmax, nz_1);
  std::vector<double> nz_2;
  read_n_of_z(nzfn_2, n_redshift_bins, cosmo.zmax, nz_2);
  std::vector<double> nz_3;
  read_n_of_z(nzfn_3, n_redshift_bins, cosmo.zmax, nz_3);
  std::vector<double> nz_4;
  read_n_of_z(nzfn_4, n_redshift_bins, cosmo.zmax, nz_4);
  std::vector<double> nz_5;
  read_n_of_z(nzfn_5, n_redshift_bins, cosmo.zmax, nz_5);

  std::vector<std::vector<double>> nz;
  nz.push_back(nz_1);
  nz.push_back(nz_2);
  nz.push_back(nz_3);
  nz.push_back(nz_4);
  nz.push_back(nz_5);

  std::vector<double> sigma_epsilon_per_bin;
  std::vector<double> ngal_per_bin;
  read_shapenoise(shape_noise_file, sigma_epsilon_per_bin, ngal_per_bin);

  set_cosmology(cosmo, &nz, &sigma_epsilon_per_bin, &ngal_per_bin);
 
  // Initialize Bispectrum
  copyConstants();

  for (double ell = 1; ell < 5. * pow(10, 4); ell *= 1.05)
  {

    double spectra = 0.0;
    if(zbin1==zbin2)
    {
      spectra = sigma_epsilon_per_bin[zbin1-1]*sigma_epsilon_per_bin[zbin1-1]/(ngal_per_bin[zbin1-1]/pow(2.9088820866e-4,2));
    }
    spectra += Pell(ell, zbin1, zbin2);
    std::cout << ell << "\t" << spectra << std::endl;
      out << ell << " " << spectra << " " << std::endl;
  }
  out.close();

  return 0;
}
