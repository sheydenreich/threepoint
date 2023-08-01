#include "gamma.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"
#include "helpers.cuh"
#include "cosmology.cuh"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

/**
 * @file calculateGamma.cu
 * This executable calculates Gamma^i
 * for predefined thetas from the Takahashi+ Bispectrum
 * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
 * @author Sven Heydenreich
 */
int main(int argc, char **argv)
{
  // Read in command line
  const char *message = R"( 
calculateGamma.x : Wrong number of command line parameters (Needed: 4)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Config file for the 3pcf
Argument 3: Outputfilename, directory needs to exist 
Argument 4: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)
Argument 5 (optional): GPU device number
Example:
./calculateGamma.x ../necessary_files/MR_cosmo.dat ../../results_MR/MapMapMap_varyingCosmos.dat ../necessary_files/nz_MR.dat 0
)";

  if (argc < 5)
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile = argv[1];
  std::string z_combi_file = argv[2];
  std::string ruv_combi_file = argv[3];
  std::string output_file = argv[4];
  int Ntomo = std::stoi(argv[5]);

  std::vector<std::string> nzfns;
  for (int i = 0; i < Ntomo; i++)
  {
    std::string nzfn = argv[6 + i];
    nzfns.push_back(nzfn);
  }

  // Check output file
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

  std::vector<std::vector<double>> ruv_combis;
  std::vector<std::vector<int>> z_combis;
  int n_combis;
  read_combis(z_combi_file, ruv_combi_file, z_combis, ruv_combis, n_combis);

  copyConstants();
  double *dev_g_array, *dev_p_array;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_g_array, Ntomo * n_redshift_bins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_p_array, Ntomo * n_redshift_bins * sizeof(double)));

  set_cosmology(cosmo, dev_g_array, dev_p_array, &nzs);

  compute_weights_bessel();

  // Calculation + Output in one
  // double lrmin = log(rmin);
  // double lrmax = log(rmax);

  auto begin = std::chrono::high_resolution_clock::now(); // Begin time measurement

  for (int i = 0; i < n_combis; i++)
  {
    double r = ruv_combis[0][i];
    double u = ruv_combis[1][i];
    double v = ruv_combis[2][i];

    int zbin1 = z_combis[0][i];
    int zbin2 = z_combis[1][i];
    int zbin3 = z_combis[2][i];

    double r2 = r * M_PI / 180. / 60.; // THIS IS THE BINNING BY JARVIS+(2004). FROM THE WEBSITE, NOT THE PAPER.
    double r3 = r2 * u;
    double r1 = v * r3 + r2;

    std::complex<double> _gamma0 = gamma0(r1, r2, r3, z_max, zbin1, zbin2, zbin3, dev_g_array, dev_p_array, Ntomo);
    std::complex<double> _gamma1 = gamma1(r1, r2, r3, z_max, zbin1, zbin2, zbin3, dev_g_array, dev_p_array, Ntomo);
    std::complex<double> _gamma2 = gamma2(r1, r2, r3, z_max, zbin1, zbin2, zbin3, dev_g_array, dev_p_array, Ntomo);
    std::complex<double> _gamma3 = gamma3(r1, r2, r3, z_max, zbin1, zbin2, zbin3, dev_g_array, dev_p_array, Ntomo);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    int completed_steps = i;
    int total_steps = n_combis;
    double progress = (completed_steps * 1.) / (total_steps);

    printf("\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step.",
           static_cast<int>(progress * 100),
           elapsed.count() * 1e-9 / 3600,
           (total_steps - completed_steps) * elapsed.count() * 1e-9 / 3600 / completed_steps,
           elapsed.count() * 1e-9 / completed_steps);
    out
        << real(_gamma0) << " "
        << imag(_gamma0) << " "
        << real(_gamma1) << " "
        << imag(_gamma1) << " "
        << real(_gamma2) << " "
        << imag(_gamma2) << " "
        << real(_gamma3) << " "
        << imag(_gamma3) << " "
        << r << " "
        << u << " "
        << v << std::endl;
  }


  cudaFree(dev_g_array);
  cudaFree(dev_p_array);

  return 0;
}
