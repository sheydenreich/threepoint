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

  std::string cosmo_paramfile, outfn, nzfn, config_file;

  cosmo_paramfile = argv[1];
  config_file = argv[2];
  outfn = argv[3];
  nzfn = argv[4];

  std::cout << "Executing " << argv[0] << " ";
  if (argc == 6)
  {
    int deviceNumber = atoi(argv[5]);
    std::cout << "on GPU " << deviceNumber << std::endl;
    cudaSetDevice(deviceNumber);
  }
  else
  {
    std::cout << "on default GPU" << std::endl;
  };

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  std::vector<double> nz;
  read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);

  // Check output file
  std::ofstream out;
  out.open(outfn.c_str());
  if (!out.is_open())
  {
    std::cerr << "Couldn't open " << outfn << std::endl;
    exit(1);
  };

  // User output
  std::cerr << "Using cosmology:" << std::endl;
  std::cerr << cosmo;
  std::cerr << "Writing to:" << outfn << std::endl;

  configGamma config;
  read_gamma_config(config_file, config);
  std::cerr << config;
  // Binning
  int steps = config.rsteps;
  int usteps = config.usteps;
  int vsteps = config.vsteps;

  double rmin = config.rmin;
  double rmax = config.rmax;
  double umin = config.umin;
  double umax = config.umax;
  double vmin = config.vmin;
  double vmax = config.vmax;

  copyConstants();

  set_cosmology(cosmo, &nz);

  compute_weights_bessel();

  // Calculation + Output in one
  double lrmin = log(rmin);
  double lrmax = log(rmax);

  auto begin = std::chrono::high_resolution_clock::now(); // Begin time measurement

  for (int i = 0; i < steps; i++)
  {
    double r = exp(lrmin + (lrmax - lrmin) / steps * (i + 0.5));
    for (int j = 0; j < usteps; j++)
    {
      double u = umin + (umax - umin) / usteps * (j + 0.5);

      for (int k = 0; k < vsteps; k++)
      {
        double v = vmin + (vmax - vmin) / vsteps * (k + 0.5);

        double r2 = r * M_PI / 180. / 60.; // THIS IS THE BINNING BY JARVIS+(2004). FROM THE WEBSITE, NOT THE PAPER.
        double r3 = r2 * u;
        double r1 = v * r3 + r2;

        std::complex<double> _gamma0 = gamma0(r1, r2, r3, z_max);
        std::complex<double> _gamma1 = gamma1(r1, r2, r3, z_max);
        std::complex<double> _gamma2 = gamma2(r1, r2, r3, z_max);
        std::complex<double> _gamma3 = gamma3(r1, r2, r3, z_max);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        int completed_steps = i * usteps * vsteps + j * vsteps + k + 1;
        int total_steps = steps * usteps * vsteps;
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
      };
    };
  };
}
