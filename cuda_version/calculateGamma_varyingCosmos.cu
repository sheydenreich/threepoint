#include "gamma.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"
#include "helpers.cuh"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono> //For time measurements
#include <cstdlib>

/**
 * @file calculateGamma_varyingCosmos.cu
 * This executable calculates Gamma^i for variations of
 * the cosmological parameters \f$h$\f, \f$\sigma_8$\f, \f$\Omega_b$\f,
 * \f$n_s$\f, \f$w$\f, \f$\Omega_m$\f, and \f$\Omega_\Lambda$\f
 * for predefined thetas from the Takahashi+ Bispectrum
 * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
 * @author Sven Heydenreich
 * @warning Currently only equilateral triangles
 * @warning thetas currently hardcoded
 * @warning Main cosmology is hardcoded (Set to either MS)
 * @todo Thetas should be read from command line
 * @todo cosmology should be read from init-file
 */

int main(int argc, char **argv)
{
  std::cout << "Executing " << argv[0] << " ";
  if (argc >= 2)
  {
    int deviceNumber = atoi(argv[1]);
    std::cout << "on GPU " << deviceNumber << std::endl;
    cudaSetDevice(deviceNumber);
  }
  else
  {
    std::cout << "on default GPU ";
  }

  std::string cosmo_paramfile, outfn, nzfn;
  bool nz_from_file = false;

  if (slics)
  {
    // Set Up Cosmology
    cosmo_paramfile = "SLICS_cosmo.dat";
    // Set output file
    outfn = "../../results_SLICS/Gammas_varyingCosmos.dat";
    // Set n_z_file
    nzfn = "nz_SLICS_euclidlike.dat";
    nz_from_file = true;
  }
  else
  {
    // Set Up Cosmology
    cosmo_paramfile = "MR_cosmo.dat";
    // Set output file
    outfn = "../../results_MR/Gammas_varyingCosmos.dat";
    // Set n_z_file
    nzfn = "nz_MR.dat";
    nz_from_file = true;
  };

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  std::vector<double> nz;
  if (nz_from_file)
  {
    // Read in n_z
    read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);
  };

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

  copyConstants();

  // Set up thetas for which ApertureStatistics are calculated
  std::vector<double> meanr{0.5, 2, 16};
  std::vector<double> meanu{0.05, 0.5, 0.95};
  std::vector<double> meanv{0.05, 0.5, 0.95};

  int N_r = meanr.size();
  int N_u = meanu.size();
  int N_v = meanv.size();

  // Set up cosmologies at which the 3pcf are
  // This can probably be done smarter
  // Sets each parameter to N_cosmo values between fac_min*Main Value and fac_max*Main Value
  int N_cosmo = 10;       // Number of variations for each parameter
  double fac_min = 0.995; // Minimum proportion of main value for each parameter
  double fac_max = 1.005; // Maximum proportion of main value for each parameter
  double fac_bin = (fac_max - fac_min) / N_cosmo;

  std::vector<cosmology> cosmos(N_cosmo * 7); ///< container for all cosmologies
  for (int i = 0; i < N_cosmo; i++)
  {
    double fac = fac_min + i * fac_bin;
    cosmology newCosmo = cosmo;
    newCosmo.h = cosmo.h * fac;
    cosmos.at(i) = newCosmo;

    newCosmo = cosmo;
    newCosmo.sigma8 = cosmo.sigma8 * fac;
    cosmos.at(i + N_cosmo) = newCosmo;

    newCosmo = cosmo;
    newCosmo.omb = cosmo.omb * fac;
    newCosmo.omc = newCosmo.om - newCosmo.omb;
    cosmos.at(i + 2 * N_cosmo) = newCosmo;

    newCosmo = cosmo;
    newCosmo.ns = cosmo.ns * fac;
    cosmos.at(i + 3 * N_cosmo) = newCosmo;

    newCosmo = cosmo;
    newCosmo.w = cosmo.w * fac;
    cosmos.at(i + 4 * N_cosmo) = newCosmo;

    newCosmo = cosmo;
    newCosmo.om = cosmo.om * fac;
    newCosmo.omc = newCosmo.om - newCosmo.omb;
    cosmos.at(i + 5 * N_cosmo) = newCosmo;

    newCosmo = cosmo;
    newCosmo.ow = cosmo.ow * fac;
    cosmos.at(i + 6 * N_cosmo) = newCosmo;
  }

  compute_weights_bessel();

  for (int i = 0; i < N_cosmo * 7; i++)
  {
    std::cout << "Doing calculations for cosmology " << i + 1 << " of " << N_cosmo * 7 << std::endl;
    auto begin = std::chrono::high_resolution_clock::now(); // Begin time measurement
    // Initialize Bispectrum
    if (nz_from_file)
    {
      set_cosmology(cosmos[i], &nz);
    }
    else
    {
      set_cosmology(cosmos[i]);
    }

    // Needed for monitoring
    int Ntotal = N_r * N_u * N_v; // Total number of bins that need to be calculated
    int step = 0;

    out << cosmos[i].h << " " << cosmos[i].sigma8 << " " << cosmos[i].omb << " " << cosmos[i].ns << " " << cosmos[i].w << " " << cosmos[i].om << " " << cosmos[i].ow << " ";

    // Calculate <MapMapMap>(theta1, theta1, theta1)
    //  Calculation only for theta1=theta2=theta3
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    for (int r_ind = 0; r_ind < N_r; r_ind++)
    {
      for (int u_ind = 0; u_ind < N_u; u_ind++)
      {
        for (int v_ind = 0; v_ind < N_v; v_ind++)
        {
          double r = meanr.at(r_ind);
          double u = meanu.at(u_ind);
          double v = meanv.at(v_ind);

          double r2 = r * M_PI / 180. / 60.;
          double r3 = r2 * u;
          double r1 = v * r3 + r2;

          std::complex<double> _gamma0 = gamma0(r1, r2, r3, z_max);
          std::complex<double> _gamma1 = gamma1(r1, r2, r3, z_max);
          std::complex<double> _gamma2 = gamma2(r1, r2, r3, z_max);
          std::complex<double> _gamma3 = gamma3(r1, r2, r3, z_max);

          out << r << " " << u << " " << v << " ";
          out << real(_gamma0) << " " << imag(_gamma0) << " ";
          out << real(_gamma1) << " " << imag(_gamma1) << " ";
          out << real(_gamma2) << " " << imag(_gamma2) << " ";
          out << real(_gamma3) << " " << imag(_gamma3) << " ";

          step++;
          end = std::chrono::high_resolution_clock::now();
          elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
          double progress = step * 1. / Ntotal;

          printf("\r [%3d%%] in %.2f s. Est. remaining: %.2f s. Average: %.2f s per step.",
                 static_cast<int>(progress * 100),
                 elapsed.count() * 1e-9,
                 (Ntotal - step) * elapsed.count() * 1e-9 / step,
                 elapsed.count() * 1e-9 / step);
        }
      }
    }
    out << std::endl;
    std::cout << std::endl;
    std::cout << "Finished cosmology " << i + 1;
    printf(" after %.2f h. Est. remaining: %.2f h. Average: %.2f min per cosmology. \n",
           elapsed.count() * 1e-9 / 3600,
           (7 * N_cosmo - i - 1) * elapsed.count() * 1e-9 / 3600 / (i + 1),
           elapsed.count() * 1e-9 / 60 / (i + 1));
  };
}
