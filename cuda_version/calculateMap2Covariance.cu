#include "apertureStatisticsCovariance.cuh"
#include "cosmology.cuh"
#include "bispectrum.cuh"
#include "helpers.cuh"
#include "cuda_helpers.cuh"
#include "halomodel.cuh"
//#include "cuba.h"

#include <iostream>
#include <chrono>

/**
 * @file calculateMap2Covariance.cpp
 * This executable calculates the covariance of <Map²> as given by the real-space estimator
 * Calculates Gaussian and Non-Gaussian term independently
 * Aperture radii, cosmology, n(z), and survey properties are read from file
 * Model uses Revised Halofit Powerspectrum, BiHalofit Bispectrum, and 1-halo term for Trispectrum
 * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 */
int main(int argc, char *argv[])
{
  // Read in CLI
  const char *message = R"( 
calculateMap2Covariance.x : Wrong number of command line parameters (Needed: 9)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas [arcmin]
Argument 3: Filename with n(z)
Argument 4: Outputfolder (needs to exist)
Argument 5: Filename for covariance parameters (ASCII, see necessary_files/cov_param for an example)
Argument 6: Calculate Gaussian? (0 or 1)
Argument 7: Calculate Non-Gaussian? (0 or 1)
Argument 8: Survey geometry, either circle, square, infinite, or rectangular
)";

  if (argc != 9)
  {
    std::cerr << message << std::endl;
    exit(-1);
  };

  std::string cosmo_paramfile = argv[1]; // Parameter file
  std::string z_combi_file = argv[2];
  std::string theta_combi_file = argv[3];
  std::string out_folder = argv[4];
  std::string covariance_paramfile = argv[5];
    int Ntomo = std::stoi(argv[6]);
  bool calculate_Gauss = std::stoi(argv[7]);
  bool calculate_NonGauss = std::stoi(argv[8]);
  std::string type_str = argv[9];
    std::vector<std::string> nzfns;
  for (int i = 0; i < Ntomo; i++)
  {
    std::string nzfn = argv[10 + i];
    nzfns.push_back(nzfn);
  }
  std::string shape_noise_file = argv[10 + Ntomo];

  std::cerr << "Using cosmology from " << cosmo_paramfile << std::endl;
  std::cerr << "Results are written to " << out_folder << std::endl;
  std::cerr << "Using covariance parameters from" << covariance_paramfile << std::endl;

  // Initializations
  covarianceParameters covPar(covariance_paramfile);


  thetaMax = covPar.thetaMax;
  lMin = 0;
  thetaMax_smaller = covPar.thetaMax_smaller;
  area = covPar.area;

    std::vector<double> sigma_epsilon_per_bin;
  std::vector<double> ngal_per_bin;
  read_shapenoise(shape_noise_file, sigma_epsilon_per_bin, ngal_per_bin);

  cosmology cosmo(cosmo_paramfile);

  // Read in n_z
  std::vector<std::vector<double>> nzs;
  for (int i = 0; i < Ntomo; i++)
  {
    std::vector<double> nz;
    read_n_of_z(nzfns.at(i), n_redshift_bins, cosmo.zmax, nz);
    nzs.push_back(nz);
  }




  if (type_str == "circle")
  {
    type = 0;
  }
  else if (type_str == "square")
  {
    type = 1;
  }
  else if (type_str == "infinite")
  {
    type = 2;
  }
  else if (type_str == "rectangular")
  {
    type = 3;
  }
  else
  {
    std::cerr << "Cov type not correctly specified" << std::endl;
    exit(-1);
  };


  copyConstants();
  double *dev_g_array, *dev_p_array;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_g_array, Ntomo * n_redshift_bins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_p_array, Ntomo * n_redshift_bins * sizeof(double)));

  double *dev_sigma_epsilon, *dev_ngal;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_sigma_epsilon, Ntomo * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_ngal, Ntomo * sizeof(double)));

  set_cosmology(cosmo, dev_g_array, dev_p_array, &nzs, &sigma_epsilon_per_bin, &ngal_per_bin, dev_sigma_epsilon, dev_ngal);

  double shapenoise[Ntomo];

  for (int i = 0; i < Ntomo; i++)
  {
    shapenoise[i] = 0.5 * sigma_epsilon_per_bin.at(i) * sigma_epsilon_per_bin.at(i) / ngal_per_bin.at(i);
  };
  double *dev_shapenoise;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_shapenoise, Ntomo * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpy(dev_shapenoise, shapenoise, Ntomo * sizeof(double), cudaMemcpyHostToDevice));

  std::vector<std::vector<double>> theta_combis;
  std::vector<std::vector<int>> z_combis;
  int n_combis;
  read_combis(z_combi_file, theta_combi_file, z_combis, theta_combis, n_combis);


  // Initialize Covariance
  initCovariance();

  if (calculate_NonGauss)
  {
    initHalomodel();
  }

  std::cerr << "Finished initializations" << std::endl;

  // Calculations

  int N = theta_combis.size();

  std::vector<double> Cov_Gauss, Cov_NonGauss;

  std::vector<double> theta_rad;
  for (int i = 0; i < N; i++)
  {
    double theta1 = convert_angle_to_rad(theta_combis.at(i).at(0)); // Conversion to rad
    theta_rad.push_back(theta1);
  }

  int N_ind = theta_rad.size(); // Number of independent theta-combinations
  int N_total = N_ind * N_ind;

  int completed_steps = 0;

  auto begin = std::chrono::high_resolution_clock::now(); // Begin time measurement
  for (int i = 0; i < N_ind; i++)
  {
    for (int j = i; j < N_ind; j++)
    {
      try
      {
        if (calculate_Gauss)
        {
          double term1 = Cov_Map2_Gauss(theta_rad.at(i), theta_rad.at(j), z_combis.at(i).at(0), z_combis.at(j).at(0), dev_g_array, Ntomo, dev_shapenoise);
          Cov_Gauss.push_back(term1);
        };
        if (calculate_NonGauss)
        {
          double term1 = Cov_Map2_NonGauss(theta_rad.at(i), theta_rad.at(j), z_combis.at(i).at(0), z_combis.at(j).at(0), dev_g_array, Ntomo);
          Cov_NonGauss.push_back(term1);
          std::cerr<<term1<<std::endl;
        };
      }
      catch (const std::exception &e)
      {
        std::cerr << e.what() << '\n';
        return -1;
      }
      // Progress for the impatient user
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
      completed_steps++;
      double progress = (completed_steps * 1.) / (N_total);

      fprintf(stderr, "\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step. Last thetas: (%.2f, %.2f) [%s]",
              static_cast<int>(progress * 100),
              elapsed.count() * 1e-9 / 3600,
              (N_total - completed_steps) * elapsed.count() * 1e-9 / 3600 / completed_steps,
              elapsed.count() * 1e-9 / completed_steps, convert_rad_to_angle(theta_rad.at(i)), convert_rad_to_angle(theta_rad.at(j)), "arcmin");
    }
  }

  // Output

  char filename[255];
  double thetaMax_deg = convert_rad_to_angle(thetaMax, "deg");

  if (calculate_Gauss)
  {
    sprintf(filename, "covMap2_%s_Gauss_thetaMax_%.2f_gpu.dat",
            type_str.c_str(), thetaMax_deg);
    std::cerr << "Writing Gaussian term to " << out_folder + filename << std::endl;
    try
    {
      writeCov(Cov_Gauss, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      writeCov(Cov_Gauss, N_ind, filename);
    }
  };

  if (calculate_NonGauss)
  {
    sprintf(filename, "covMap2_%s_NonGauss_thetaMax_%.2f_gpu.dat",
            type_str.c_str(), thetaMax_deg);
    std::cerr << "Writing Non-Gaussian term to " << out_folder + filename << std::endl;
    try
    {
      writeCov(Cov_NonGauss, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      writeCov(Cov_Gauss, N_ind, filename);
    }
  };
  return 0;
}
