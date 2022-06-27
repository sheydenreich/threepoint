#include "apertureStatisticsCovariance.cuh"
#include "cosmology.cuh"
#include "bispectrum.cuh"
#include "helpers.cuh"
#include "cuda_helpers.cuh"
#include "halomodel.cuh"
#include "cuba.h"

#include <iostream>
#include <chrono>

int main(int argc, char *argv[])
{
  // Read in CLI
  const char *message = R"( 
calculateMap2Covariance.x : Wrong number of command line parameters (Needed: 10)
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
  std::string thetasfn = argv[2];
  std::string nzfn = argv[3];
  std::string out_folder = argv[4];
  std::string covariance_paramfile = argv[5];
  bool calculate_Gauss = std::stoi(argv[6]);
  bool calculate_NonGauss = std::stoi(argv[7]);
  std::string type_str = argv[8];

  if(calculate_NonGauss)
  {
    std::cerr << "Non Gauss not implemented"<<std::endl;
    exit(-2);
  };

  std::cerr << "Using cosmology from " << cosmo_paramfile << std::endl;
  std::cerr << "Using thetas from " << thetasfn << std::endl;
  std::cerr << "Using n(z) from " << nzfn << std::endl;
  std::cerr << "Results are written to " << out_folder << std::endl;
  std::cerr << "Using covariance parameters from" << covariance_paramfile << std::endl;

  // Initializations
  covarianceParameters covPar(covariance_paramfile);
  constant_powerspectrum = covPar.shapenoiseOnly;

  if (constant_powerspectrum)
  {
    std::cerr << "WARNING: Uses constant powerspectrum" << std::endl;
  };

  thetaMax = covPar.thetaMax;
  sigma = covPar.shapenoise_sigma;
  n = covPar.galaxy_density;
  lMin = 0; // 2*M_PI/thetaMax;
  thetaMax_smaller=covPar.thetaMax_smaller;
  area=covPar.area;

  cosmology cosmo(cosmo_paramfile);

  std::vector<double> nz;
  try
  {
    read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return -1;
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

  set_cosmology(cosmo, &nz);

  std::vector<double> thetas;

  try
  {
    read_thetas(thetasfn, thetas);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return -1;
  }

  // Initialize Covariance
  initCovariance();

  std::cerr << "Finished copying constants" << std::endl;

  std::cerr << "Using n(z) from " << nzfn << std::endl;

  std::cerr << "Finished initializations" << std::endl;

  // Calculations

  int N = thetas.size();

  std::vector<double> Cov_Gauss, Cov_NonGauss;

  std::vector<double> theta_rad;
  for (int i = 0; i < N; i++)
  {
    double theta1 = convert_angle_to_rad(thetas.at(i)); // Conversion to rad
    theta_rad.push_back(theta1);
  }

  int N_ind = theta_rad.size(); // Number of independent theta-combinations
  int N_total = N_ind*N_ind;

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
          double term1 = Cov_Map2_Gauss(theta_rad.at(i), theta_rad.at(j));
          Cov_Gauss.push_back(term1);
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
              elapsed.count() * 1e-9 / completed_steps, thetas.at(i), thetas.at(j), "arcmin");
    }
  }

  // Output

  char filename[255];
  double n_deg = n / convert_rad_to_angle(1, "deg") / convert_rad_to_angle(1, "deg");
  double thetaMax_deg = convert_rad_to_angle(thetaMax, "deg");

  if (calculate_Gauss)
  {
    sprintf(filename, "covMap2_%s_Gauss_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
            type_str.c_str(), sigma, n_deg, thetaMax_deg);
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
  return 0;
}
