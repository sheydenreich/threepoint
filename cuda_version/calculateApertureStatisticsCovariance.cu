#include "apertureStatisticsCovariance.cuh"
#include "cosmology.cuh"
#include "bispectrum.cuh"
#include "helpers.cuh"
#include "cuda_helpers.cuh"
#include "halomodel.cuh"


#include <iostream>
#include <chrono>

/**
 * @file calculateApertureStatisticsCovariance.cpp
 * This executable calculates the covariance of <MapMapMap> as given by the real-space estimator
 * Calculates Terms T1-T7 individually
 * Aperture radii, cosmology, n(z), and survey properties are read from file
 * Model uses Revised Halofit Powerspectrum, BiHalofit Bispectrum, and 1-halo terms for Tri- and Pentaspectrum
 * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 */
int main(int argc, char *argv[])
{
  // Read in CLI
  const char *message = R"( 
calculateApertureStatisticsCovariance.x : Wrong number of command line parameters (Needed: 10)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas [arcmin]
Argument 3: Filename with n(z)
Argument 4: Outputfolder (needs to exist)
Argument 5: Filename for covariance parameters (ASCII, see necessary_files/cov_param for an example)
Argument 6: Calculate T1? (0 or 1)
Argument 7: Calculate T2? (0 or 1)
Argument 8: Calculate T4? (0 or 1)
Argument 9: Calculate T5? (0 or 1)
Argument 10: Calculate T6? (0 or 1)
Argument 11: Calculate T7? (0 or 1)
Argument 12: Calculate T7_2H? (0 or 1)
Argument 13: Survey geometry, either circle, square, infinite, or rectangular
)";

  if (argc != 14)
  {
    std::cerr << message << std::endl;
    exit(-1);
  };

  std::string cosmo_paramfile = argv[1]; // Parameter file
  std::string thetasfn = argv[2];
  std::string nzfn = argv[3];
  std::string out_folder = argv[4];
  std::string covariance_paramfile = argv[5];
  bool calculate_T1 = std::stoi(argv[6]);
  bool calculate_T2 = std::stoi(argv[7]);
  bool calculate_T4 = std::stoi(argv[8]);
  bool calculate_T5 = std::stoi(argv[9]);
  bool calculate_T6 = std::stoi(argv[10]);
  bool calculate_T7 = std::stoi(argv[11]);
  bool calculate_T7_2h = std::stoi(argv[12]);
  std::string type_str = argv[13];

  std::cerr<<"Calculating term1:"<<calculate_T1<<std::endl;
  std::cerr<<"Calculating term2:"<<calculate_T2<<std::endl;
  std::cerr<<"Calculating term4:"<<calculate_T4<<std::endl;
  std::cerr<<"Calculating term5:"<<calculate_T5<<std::endl;
  std::cerr<<"Calculating term6:"<<calculate_T6<<std::endl;
  std::cerr<<"Calculating term7:"<<calculate_T7<<std::endl;
  std::cerr<<"Calculating term7 (2h):"<<calculate_T7_2h<<std::endl;



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
  lMin = 0;
  thetaMax_smaller = covPar.thetaMax_smaller;
  area = covPar.area;

  // Set Cosmology
  cosmology cosmo(cosmo_paramfile);

  // Read in n(z)
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

  // Set survey geometry
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

  std::cerr<<"Using survey geometry "<<type_str<<std::endl;

  set_cosmology(cosmo, &nz);

  // Set aperture radii
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

  // For T5, T6 and T7, halomodel calculations are needed, so this needs to be initialized.
  if (calculate_T5 || calculate_T6 || calculate_T7)
  {
    initHalomodel();
  }

  std::cerr << "Finished copying constants" << std::endl;

  std::cerr << "Using n(z) from " << nzfn << std::endl;

  std::cerr << "Finished initializations" << std::endl;

  // Calculations

  int N = thetas.size();

  std::vector<double> Cov_term1s, Cov_term2s, Cov_term4s, Cov_term5s, Cov_term6s, Cov_term7s, Cov_term7_2hs;

  std::vector<std::vector<double>> theta_combis;
  for (int i = 0; i < N; i++)
  {
    double theta1 = convert_angle_to_rad(thetas.at(i)); // Conversion to rad
    for (int j = i; j < N; j++)
    {
      double theta2 = convert_angle_to_rad(thetas.at(j));
      for (int k = j; k < N; k++)
      {
        double theta3 = convert_angle_to_rad(thetas.at(k));
        std::vector<double> thetas_123 = {theta1, theta2, theta3};

        theta_combis.push_back(thetas_123);
      }
    }
  }

  int N_ind = theta_combis.size(); // Number of independent theta-combinations
  int N_total = N_ind * (N_ind + 1) / 2;

  int completed_steps = 0;

  auto begin = std::chrono::high_resolution_clock::now(); // Begin time measurement
  for (int i = 0; i < N_ind; i++)
  {
    for (int j = i; j < N_ind; j++)
    {
      try
      {
        if (calculate_T1)
        {
          double term1 = T1_total(theta_combis.at(i), theta_combis.at(j));
          Cov_term1s.push_back(term1);
        };
        if (calculate_T2)
        {
          double term2 = T2_total(theta_combis.at(i), theta_combis.at(j));
          Cov_term2s.push_back(term2);
        };
        if (calculate_T4)
        {
          double term4 = T4_total(theta_combis.at(i), theta_combis.at(j));
          Cov_term4s.push_back(term4);
        };
        if (calculate_T5)
        {
          double term5 = T5_total(theta_combis.at(i), theta_combis.at(j));
          Cov_term5s.push_back(term5);
        }
        if (calculate_T6)
        {
          double term6 = T6_total(theta_combis.at(i), theta_combis.at(j));
          Cov_term6s.push_back(term6);
        }
        if (calculate_T7)
        {
          double term7 = T7_total(theta_combis.at(i), theta_combis.at(j));
          Cov_term7s.push_back(term7);
        }
        if (calculate_T7_2h)
        {
          double term7_2h = T7_SSC(theta_combis.at(i), theta_combis.at(j));
          Cov_term7_2hs.push_back(term7_2h);
        }
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

      fprintf(stderr, "\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step. Last thetas: (%.2f, %.2f, %.2f, %.2f, %.2f, %.2f) [%s]",
              static_cast<int>(progress * 100),
              elapsed.count() * 1e-9 / 3600,
              (N_total - completed_steps) * elapsed.count() * 1e-9 / 3600 / completed_steps,
              elapsed.count() * 1e-9 / completed_steps,
              convert_rad_to_angle(theta_combis.at(i).at(0)), convert_rad_to_angle(theta_combis.at(i).at(1)), convert_rad_to_angle(theta_combis.at(i).at(2)),
              convert_rad_to_angle(theta_combis.at(j).at(0)), convert_rad_to_angle(theta_combis.at(j).at(1)), convert_rad_to_angle(theta_combis.at(j).at(2)), "arcmin");
    }
  }





  // Output

  char filename[255];
  double n_deg = n / convert_rad_to_angle(1, "deg") / convert_rad_to_angle(1, "deg");
  double thetaMax_deg = convert_rad_to_angle(thetaMax, "deg");

  if (calculate_T1)
  {
    sprintf(filename, "cov_%s_term1Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
            type_str.c_str(), sigma, n_deg, thetaMax_deg);
    std::cerr << "Writing Term1 to " << out_folder + filename << std::endl;
    try
    {
      writeCov(Cov_term1s, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      writeCov(Cov_term1s, N_ind, filename);
    }
  };

  if (calculate_T2)
  {
    sprintf(filename, "cov_%s_term2Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
            type_str.c_str(), sigma, n_deg, thetaMax_deg);
    std::cerr << "Writing Term2 to " << out_folder + filename << std::endl;

    try
    {
      writeCov(Cov_term2s, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      writeCov(Cov_term2s, N_ind, filename);
    }
  };

  if (calculate_T4)
  {
    sprintf(filename, "cov_%s_term4Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
            type_str.c_str(), sigma, n_deg, thetaMax_deg);

    try
    {
      writeCov(Cov_term4s, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      writeCov(Cov_term4s, N_ind, filename);
    }
  };

  if (calculate_T5)
  {
    sprintf(filename, "cov_%s_term5Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
            type_str.c_str(), sigma, n_deg, thetaMax_deg);

    try
    {
      writeCov(Cov_term5s, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      writeCov(Cov_term5s, N_ind, filename);
    }
  };

  if (calculate_T6)
  {
    sprintf(filename, "cov_%s_term6Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
            type_str.c_str(), sigma, n_deg, thetaMax_deg);

    try
    {
      writeCov(Cov_term6s, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      writeCov(Cov_term6s, N_ind, filename);
    }
  };

  if (calculate_T7)
  {
    sprintf(filename, "cov_%s_term7Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
            type_str.c_str(), sigma, n_deg, thetaMax_deg);

    try
    {
      writeCov(Cov_term7s, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      writeCov(Cov_term7s, N_ind, filename);
    }
  };

    if (calculate_T7_2h)
  {
    sprintf(filename, "cov_%s_term7_2h_Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
            type_str.c_str(), sigma, n_deg, thetaMax_deg);

    try
    {
      writeCov(Cov_term7_2hs, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      writeCov(Cov_term7_2hs, N_ind, filename);
    }
  };


  return 0;
}
