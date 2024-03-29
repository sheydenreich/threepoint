/**
 * @file calculateApertureStatisticsCovariance.cpp
 * This executable calculates Cov_{Gauss}(<MapMapMap>) for predefined thetas from the
 * Takahashi+ Power Spectrum
 * If PARALLEL_INTEGRATION is true, the code is parallelized over the integration point calculation
 * @author Sven Heydenreich
 */

#include "apertureStatisticsCovariance.hpp"
#include "helper.hpp"
#include "cosmology.hpp"

#include <iostream>
#include <chrono>

int main(int argc, char *argv[])
{
  // Read in CLI
  const char *message = R"( 
calculateApertureStatisticsCovariance.x : Wrong number of command line parameters (Needed: 12)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas [unit]
Argument 3: Filename with n(z)
Argument 4: Outputfolder (needs to exist)
Argument 5: Theta_Max [unit], this is the radius for a circular survey and the sidelength for a square survey
Argument 6: sigma, shapenoise (for both components)
Argument 7: n [unit^-2] Galaxy numberdensity
Argument 8: unit, either arcmin, deg, or rad
Argument 9: Calculate T1? (0 or 1)
Argument 10: Calculate T2? (0 or 1)
Argument 11: Calculate T4? (0 or 1)
Argument 12: Survey geometry, either circle, square, or infinite
)";

  if (argc != 13)
  {
    std::cerr << message << std::endl;
    exit(-1);
  };

  std::string cosmo_paramfile = argv[1]; // Parameter file
  std::string thetasfn = argv[2];
  std::string nzfn = argv[3];
  std::string out_folder = argv[4];
  double thetaMax = std::stod(argv[5]);
  double sigma = std::stod(argv[6]);
  double n = std::stod(argv[7]);
  std::string unit = argv[8];
  bool calculate_T1 = std::stoi(argv[9]);
  bool calculate_T2 = std::stoi(argv[10]);
  bool calculate_T4 = std::stoi(argv[11]);
  std::string type = argv[12];

  std::cerr << "Using cosmology from " << cosmo_paramfile << std::endl;
  std::cerr << "Using thetas from " << thetasfn << std::endl;
  std::cerr << "Using n(z) from " << nzfn << std::endl;
  std::cerr << "Results are written to " << out_folder << std::endl;

#if CONSTANT_POWERSPECTRUM
  std::cerr << "WARNING: Uses constant powerspectrum" << std::endl;
#endif

  // Initializations

  double thetaMaxRad = convert_angle_to_rad(thetaMax, unit);
  double nRad = n / convert_angle_to_rad(1, unit) / convert_angle_to_rad(1, unit);

  cosmology cosmo(cosmo_paramfile);

  std::vector<double> nz;
  try
  {
    read_n_of_z(nzfn, 100, cosmo.zmax, nz);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return -1;
  }

  BispectrumCalculator bispectrum(&cosmo, nz, 100, cosmo.zmax);
  bispectrum.sigma = sigma;
  bispectrum.n = nRad;
  ApertureStatistics apertureStatistics(&bispectrum);
  ApertureStatisticsCovariance covariance(type, thetaMaxRad, &apertureStatistics);
  covariance.lMin=0;

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

  // Calculations

  int N = thetas.size();
  int N_ind = N * (N + 1) * (N + 2) / 6; // Number of independent theta-combinations
  int N_total = N_ind * N_ind;

  std::vector<double> Cov_term1s, Cov_term2s, Cov_term4s;

  int completed_steps = 0;

  auto begin = std::chrono::high_resolution_clock::now(); // Begin time measurement
  for (int i = 0; i < N; i++)
  {
    double theta1 = convert_angle_to_rad(thetas.at(i), unit); // Conversion to rad
    for (int j = i; j < N; j++)
    {
      double theta2 = convert_angle_to_rad(thetas.at(j), unit);
      for (int k = j; k < N; k++)
      {
        double theta3 = convert_angle_to_rad(thetas.at(k), unit);
        std::vector<double> thetas_123 = {theta1, theta2, theta3};
        for (int l = 0; l < N; l++)
        {
          double theta4 = convert_angle_to_rad(thetas.at(l), unit); // Conversion to rad
          for (int m = l; m < N; m++)
          {
            double theta5 = convert_angle_to_rad(thetas.at(m), unit);
            for (int n = m; n < N; n++)
            {
              double theta6 = convert_angle_to_rad(thetas.at(n), unit);
              std::vector<double> thetas_456 = {theta4, theta5, theta6};

              try
              {
                if (calculate_T1)
                {
                  double term1 = covariance.T1_total(thetas_123, thetas_456);
                  Cov_term1s.push_back(term1);
                };
                if (calculate_T2)
                {
                  double term2 = covariance.T2_total(thetas_123, thetas_456);
                  Cov_term2s.push_back(term2);
                };
                if (calculate_T4)
                {
                  double term4 = covariance.T4_total(thetas_123, thetas_456);
                  Cov_term4s.push_back(term4);
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

              fprintf(stderr, "\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step. Last thetas: (%.2f, %.2f, %.2f, %.2f, %.2f, %.2f) [%s]",
                      static_cast<int>(progress * 100),
                      elapsed.count() * 1e-9 / 3600,
                      (N_total - completed_steps) * elapsed.count() * 1e-9 / 3600 / completed_steps,
                      elapsed.count() * 1e-9 / completed_steps,
                      convert_rad_to_angle(theta1, unit), convert_rad_to_angle(theta2, unit), convert_rad_to_angle(theta3, unit),
                      convert_rad_to_angle(theta4, unit), convert_rad_to_angle(theta5, unit), convert_rad_to_angle(theta6, unit), unit.c_str());
            }
          }
        }
      }
    }
  }

  // Output

  char filename[255];

  if (calculate_T1)
  {
    sprintf(filename, "cov_%s_term1Numerical_sigma_%.1f_n_%.2f_thetaMax_%.2f.dat",
            type.c_str(), sigma, n, thetaMax);
    std::cerr<<"Writing Term1 to "<<out_folder+filename<<std::endl;
    try
    {
      covariance.writeCov(Cov_term1s, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      covariance.writeCov(Cov_term1s, N_ind, filename);
    }
  };

  if (calculate_T2)
  {
    sprintf(filename, "cov_%s_term2Numerical_sigma_%.1f_n_%.2f_thetaMax_%.2f.dat",
            type.c_str(), sigma, n, thetaMax);
    std::cerr<<"Writing Term2 to "<<out_folder+filename<<std::endl;

    try
    {
      covariance.writeCov(Cov_term2s, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      covariance.writeCov(Cov_term2s, N_ind, filename);
    }
  };

  if (calculate_T4)
  {
    sprintf(filename, "cov_%s_term4Numerical_sigma_%.1f_n_%.2f_thetaMax_%.2f.dat",
            type.c_str(), sigma, n, thetaMax);

    try
    {
      covariance.writeCov(Cov_term4s, N_ind, out_folder + filename);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      std::cerr << "Writing instead to current directory!" << std::endl;
      covariance.writeCov(Cov_term4s, N_ind, filename);
    }
  };

  return 0;
}