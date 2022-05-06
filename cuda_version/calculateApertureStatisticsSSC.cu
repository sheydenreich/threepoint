// #include "apertureStatisticsCovariance.cuh"
// #include "cosmology.cuh"
// #include "bispectrum.cuh"
// #include "helpers.cuh"
// #include "cuda_helpers.cuh"

// #include <iostream>
// #include <chrono>

// int main(int argc, char *argv[])
// {
//   // Read in CLI
//   const char *message = R"( 
// calculateApertureStatisticsSSC.x : Wrong number of command line parameters (Needed: 12)
// Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
// Argument 2: Filename with thetas [arcmin]
// Argument 3: Filename with n(z)
// Argument 4: Outputfolder (needs to exist)
// Argument 5: Filename for covariance parameters (ASCII, see necessary_files/cov_param for an example)
// )";

//   if (argc != 6)
//   {
//     std::cerr << message << std::endl;
//     exit(-1);
//   };


//   std::string cosmo_paramfile = argv[1]; // Parameter file
//   std::string thetasfn = argv[2];
//   std::string nzfn = argv[3];
//   std::string out_folder = argv[4];
//   std::string covariance_paramfile = argv[5];


//   std::cerr << "Using cosmology from " << cosmo_paramfile << std::endl;
//   std::cerr << "Using thetas from " << thetasfn << std::endl;
//   std::cerr << "Using n(z) from " << nzfn << std::endl;
//   std::cerr << "Results are written to " << out_folder << std::endl;
//   std::cerr << "Using covariance parameters from" <<covariance_paramfile<<std::endl;


//   // Initializations
//   covarianceParameters covPar(covariance_paramfile);
//   constant_powerspectrum=covPar.shapenoiseOnly;

//   if(constant_powerspectrum)
//   {
//   std::cerr << "WARNING: Uses constant powerspectrum" << std::endl;
//   };  

//   thetaMax = covPar.thetaMax;
//   sigma = covPar.shapenoise_sigma;
//   n = covPar.galaxy_density;
//   lMin = 2*M_PI/thetaMax;
//   type=1;

//   cosmology cosmo(cosmo_paramfile);


//   std::vector<double> nz;
//   try
//   {
//     read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);
//   }
//   catch (const std::exception &e)
//   {
//     std::cerr << e.what() << '\n';
//     return -1;
//   }

//   set_cosmology(cosmo, &nz);

//   std::vector<double> thetas;

//   try
//   {
//     read_thetas(thetasfn, thetas);
//   }
//   catch (const std::exception &e)
//   {
//     std::cerr << e.what() << '\n';
//     return -1;
//   }

//   // Initialize Covariance
//   initSSC();


 
//   std::cerr<<"Finished initializations"<<std::endl;

//    // Calculations

//    int N = thetas.size();
//    int N_ind = N * (N + 1) * (N + 2) / 6; // Number of independent theta-combinations
//    int N_total = N_ind * N_ind;
 
//    std::vector<double> Cov_SSCs;
 
//    int completed_steps = 0;
 
//    auto begin = std::chrono::high_resolution_clock::now(); // Begin time measurement
//    for (int i = 0; i < N; i++)
//    {
//      double theta1 = convert_angle_to_rad(thetas.at(i)); // Conversion to rad
//      for (int j = i; j < N; j++)
//      {
//        double theta2 = convert_angle_to_rad(thetas.at(j));
//        for (int k = j; k < N; k++)
//        {
//          double theta3 = convert_angle_to_rad(thetas.at(k));
//          std::vector<double> thetas_123 = {theta1, theta2, theta3};
//          for (int l = 0; l < N; l++)
//          {
//            double theta4 = convert_angle_to_rad(thetas.at(l)); // Conversion to rad
//            for (int m = l; m < N; m++)
//            {
//              double theta5 = convert_angle_to_rad(thetas.at(m));
//              for (int n = m; n < N; n++)
//              {
//                double theta6 = convert_angle_to_rad(thetas.at(n));
//                std::vector<double> thetas_456 = {theta4, theta5, theta6};
 
//                try
//                {

//                    double res = Cov_SSC(thetas_123, thetas_456);
//                    Cov_SSCs.push_back(res);
//                }

//                catch (const std::exception &e)
//                {
//                  std::cerr << e.what() << '\n';
//                  return -1;
//                }
 
//                // Progress for the impatient user
//                auto end = std::chrono::high_resolution_clock::now();
//                auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
//                completed_steps++;
//                double progress = (completed_steps * 1.) / (N_total);
 
//                fprintf(stderr, "\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step. Last thetas: (%.2f, %.2f, %.2f, %.2f, %.2f, %.2f) [%s]",
//                        static_cast<int>(progress * 100),
//                        elapsed.count() * 1e-9 / 3600,
//                        (N_total - completed_steps) * elapsed.count() * 1e-9 / 3600 / completed_steps,
//                        elapsed.count() * 1e-9 / completed_steps,
//                        convert_rad_to_angle(theta1), convert_rad_to_angle(theta2), convert_rad_to_angle(theta3),
//                        convert_rad_to_angle(theta4), convert_rad_to_angle(theta5), convert_rad_to_angle(theta6), "arcmin");
//              }
//            }
//          }
//        }
//      }
//    }
 
//    // Output
 
//    char filename[255];
//    double n_deg = n / convert_rad_to_angle(1, "deg") / convert_rad_to_angle(1, "deg");
//    double thetaMax_deg = convert_rad_to_angle(thetaMax, "deg");
 

//      sprintf(filename, "cov_SSC_sigma_%.1f_n_%.2f_thetaMax_%.2f_gpu.dat",
//              sigma, n_deg, thetaMax_deg);
//      std::cerr<<"Writing SSC to "<<out_folder+filename<<std::endl;
//      try
//      {
//        writeCov(Cov_SSCs, N_ind, out_folder + filename);
//      }
//      catch (const std::exception &e)
//      {
//        std::cerr << e.what() << '\n';
//        std::cerr << "Writing instead to current directory!" << std::endl;
//        writeCov(Cov_SSCs, N_ind, filename);
//      }
   
//   return 0;
//  }

 