#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

/**
 * @file calculateApertureStatistics.cu
 * This executable calculates <MapMapMap> from the
 * Takahashi+ Bispectrum
 * Aperture radii are read from file and <MapMapMap> is only calculated for
 * independent combis of thetas Code uses CUDA and cubature library  (See
 * https://github.com/stevengj/cubature for documentation)
 * @author Sven Heydenreich
 */
int main(int argc, char *argv[]) {
  // Read in command line

  const char *message = R"( 
calculateApertureStatistics.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename for covariance parameters (ASCII, see necessary_files/HOWLS_covpar.dat for an example)
Argument 3: Filename with thetas [arcmin]
Argument 4: Covariance mode, 2: second-order, 3: third-order, 23: cross
Argument 5: Outputfilename, directory needs to exist 
Argument 6: 0: calculate only variance, or 1: calculate full covariance
Argument 7: Shapenoise, 0: ignore, 1: calculate
Argument 8: 0: use analytic n(z) (only works for MR and SLICS), or 1: use n(z) from file                  
Argument 9 (optional): Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/HOWLS_covariance.dat ../necessary_files/HOWLS_thetas.dat 3 ../../results_MR/MapMapMap_covariance.dat 1 1 1 ../necessary_files/nz_MR.dat
)";

  if (argc < 8) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile, covariance_paramfile, thetasfn, outfn, nzfn;
  bool nz_from_file = false;
  bool calculate_covariance = false;
  bool shapenoise = false;
  int covariance_mode = 0;

  cosmo_paramfile = argv[1];
  covariance_paramfile = argv[2];
  thetasfn = argv[3];
  covariance_mode = std::stoi(argv[4]);
  outfn = argv[5];
  calculate_covariance = std::stoi(argv[6]);
  shapenoise = std::stoi(argv[7]);
  nz_from_file = std::stoi(argv[8]);
  if (nz_from_file) {
    nzfn = argv[9];
  };

  if(covariance_mode != 2 && covariance_mode != 3 && covariance_mode != 23)
  {
    std::cerr << "Invalid covariance mode" << std::endl
    << "Valid modes: 2,3,23, given mode: " << covariance_mode << std::endl;
    exit(1);
  }

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);
  double dz = cosmo.zmax / ((double)n_redshift_bins - 1); // redshift binsize

  // Read in n_z
  std::vector<double> nz;
  if (nz_from_file) {

    read_n_of_z(nzfn, dz, n_redshift_bins, nz);
  };

  // Read in covariance parameters
  covarianceParameters covPar;
  read_covariance_param(covariance_paramfile, covPar);


  // Check if output file can be opened
  std::ofstream out;
  out.open(outfn.c_str());
  if (!out.is_open()) {
    std::cerr << "Couldn't open " << outfn << std::endl;
    exit(1);
  };

  // Read in thetas
  std::vector<double> thetas;
  read_thetas(thetasfn, thetas);
  int N = thetas.size();
  
  // User output
  std::cerr << "Using cosmology from " << cosmo_paramfile << ":" << std::endl;
  std::cerr << cosmo;
  std::cerr << "Using thetas in " << thetasfn << std::endl;
  std::cerr << "Covariance from " << covariance_paramfile << ":" << std::endl;
  std::cerr << covPar;
  if(shapenoise)
    std::cerr << "Calculating covariance WITH shapenoise" << std::endl;
  else
    std::cerr << "Calculating covariance WITHOUT shapenoise" << std::endl;
  std::cerr << "Writing to:" << outfn << std::endl;

  // Initialize Bispectrum

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A96, &A96, 48 * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_W96, &W96, 48 * sizeof(double)));

  if (nz_from_file) {
    std::cerr << "Using n(z) from " << nzfn << std::endl;
    set_cosmology(cosmo, dz, nz_from_file, &nz);
  } else {
    set_cosmology(cosmo, dz);
  };

//   // Borders of integral
//   double phiMin = 0.0;
//   double phiMax = 6.28319;
//   double lMin = 1;


if(covariance_mode == 3)
{

    // Set up vector for aperture statistics

  int completed_steps = 0;
  int Ntotal;
  std::vector<double> Cov_MapMapMaps;
  std::vector<double> Cov_MapMapMaps_term2;
    if(calculate_covariance)
    {
      Ntotal = pow(N*(N+1)*(N+2)/6,2);
    }
    else
    {
      Ntotal = N*(N+1)*(N+2)/6;
    }
    Cov_MapMapMaps.reserve(Ntotal);
    Cov_MapMapMaps_term2.reserve(Ntotal);


  auto begin=std::chrono::high_resolution_clock::now(); //Begin time measurement
  //Calculate <MapMapMap>(theta1, theta2, theta3) 
  //This does the calculation only for theta1<=theta2<=theta3, but because of
  //the properties of omp collapse, the for-loops are defined starting from 0
  for (int i=0; i<N; i++)
      {
      double theta1=thetas.at(i)*3.1416/180./60; //Conversion to rad
      for (int j=i; j<N; j++)
      {
      double theta2=thetas.at(j)*3.1416/180./60.;
      for(int k=j; k<N; k++)
          {
          double theta3=thetas.at(k)*3.1416/180./60.;
          double thetas_123[3]={theta1, theta2, theta3};
          if(!calculate_covariance)
                  {
          double thetas_456[3]={theta1, theta2, theta3};
          
          double MapMapMap=Gaussian_MapMapMap_Covariance(thetas_123,thetas_456,covPar,shapenoise); //Do calculation
          double MapMapMap_term2=Gaussian_MapMapMap_Covariance_term2(thetas_123,thetas_456,covPar,shapenoise); //Do calculation

              // Do assigment (including permutations)
              Cov_MapMapMaps.push_back(MapMapMap);
              Cov_MapMapMaps_term2.push_back(MapMapMap_term2);

              auto end = std::chrono::high_resolution_clock::now();
              auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
              completed_steps++;
              double progress = (completed_steps*1.)/(Ntotal);
              
              printf("\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step.",
                static_cast<int>(progress*100),
                elapsed.count()*1e-9/3600,
                (Ntotal-completed_steps)*elapsed.count()*1e-9/3600/completed_steps,
                elapsed.count()*1e-9/completed_steps);
              
                    }
              else
                    {
              for(int ii=0; ii<N; ii++)
                        {
                  double theta4=thetas.at(ii)*3.1416/180./60; //Conversion to rad
                  for(int jj=ii; jj<N; jj++)
                            {
                  double theta5=thetas.at(jj)*3.1416/180./60.;
                  for(int kk=jj; kk<N; kk++)
                                {                          
                                  
                      double theta6=thetas.at(kk)*3.1416/180./60.;
                      double thetas_456[3]={theta4, theta5, theta6};

                      double MapMapMap=Gaussian_MapMapMap_Covariance(thetas_123,thetas_456,covPar,shapenoise); //Do calculation
                      double MapMapMap_term2=Gaussian_MapMapMap_Covariance_term2(thetas_123,thetas_456,covPar,shapenoise); //Do calculation
            
                      // Do assigment (including permutations)
                      Cov_MapMapMaps.push_back(MapMapMap);
                      Cov_MapMapMaps_term2.push_back(MapMapMap_term2);

                      auto end = std::chrono::high_resolution_clock::now();
                      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                      completed_steps++;
                      double progress = (completed_steps*1.)/(Ntotal);
                      
                      printf("\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step. Current thetas: (%.1f, %.1f, %.1f, %.1f, %.1f, %.1f)",
                        static_cast<int>(progress*100),
                        elapsed.count()*1e-9/3600,
                        (Ntotal-completed_steps)*elapsed.count()*1e-9/3600/completed_steps,
                        elapsed.count()*1e-9/completed_steps,
                        theta1*180*60/3.1416, theta2*180*60/3.1416, theta3*180*60/3.1416, theta4*180*60/3.1416, theta5*180*60/3.1416, theta6*180*60/3.1416);
                                }
                            }
                        }
                    }
              
                };
            };
        };
    std::cout << std::endl << "Done! Writing output..." << std::endl;

    // Output
    //Print out ==> Should not be parallelized!!!
    int steps = 0;
    if(!calculate_covariance)
      {
      for (int i=0; i<N; i++)
        {
          for(int j=i; j<N; j++)
      {
              for(int k=j; k<N; k++)
          {
      out<<thetas[i]<<" "
        <<thetas[j]<<" "
        <<thetas[k]<<" "
        <<Cov_MapMapMaps.at(steps)<<" "
        <<Cov_MapMapMaps_term2.at(steps)<<" "
            <<std::endl;
            steps++;
          };
      };
        };
      }
    else
      {
        for (int i=0; i<N; i++)
    {
      for(int j=i; j<N; j++)
        {
          for(int k=j; k<N; k++)
      {
        for(int ii=0; ii<N; ii++)
          {
            for(int jj=ii; jj<N; jj++)
        {
          for(int kk=jj; kk<N; kk++)
            {
              out<<thetas[i]<<" "
          <<thetas[j]<<" "
          <<thetas[k]<<" "
          <<thetas[ii]<<" "
          <<thetas[jj]<<" "
          <<thetas[kk]<<" "
          <<Cov_MapMapMaps.at(steps)<<" "
          <<Cov_MapMapMaps_term2.at(steps)<<" "
          <<std::endl;
          steps++;
            }
        }
          }
      };
        };
    };
        
      }
  }

  if(covariance_mode == 2)
  {

    int completed_steps = 0;
    int Ntotal;
    std::vector<double> Cov_MapMapMaps;
      if(calculate_covariance)
      {
        Ntotal = N*N;
      }
      else
      {
        Ntotal = N;
      }
      Cov_MapMapMaps.reserve(Ntotal);
  
    auto begin=std::chrono::high_resolution_clock::now(); //Begin time measurement
    //Calculate <MapMapMap>(theta1, theta2, theta3) 
    //This does the calculation only for theta1<=theta2<=theta3, but because of
    //the properties of omp collapse, the for-loops are defined starting from 0
    for (int i=0; i<N; i++)
    {
      double theta_1=thetas.at(i)*3.1416/180./60; //Conversion to rad
      if(!calculate_covariance)
      {
        double theta_2=theta_1;
        double MapMap=Gaussian_Map2_Covariance(theta_1,theta_2,covPar,shapenoise); //Do calculation
                
        // Do assigment (including permutations)
        Cov_MapMapMaps.push_back(MapMap);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        completed_steps++;
        double progress = (completed_steps*1.)/(Ntotal);
        
        printf("\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step.",
          static_cast<int>(progress*100),
          elapsed.count()*1e-9/3600,
          (Ntotal-completed_steps)*elapsed.count()*1e-9/3600/completed_steps,
          elapsed.count()*1e-9/completed_steps);
                
      }
      else
      {
        for(int ii=0; ii<N; ii++)
        {
          double theta_2=thetas.at(ii)*3.1416/180./60; //Conversion to rad
          double MapMap=Gaussian_Map2_Covariance(theta_1,theta_2,covPar,shapenoise); //Do calculation
  
          Cov_MapMapMaps.push_back(MapMap);
          auto end = std::chrono::high_resolution_clock::now();
          auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
          completed_steps++;
          double progress = (completed_steps*1.)/(Ntotal);
          
          printf("\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step. Current thetas: (%.1f, %.1f)",
            static_cast<int>(progress*100),
            elapsed.count()*1e-9/3600,
            (Ntotal-completed_steps)*elapsed.count()*1e-9/3600/completed_steps,
            elapsed.count()*1e-9/completed_steps,
            theta_1*180*60/3.1416, theta_2*180*60/3.1416);
        }
      }
    }
      std::cout << std::endl << "Done! Writing output..." << std::endl;
  
      // Output
      //Print out ==> Should not be parallelized!!!
      int steps = 0;
      if(!calculate_covariance)
      {
        for (int i=0; i<N; i++)
        {
          out<<thetas[i]<<" "
          <<Cov_MapMapMaps.at(steps)<<" "
          <<std::endl;
          steps++;
        }
      }
      else
      {
        for (int i=0; i<N; i++)
        {
          for(int ii=0; ii<N; ii++)
          {
            out<<thetas[i]<<" "
            <<thetas[ii]<<" "
            <<Cov_MapMapMaps.at(steps)<<" "
            <<std::endl;
            steps++;
          }
        }
      }
  }

  if(covariance_mode == 23)
  {
    std::cerr << "Error: Cross-covariance not yet implemented." << std::endl;
    exit(1);
  }


  std::cout << "Done." << std::endl;
    

  return 0;
}
