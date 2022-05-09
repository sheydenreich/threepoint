#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "helpers.cuh"
#include "halomodel.cuh"
#include "apertureStatisticsCovariance.cuh"
#include "cuba.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
/**
 * @file calculateApertureStatistics.cu
 * This executable calculates <MapMapMap> from the
 * Takahashi+ Bispectrum
 * Aperture radii are read from file and <MapMapMap> is only calculated for
 * independent combis of thetas Code uses CUDA and cubature library  (See
 * https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 */
int main(int argc, char *argv[]) {
  // Read in command line

  const char *message = R"( 
calculateApertureStatistics.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas [arcmin]
Argument 3: Outputfilename, directory needs to exist 
Argument 4: 0: use analytic n(z) (only works for MR and SLICS), or 1: use n(z) from file                  
Argument 5 (optional): Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateApertureStatistics.x ../necessary_files/MR_cosmo.dat ../necessary_files/HOWLS_thetas.dat ../../results_MR/MapMapMap_bispec_gpu_nz.dat 1 ../necessary_files/nz_MR.dat
)";


  if (argc < 5) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile, thetasfn, outfn, nzfn;
  bool nz_from_file = false;

  cosmo_paramfile = argv[1];
  thetasfn = argv[2];
  outfn = argv[3];
  nz_from_file = std::stoi(argv[4]);
  if (nz_from_file) {
    nzfn = argv[5];
  };

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  // Read in n_z
  std::vector<double> nz;
  if (nz_from_file) {
    read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);
  };

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
  std::cerr << "Writing to:" << outfn << std::endl;

 
  int ncores=0;
  int pcores=0;
  cubacores(&ncores , &pcores);
  cubaaccel(&ncores, &pcores);

  // Initialize Bispectrum

  copyConstants();

  if (nz_from_file) {
    std::cerr << "Using n(z) from " << nzfn << std::endl;
    set_cosmology(cosmo, &nz);
  } 
  else 
  {
    set_cosmology(cosmo);
  };

  initHalomodel();
  //thetas={2, 4, 8};
  //N=thetas.size();
  // Set up vector for aperture statistics
  int Ntotal = factorial(N+3)/factorial(N-1)/24; // Total number of bins that need to be calculated, = (N+4+1) ncr 3
  //int Ntotal=N;
  std::vector<double> Map4s;

  // Needed for monitoring

  int step = 0;

  
  //Calculate <MapMapMap>(theta1, theta2, theta3) in three loops
  //Calculation only for theta1<=theta2<=theta3
  for (int i = 0; i < N; i++) 
  {
    double theta1 = convert_angle_to_rad(thetas.at(i)); // Conversion to rad

    for (int j = i; j < N; j++) 
    {
      double theta2 = convert_angle_to_rad(thetas.at(j));

      for (int k = j; k < N; k++) 
      {

        double theta3 = convert_angle_to_rad(thetas.at(k));
        for (int l = k; l < N; l++) 
        {

          double theta4 = convert_angle_to_rad(thetas.at(l));
          std::vector<double> thetas_calc = {theta1, theta2, theta3, theta4};
          // Progress for the impatient user (Thetas in arcmin)
          step += 1;
          //std::cout<<step<<std::endl;
          std::cout << step << "/" << Ntotal << ": Thetas:" << thetas.at(i) << " "
                    << thetas.at(j) << " " << thetas.at(k) << " "<< thetas.at(l)<<std::endl;// " \r";
          //std::cout.flush();

          double Map4_ = Map4(thetas_calc); // Do calculation
          std::cerr<<Map4_<<std::endl;

          Map4s.push_back(Map4_);
        };
      };
    };
  };

  // Output
  step = 0;
  for (int i = 0; i < N; i++) {
    for (int j = i; j < N; j++) {
      for (int k = j; k < N; k++) {
        for (int l=k; l<N; l++){
        out << thetas[i] << " " << thetas[j] << " " << thetas[k] << " " << thetas[l] << " "
            << Map4s.at(step) << " " << std::endl;
        step++;
        };
      };
    };
  };

  return 0;
}
