#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "helpers.cuh"

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
 * @author Pierre Burger
 */
int main(int argc, char *argv[])
{
  // Read in command line

  const char *message = R"( 
calculateApertureStatistics.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas [arcmin]
Argument 3: Outputfilename, directory needs to exist 
Argument 4: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateApertureStatistics.x ../necessary_files/MR_cosmo.dat ../necessary_files/HOWLS_thetas.dat ../../results_MR/MapMapMap_bispec_gpu_nz.dat ../necessary_files/nz_MR.dat
)";

  if (argc < 4) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile = argv[1];
  std::string z_combi_file = argv[2];
  std::string theta_combi_file = argv[3];
  std::string output_file = argv[4];

  std::string nzfn_1 = argv[5];
  std::string nzfn_2 = argv[6];
  std::string nzfn_3 = argv[7];
  std::string nzfn_4 = argv[8];
  std::string nzfn_5 = argv[9];

  std::string shape_noise_file = argv[10];
  

  // Check if output file can be opened
  std::ofstream out;
  out.open(output_file.c_str());
  if (!out.is_open())
  {
    std::cerr << "Couldn't open " << output_file << std::endl;
    exit(1);
  };

  std::vector<std::vector<double>> theta_combis;
  std::vector<std::vector<int>> z_combis;
  int n_combis;
  read_combis(z_combi_file, theta_combi_file, z_combis, theta_combis, n_combis);

  // for (int i = 0; i < n_combis; i++){
  //   if(z_combis[2][i]>0)
  //   {
  //     std::cerr<< theta_combis[0][i] << "\t" << theta_combis[1][i] << "\t" << theta_combis[2][i] << "\t" << z_combis[0][i] << "\t" << z_combis[1][i] << "\t" << z_combis[2][i] << std::endl;
  //   }
  //   else{
  //     std::cerr<< theta_combis[0][i] << "\t" << theta_combis[1][i] << "\t" << z_combis[0][i] << "\t" << z_combis[1][i] << std::endl;
  //   }
  // };

  std::vector<double> sigma_epsilon_per_bin;
  std::vector<double> ngal_per_bin;
  read_shapenoise(shape_noise_file, sigma_epsilon_per_bin, ngal_per_bin);
 
  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  // Read in n_z
  std::vector<double> nz_1;
  read_n_of_z(nzfn_1, n_redshift_bins, cosmo.zmax, nz_1);
  std::vector<double> nz_2;
  read_n_of_z(nzfn_2, n_redshift_bins, cosmo.zmax, nz_2);
  std::vector<double> nz_3;
  read_n_of_z(nzfn_3, n_redshift_bins, cosmo.zmax, nz_3);
  std::vector<double> nz_4;
  read_n_of_z(nzfn_4, n_redshift_bins, cosmo.zmax, nz_4);
  std::vector<double> nz_5;
  read_n_of_z(nzfn_5, n_redshift_bins, cosmo.zmax, nz_5);

  std::vector<std::vector<double>> nz;
  nz.push_back(nz_1);
  nz.push_back(nz_2);
  nz.push_back(nz_3);
  nz.push_back(nz_4);
  nz.push_back(nz_5);

  copyConstants();
  set_cosmology(cosmo, &nz, &sigma_epsilon_per_bin, &ngal_per_bin);


  std::vector<double> Map_vector;

  for (int i = 0; i < n_combis; i++){
    if(z_combis[2][i]==999)
    {
      double theta = convert_angle_to_rad(theta_combis[0][i]); // Conversion to rad
      std::vector<int> zbins_calc = {z_combis[0][i], z_combis[1][i]};
      double Map2_value = Map2(theta,zbins_calc); // Do calculation
      Map_vector.push_back(Map2_value);
      //std::cout<< Map2_value <<std::endl;
      std::cerr << i << "/" << n_combis << ": Theta= " << theta_combis[0][i] << " "
                                        << ": zbin1= " << z_combis[0][i] << " "
                                        << ": zbin2= " << z_combis[1][i] << " "
                                        << ": Map2= " << Map2_value << " \r";                               
      std::cerr.flush();
    }
    else{

      double theta_1_rad = convert_angle_to_rad(theta_combis[0][i]); // Conversion to rad
      double theta_2_rad = convert_angle_to_rad(theta_combis[1][i]); // Conversion to rad
      double theta_3_rad = convert_angle_to_rad(theta_combis[2][i]); // Conversion to rad
      std::vector<double> thetas_calc = {theta_1_rad, theta_2_rad, theta_3_rad};
      std::vector<int> zbins_calc = {z_combis[0][i], z_combis[1][i], z_combis[2][i]};
      double Map3_value = MapMapMap(thetas_calc,zbins_calc);
      Map_vector.push_back(Map3_value);
      //std::cout<< Map3_value << std::endl;
      std::cerr << i << "/" << n_combis << ": Theta1= " << theta_combis[0][i] << " "
                                        << ": Theta2= " << theta_combis[1][i] << " "
                                        << ": Theta3= " << theta_combis[2][i] << " "
                                        << ": zbin1= " << z_combis[0][i] << " "
                                        << ": zbin2= " << z_combis[1][i] << " "
                                        << ": zbin3= " << z_combis[2][i] << " "
                                        << ": Map3= " << Map3_value << " \r";
      std::cerr.flush();
    }
  }

  // Output
  for (int i = 0; i < n_combis; i++)
  {
    out << Map_vector.at(i) << " " << std::endl;
  };


  return 0;
}
