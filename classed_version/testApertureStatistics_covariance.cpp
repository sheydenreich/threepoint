/**
 * @file calculateApertureStatisticsCovariance.cpp
 * This executable calculates Cov_{Gauss}(<MapMapMap>) for predefined thetas from the
 * Takahashi+ Power Spectrum
 * If PARALLEL_INTEGRATION is true, the code is parallelized over the integration point calculation
 * @author Sven Heydenreich
 * @warning thetas currently hardcoded
 * @warning Output is hardcoded
 * @todo Thetas should be read from command line
 * @todo Outputfilename should be read from command line
 */


#include "apertureStatistics.hpp"
#include <fstream>
#include <chrono>

#define ONLY_DIAGONAL false

int main()
{
  // Set Up Cosmology
  struct cosmology cosmo;
    int n_los = 32; //number of lines-of-sight considered for covariance
    double survey_area;
  if(slics)
    {
      printf("using SLICS cosmology...\n");
      cosmo.h=0.6898;     // Hubble parameter
      cosmo.sigma8=0.826; // sigma 8
      cosmo.omb=0.0473;   // Omega baryon
      cosmo.omc=0.2905-cosmo.omb;   // Omega CDM
      cosmo.ns=0.969;    // spectral index of linear P(k)
      cosmo.w=-1.0;
      cosmo.om = cosmo.omb+cosmo.omc;
      cosmo.ow = 1-cosmo.om;
      survey_area = 10.*10.*n_los*pow(M_PI/180.,2);

    }
  else
    {
      printf("using Millennium cosmology...\n");
      cosmo.h = 0.73;
      cosmo.sigma8 = 0.9;
      cosmo.omb = 0.045;
      cosmo.omc = 0.25 - cosmo.omb;
      cosmo.ns = 1.;
      cosmo.w = -1.0;
      cosmo.om = cosmo.omc+cosmo.omb;
      cosmo.ow = 1.-cosmo.om;
      survey_area = 4.*4.*n_los*pow(M_PI/180.,2);
    }

  #if CONSTANT_POWERSPECTRUM
  std::cerr<<"Uses constant powerspectrum"<<std::endl;
  double sigma=0.3; //Shapenoise
  double n = 291.271; //source galaxy density [arcmin^-2]
  double P=sigma/n/(180*60/M_PI)/(180*60/M_PI); //Powerspectrum [rad^2]
  std::cerr<<"with shapenoise:"<<sigma
	   <<" and galaxy number density:"<<n<<" rad^-2"<<std::endl;  
  #endif
  

  //Initialize Bispectrum

  int n_z=100; //Number of redshift bins for grids
  double z_max=1.1; //maximal redshift

  bool fastCalc=false; //whether calculations should be sped up
  BispectrumCalculator bispectrum(cosmo, n_z, z_max, fastCalc);

  //Initialize Aperture Statistics  
  ApertureStatistics apertureStatistics(&bispectrum);

  // Set up thetas for which ApertureStatistics are calculated
  std::vector<double> thetas{1, 2, 3, 4, 5, 6, 7}; //Thetas in arcmin
  int N=thetas.size();
  

int i = 0;
int j = 1;
int k = 2;
int ii = 3;
int jj = 4;
int kk = 5;

double thetas_123[3]={thetas.at(i), thetas.at(j), thetas.at(k)};
double thetas_456[3]={thetas.at(ii), thetas.at(jj), thetas.at(kk)};
double reference = apertureStatistics.MapMapMap_covariance_Gauss(thetas_123,thetas_456,1.);

std::cout << "*****************************************" << std::endl;
std::cout << "Testing permutations in theta_123:" << std::endl;
std::cout << "Thetas, relative errors,\t permuted,\t reference: \n";

int index_123[3] = {i,j,k};
std::sort(index_123,index_123+3);
do{
    double thetas_123_perm[3] = {thetas.at(index_123[0]), thetas.at(index_123[1]), thetas.at(index_123[2])};
    double permuted = apertureStatistics.MapMapMap_covariance_Gauss(thetas_123_perm,thetas_456,1.);
    std::cout << "(" <<
    thetas_123_perm[0] << "," <<
    thetas_123_perm[1] << "," <<
    thetas_123_perm[2] << ";" <<
    thetas_456[0] << "," <<
    thetas_456[1] << "," <<
    thetas_456[2] << "): " <<
     (permuted/reference-1.)*100 << "\%, \t" <<
     permuted << ", \t" <<
     reference << std::endl;
}
while(std::next_permutation(index_123,index_123+3));
std::cout << std::endl;


std::cout << "*****************************************" << std::endl;
std::cout << "Testing permutations in theta_456:" << std::endl;
std::cout << "Thetas, relative errors,\t permuted,\t reference: \n";

int index_456[3] = {ii,jj,kk};
std::sort(index_456,index_456+3);
do{
    double thetas_456_perm[3] = {thetas.at(index_456[0]), thetas.at(index_456[1]), thetas.at(index_456[2])};
    double permuted = apertureStatistics.MapMapMap_covariance_Gauss(thetas_123,thetas_456_perm,1.);
    std::cout << "(" <<
    thetas_123[0] << "," <<
    thetas_123[1] << "," <<
    thetas_123[2] << ";" <<
    thetas_456_perm[0] << "," <<
    thetas_456_perm[1] << "," <<
    thetas_456_perm[2] << "): " <<
     (permuted/reference-1.)*100 << "\%,\t" <<
     permuted << ",\t" <<
     reference << std::endl;
}
while(std::next_permutation(index_456,index_456+3));
std::cout << std::endl;

std::cout << "*****************************************" << std::endl;
std::cout << "Testing permutations of theta_123 and theta_456:" << std::endl;
std::cout << "Relative error: \t";

double permuted = apertureStatistics.MapMapMap_covariance_Gauss(thetas_456,thetas_123,1.);
     std::cout << (permuted/reference-1.)*100 << "\%,\t" <<
     permuted << ",\t" <<
     reference << std::endl;



  return 0;
}
