/**
 * @file calculateApertureStatistics_varyingCosmos.cpp
 * This executable calculates <MapMapMap> for variations of
 * the cosmological parameters \f$h$\f, \f$\sigma_8$\f, \f$\Omega_b$\f, 
 * \f$n_s$\f, \f$w$\f, \f$\Omega_m$\f, and \f$\Omega_\Lambda$\f 
 * for predefined thetas from the Takahashi+ Bispectrum
 * If PARALLEL_INTEGRATION is true, the code is parallelized over the integration point calculation
 * If PARALLEL_RADII is true, the code is parallelized over the aperture radii
 * @author Laila Linke
 * @warning Currently only equilateral triangles
 * @warning thetas currently hardcoded
 * @warning Main cosmology is hardcoded (Set to either MS or SLICS)
 * @todo Thetas should be read from command line
 * @todo cosmology should be read from command line
 */

#include "apertureStatistics.hpp"
#include <fstream>
#include <string>
#include <chrono>
int main()
{
  // Set up main cosmology 
  struct cosmology cosmo;
  std::string outfn;
  
  if(slics)
    {
      printf("using SLICS cosmology...");
      cosmo.h=0.6898;     // Hubble parameter
      cosmo.sigma8=0.826; // sigma 8
      cosmo.omb=0.0473;   // Omega baryon
      cosmo.omc=0.2905-cosmo.omb;   // Omega CDM
      cosmo.ns=0.969;    // spectral index of linear P(k)
      cosmo.w=-1.0;
      cosmo.om = cosmo.omb+cosmo.omc;
      cosmo.ow = 1-cosmo.om;
      outfn="../results_SLICS/Map3_varyingCosmos.dat";
    }
  else
    {
      printf("using Millennium cosmology...");
      cosmo.h = 0.73;
      cosmo.sigma8 = 0.9;
      cosmo.omb = 0.045;
      cosmo.omc = 0.25 - cosmo.omb;
      cosmo.ns = 1.;
      cosmo.w = -1.0;
      cosmo.om = cosmo.omc+cosmo.omb;
      cosmo.ow = 1.-cosmo.om;
      outfn="../../results_MR/Map3_varyingCosmos.dat";
    }

  int n_z=400; //Number of redshift bins for grids
  double z_max=1.1; //maximal redshift
  bool fastCalc=false; //whether calculations should be sped up
  
  // Set up thetas for which ApertureStatistics are calculated
  std::vector<double> thetas{0.5, 1, 2, 4, 8, 16, 32}; //Thetas in arcmin
  int N=thetas.size();

  // Set up cosmologies at which Map^3 is calculated
  // This can probably be done smarter
  // Sets each parameter to N_cosmo values between fac_min*Main Value and fac_max*Main Value
  int N_cosmo=10; //Number of variations for each parameter
  double fac_min=0.95; //Minimum proportion of main value for each parameter
  double fac_max=1.05; //Maximum proportion of main value for each parameter
  double fac_bin=(fac_max-fac_min)/N_cosmo;
  
  std::vector<cosmology> cosmos(N_cosmo*7); ///<container for all cosmologies
  for(int i=0; i<N_cosmo; i++)
    {
      double fac=fac_min+i*fac_bin;
      cosmology newCosmo=cosmo;
      newCosmo.h=cosmo.h*fac;
      cosmos.at(i)=newCosmo;

      newCosmo=cosmo;
      newCosmo.sigma8=cosmo.sigma8*fac;
      cosmos.at(i+N_cosmo)=newCosmo;

      newCosmo=cosmo;
      newCosmo.omb=cosmo.omb*fac;
      newCosmo.omc=newCosmo.om-newCosmo.omb;
      cosmos.at(i+2*N_cosmo)=newCosmo;

      newCosmo=cosmo;
      newCosmo.ns=cosmo.ns*fac;
      cosmos.at(i+3*N_cosmo)=newCosmo;

      newCosmo=cosmo;
      newCosmo.w=cosmo.w*fac;
      cosmos.at(i+4*N_cosmo)=newCosmo;
      
      newCosmo=cosmo;
      newCosmo.om=cosmo.om*fac;
      newCosmo.omc=newCosmo.om-newCosmo.omb;
      cosmos.at(i+5*N_cosmo)=newCosmo;

      newCosmo=cosmo;
      newCosmo.ow=cosmo.ow*fac;
      cosmos.at(i+6*N_cosmo)=newCosmo;
    }


  // Calculation of Map^3
  std::ofstream out;
  out.open(outfn.c_str());
  std::cout<<"Writing results to "<<outfn<<std::endl;

#if CUBATURE
  std::cout<<"Using cubature for integration"<<std::endl;
#else
  std::cout<<"Using GSL for integration"<<std::endl;
#endif
  
  for(int i=0; i<N_cosmo*7; i++)
    {
      std::cout<<"Doing calculations for cosmology "<<i+1<<" of "<<N_cosmo*7<<std::endl;
      auto begin=std::chrono::high_resolution_clock::now(); //Begin time measurement
      // Initialize Bispectrum
      BispectrumCalculator bispectrum(cosmos.at(i), n_z, z_max, fastCalc);
      
      //Initialize Aperture Statistics  
      ApertureStatistics apertureStatistics(&bispectrum);
      
      
      //Needed for monitoring
      int Ntotal=N;//N*(N+1)*(N+2)/6.; //Total number of bins that need to be calculated, = (N+3+1) ncr 3
      int step=0;

      out<<cosmos[i].h<<" "<<cosmos[i].sigma8<<" "<<cosmos[i].omb<<" "<<cosmos[i].ns<<" "<<cosmos[i].w<<" "<<cosmos[i].om<<" "<<cosmos[i].ow<<" ";
      // out<<bispectrum.h<<" "<<bispectrum.sigma8<<" "<<bispectrum.omb<<" "<<bispectrum.ns<<" "<<bispectrum.w<<" "<<bispectrum.om<<" "<<bispectrum.ow<<" ";
      
      //Calculate <MapMapMap>(theta1, theta1, theta1)
      // Calculation only for theta1=theta2=theta3
      for (int j=0; j<N; j++)
	{
	  double theta=thetas.at(j)*3.1416/180./60; //Conversion to rad
	  double thetas_calc[3]={theta, theta, theta};
	  //Progress for the impatient user (Thetas in arcmin)
	  step+=1;
	  std::cout<<step<<"/"<<Ntotal<<": Thetas:"<<thetas.at(j)<<" "<<thetas.at(j)<<" "<<thetas.at(j)<<" \r"; //\r is so that only one line is shown
	  std::cout.flush();

	  double MapMapMap=apertureStatistics.MapMapMap(thetas_calc); //Do calculation
	  out<<MapMapMap<<" ";
	  
	};
      out<<std::endl;
      // Stop measuring time and calculate the elapsed time
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
      
      std::cout<<"Time needed for last cosmology:"<<elapsed.count()*1e-9<<std::endl;
    };
}
