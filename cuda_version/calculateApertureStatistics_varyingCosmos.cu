#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono> //For time measurements
/**
 * @file calculateApertureStatistics_varyingCosmos.cu
 * This executable calculates <MapMapMap> for variations of
 * the cosmological parameters \f$h$\f, \f$\sigma_8$\f, \f$\Omega_b$\f, 
 * \f$n_s$\f, \f$w$\f, \f$\Omega_m$\f, and \f$\Omega_\Lambda$\f 
 * for predefined thetas from the Takahashi+ Bispectrum
 * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 * @warning Currently only equilateral triangles
 * @warning thetas currently hardcoded
 * @warning Main cosmology is hardcoded (Set to either MS)
 * @todo Thetas should be read from command line
 * @todo cosmology should be read from command line
 */

int main()
{
    

  std::string cosmo_paramfile, outfn;

  if(slics)
    {
      // Set Up Cosmology
      cosmo_paramfile="SLICS_cosmo.dat";
      // Set output file
      outfn="../../results_SLICS/MapMapMap_varyingCosmos.dat";
    }
  else
    {
      // Set Up Cosmology
      cosmo_paramfile="MR_cosmo.dat";
      // Set output file
      outfn="../../results_MR/MapMapMap_varyingCosmos.dat";
    };
  
  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  // Check output file
  std::ofstream out;
  out.open(outfn.c_str());
  if(!out.is_open())
    {
      std::cerr<<"Couldn't open "<<outfn<<std::endl;
      exit(1);
    };

  // User output
  std::cerr<<"Using cosmology:"<<std::endl;
  std::cerr<<cosmo;
  std::cerr<<"Writing to:"<<outfn<<std::endl;
  
  //Initialize Bispectrum

  double dz = cosmo.zmax/((double) n_redshift_bins); //redshift binsize
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A96,&A96,48*sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_W96,&W96,48*sizeof(double)));


    
  // Set up thetas for which ApertureStatistics are calculated
  std::vector<double> thetas{0.5};//{0.5, 1, 2};//, 4, 8, 16, 32}; //Thetas in arcmin
  int N=thetas.size();

  // Borders of integral
  double phiMin=0.0;
  double phiMax=6.28319;
  double lMin=1;

    // Set up cosmologies at which Map^3 is calculated
  // This can probably be done smarter
  // Sets each parameter to N_cosmo values between fac_min*Main Value and fac_max*Main Value
  int N_cosmo=10; //Number of variations for each parameter
  double fac_min=0.9995; //Minimum proportion of main value for each parameter
  double fac_max=1.0005; //Maximum proportion of main value for each parameter
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


  for(int i=0; i<N_cosmo*7; i++)
    {
      std::cout<<"Doing calculations for cosmology "<<i+1<<" of "<<N_cosmo*7<<std::endl;
      auto begin=std::chrono::high_resolution_clock::now(); //Begin time measurement
      // Initialize Bispectrum
      set_cosmology(cosmos[i], dz);
     
      
      //Needed for monitoring
      int Ntotal=N;//Total number of bins that need to be calculated
      int step=0;

      out<<cosmos[i].h<<" "<<cosmos[i].sigma8<<" "<<cosmos[i].omb<<" "<<cosmos[i].ns<<" "<<cosmos[i].w<<" "<<cosmos[i].om<<" "<<cosmos[i].ow<<" ";

      
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

	  double Map3=MapMapMap(thetas_calc, phiMin, phiMax, lMin); //Do calculation
	  out<<Map3<<" ";
	  
	};
      out<<std::endl;
      // Stop measuring time and calculate the elapsed time
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
      
      std::cout<<"Time needed for last cosmology:"<<elapsed.count()*1e-9<<std::endl;
    };
}
