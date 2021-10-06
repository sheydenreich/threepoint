#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"
#include "cosmology.cuh"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
/**
 * @file calculateApertureStatistics.cu
 * This executable calculates <MapMapMap> for predefined thetas from the
 * Takahashi+ Bispectrum
 * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 * @warning thetas currently hardcoded
 * @warning Output is hardcoded
 * @todo Thetas should be read from command line
 * @todo Outputfilename should be read from command line
 */
int main()
{
  std::string cosmo_paramfile, outfn, nzfn;
  bool nz_from_file=false;

  if(slics)
    {
      // Set Up Cosmology
      cosmo_paramfile="SLICS_cosmo.dat";
      // Set output file
      outfn="../../results_SLICS/MapMapMap_bispec_gpu_nz.dat";
      // Set n_z_file
      nzfn="nz_SLICS_euclidlike.dat";
      nz_from_file=true;
    }
  else
    {
      // Set Up Cosmology
      cosmo_paramfile="MR_cosmo.dat";
      // Set output file
      outfn="../../results_MR/MapMapMap_bispec_gpu_nz.dat";
      // Set n_z_file
      nzfn="nz_MR.dat";
      nz_from_file=true;
    };
  
  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);
  double dz = cosmo.zmax/((double) n_redshift_bins); //redshift binsize

  std::vector<double> nz;
  if(nz_from_file)
    {
      // Read in n_z
      read_n_of_z(nzfn, dz, n_redshift_bins, nz);
    };
  
  // Check output file
  std::ofstream out;
  out.open(outfn.c_str());
  if(!out.is_open())
    {
      std::cerr<<"Couldn't open "<<outfn<<std::endl;
      exit(1);
    };

  // User output
  std::cerr<<"Using cosmology from "<<cosmo_paramfile<<":"<<std::endl;
  std::cerr<<cosmo;
  std::cerr<<"Writing to:"<<outfn<<std::endl;
  
  //Initialize Bispectrum

 
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A96,&A96,48*sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_W96,&W96,48*sizeof(double)));

  if(nz_from_file)
    {
      set_cosmology(cosmo, dz, nz_from_file, &nz);
    }
  else
    {
      set_cosmology(cosmo, dz);
    };
  
  // Set up thetas for which ApertureStatistics are calculated
  std::vector<double> thetas{0.5, 1, 2, 4, 8, 16, 32}; //Thetas in arcmin
  int N=thetas.size();

  // Borders of integral
  double phiMin=0.0;
  double phiMax=6.28319;
  double lMin=1;
  
  // Set up vector for aperture statistics
  std::vector<double> MapMapMaps(N*N*N);

  //Needed for monitoring
  int Ntotal=N*(N+1)*(N+2)/6.; //Total number of bins that need to be calculated, = (N+3+1) ncr 3
  int step=0;

  //Calculate <MapMapMap>(theta1, theta2, theta3) in three loops
  // Calculation only for theta1<=theta2<=theta3, other combinations are assigned
  for (int i=0; i<N; i++)
    {
      double theta1=thetas.at(i)*3.1416/180./60; //Conversion to rad
      
      for (int j=i; j<N; j++)
	{
	  double theta2=thetas.at(j)*3.1416/180./60.;

	  for(int k=j; k<N; k++)
	    {

	      double theta3=thetas.at(k)*3.1416/180./60.;
	      double thetas_calc[3]={theta1, theta2, theta3};
	      //Progress for the impatient user (Thetas in arcmin)
	      step+=1;
	      std::cout<<step<<"/"<<Ntotal<<": Thetas:"<<thetas.at(i)<<" "<<thetas.at(j)<<" "<<thetas.at(k)<<" \r";
	      std::cout.flush();

	      double Map3=MapMapMap(thetas_calc, phiMin, phiMax, lMin); //Do calculation
	      
	      // Do assigment (including permutations)
	      MapMapMaps.at(i*N*N+j*N+k)=Map3;
	      MapMapMaps.at(i*N*N+k*N+j)=Map3;
	      MapMapMaps.at(j*N*N+i*N+k)=Map3;
	      MapMapMaps.at(j*N*N+k*N+i)=Map3;
	      MapMapMaps.at(k*N*N+i*N+j)=Map3;
	      MapMapMaps.at(k*N*N+j*N+i)=Map3;
	    };
	};
    };

  //Output
  for (int i=0; i<N; i++)
    {
      for(int j=0; j<N; j++)
	{
	  for(int k=0; k<N; k++)
	    {
	      out<<thetas[i]<<" "
		 <<thetas[j]<<" "
		 <<thetas[k]<<" "
		 <<MapMapMaps.at(k*N*N+i*N+j)<<" "
		 <<std::endl;
	    };
	};
    };
    




  
  return 0;
}
