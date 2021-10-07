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
 * This executable calculates <MapMapMap> from the
 * Takahashi+ Bispectrum
 * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 */
int main(int argc, char* argv[])
{
  // Read in command line

  const char* message = R"( 
calculateApertureStatistics.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas
Argument 3: Outputfilename, directory needs to exist 
Argument 4: 0: use analytic n(z) (only works for MR and SLICS), or 1: use n(z) from file                  
Argument 5 (optional): Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateApertureStatistics.x ../necessary_files/MR_cosmo.dat ../../results_MR/MapMapMap_bispec_gpu_nz.dat 1 ../necessary_files/nz_MR.dat
)";

  if(argc < 5)
    {
      std::cerr<<message<<std::endl;
      exit(1);
    };

  std::string cosmo_paramfile, thetasfn, outfn, nzfn;
  bool nz_from_file=false;

  cosmo_paramfile=argv[1];
  thetasfn=argv[2];
  outfn=argv[3];
  nz_from_file=std::stoi(argv[4]);
  if(nz_from_file)
    {
      nzfn=argv[5];
    };
  
 
  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);
  double dz = cosmo.zmax/((double) n_redshift_bins-1); //redshift binsize

  // Read in n_z
  std::vector<double> nz;
  if(nz_from_file)
    {

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

  // Read in thetas
  std::vector<double> thetas;
  read_thetas(thetasfn, thetas);
  int N=thetas.size();
  
  // User output
  std::cerr<<"Using cosmology from "<<cosmo_paramfile<<":"<<std::endl;
  std::cerr<<cosmo;
  std::cerr<<"Using thetas in "<<thetasfn<<std::endl;
  std::cerr<<"Writing to:"<<outfn<<std::endl;
  
  //Initialize Bispectrum

 
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A96,&A96,48*sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_W96,&W96,48*sizeof(double)));

  if(nz_from_file)
    {
      std::cerr<<"Using n(z) from "<<nzfn<<std::endl;
      set_cosmology(cosmo, dz, nz_from_file, &nz);
    }
  else
    {
      set_cosmology(cosmo, dz);
    };
  


  // Borders of integral
  double phiMin=0.0;
  double phiMax=6.28319;
  double lMin=1;
  
  // Set up vector for aperture statistics
  int Ntotal=N*(N+1)*(N+2)/6.; //Total number of bins that need to be calculated, = (N+3+1) ncr 3
  std::vector<double> MapMapMaps;

  //Needed for monitoring

  int step=0;

  //Calculate <MapMapMap>(theta1, theta2, theta3) in three loops
  // Calculation only for theta1<=theta2<=theta3
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
	      MapMapMaps.push_back(Map3);
	    };
	};
    };

  //Output
  step=0;
  for (int i=0; i<N; i++)
    {
      for(int j=i; j<N; j++)
	{
	  for(int k=j; k<N; k++)
	    {
	      out<<thetas[i]<<" "
		 <<thetas[j]<<" "
		 <<thetas[k]<<" "
		 <<MapMapMaps.at(step)<<" "
		 <<std::endl;
	      step++;
	    };
	};
    };
    




  
  return 0;
}
