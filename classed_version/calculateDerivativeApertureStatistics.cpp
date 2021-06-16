/**
 * @file calculateDerivativeApertureStatistics.cpp
 * This executable calculates the derivative of <MapMapMap> wrt to
 * the cosmological parameters \f$h$\f, \f$\sigma_8$\f, \f$\Omega_b$\f, 
 * \f$n_s$\f, \f$w$\f, \f$\Omega_m$\f, and \f$\Omega_\Lambda$\f 
 * for predefined thetas from the Takahashi+ Bispectrum
 * Uses 5-point stencil with stepsize h
 * If PARALLEL_INTEGRATION is true, the code is parallelized over the integration point calculation
 * If PARALLEL_RADII is true, the code is parallelized over the aperture radii
 * @author Laila Linke
 * @warning thetas currently hardcoded
 * @warning Output is hardcoded
 * @warning h (Stencilsize) is hardcoded
 * @warning Main cosmology is hardcoded (Set to either MS or SLICS)
 * @todo Thetas should be read from command line
 * @todo Outputfilename should be read from command line
 * @todo h should be read from command line
 * @todo cosmology should be read from command line
 */

#include "apertureStatistics.hpp"
#include <fstream>
#include <string>
#include <chrono> // For timing
int main()
{
  // Set up cosmology (at which derivative is calculated)
  struct cosmology cosmo; ///<cosmology at which derivative is calculated
  
  std::string outfn; ///< Outputfilename

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
      outfn="../results_SLICS/dMapMapMap.dat";
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
      outfn="../results_MR/dMapMapMap.dat";
    }

  int n_z=400; //Number of redshift bins for grids
  double z_max=1.1; //maximal redshift
  bool fastCalc=false; //whether calculations should be sped up
  
  // Set up thetas for which ApertureStatistics are calculated
  std::vector<double> thetas{0.5, 1, 2, 4, 8, 16, 32}; //Thetas in arcmin
  int N=thetas.size();

  // Set up cosmologies at which Map^3 is calculated
  // This can probably be done smarter

  double h=0.02; ///<Stepsize of Stencil
  std::vector<cosmology> cosmos; ///<container for all cosmologies

  cosmology newCosmo=cosmo;
  newCosmo.h-=cosmo.h*2*h;
  cosmos.push_back(newCosmo);
  newCosmo.h-=cosmo.h*h;
  cosmos.push_back(newCosmo);
  newCosmo.h+=cosmo.h*h;
  cosmos.push_back(newCosmo);
  newCosmo.h+=cosmo.h*2*h;
  cosmos.push_back(newCosmo);

  newCosmo=cosmo;
  newCosmo.sigma8-=cosmo.sigma8*2*h;
  cosmos.push_back(newCosmo);
  newCosmo.sigma8-=cosmo.sigma8*h;
  cosmos.push_back(newCosmo);
  newCosmo.sigma8+=cosmo.sigma8*h;
  cosmos.push_back(newCosmo);
  newCosmo.sigma8+=cosmo.sigma8*2*h;
  cosmos.push_back(newCosmo);

  newCosmo=cosmo;
  newCosmo.omb-=cosmo.omb*2*h;
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  newCosmo.omb-=cosmo.omb*h;
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  newCosmo.omb+=cosmo.omb*h;
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  newCosmo.omb+=cosmo.omb*2*h;
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);

  newCosmo=cosmo;
  newCosmo.ns-=cosmo.ns*2*h;
  cosmos.push_back(newCosmo);
  newCosmo.ns-=cosmo.ns*h;
  cosmos.push_back(newCosmo);
  newCosmo.ns+=cosmo.ns*h;
  cosmos.push_back(newCosmo);
  newCosmo.ns+=cosmo.ns*2*h;
  cosmos.push_back(newCosmo);
  
  newCosmo=cosmo;
  newCosmo.w-=cosmo.w*2*h;
  cosmos.push_back(newCosmo);
  newCosmo.w-=cosmo.w*h;
  cosmos.push_back(newCosmo);
  newCosmo.w+=cosmo.w*h;
  cosmos.push_back(newCosmo);
  newCosmo.w+=cosmo.w*2*h;
  cosmos.push_back(newCosmo);

  newCosmo=cosmo;
  newCosmo.om-=cosmo.om*2*h;
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  newCosmo.om-=cosmo.om*h;
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  newCosmo.om+=cosmo.om*h;
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  newCosmo.om+=cosmo.om*2*h;
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);

  newCosmo=cosmo;
  newCosmo.ow-=cosmo.ow*2*h;
  cosmos.push_back(newCosmo);
  newCosmo.ow-=cosmo.ow*h;
  cosmos.push_back(newCosmo);
  newCosmo.ow+=cosmo.ow*h;
  cosmos.push_back(newCosmo);
  newCosmo.ow+=cosmo.ow*2*h;
  cosmos.push_back(newCosmo);

  int Ncosmos=cosmos.size();///<Number of cosmologies
  
  // Calculation of Map^3
  double MapMapMaps[Ncosmos][N*N*N]; ///<Array which will contain MapMapMap calculated


#if CUBATURE
  std::cout<<"Using cubature for integration"<<std::endl;
#else
  std::cout<<"Using GSL for integration"<<std::endl;
#endif
  
  for(int i=0; i<Ncosmos; i++)
    {
      std::cout<<"Doing calculations for cosmology "<<i<<" of "<<Ncosmos<<std::endl;
      auto begin=std::chrono::high_resolution_clock::now(); //Begin time measurement
      // Initialize Bispectrum
      BispectrumCalculator bispectrum(cosmos[i], n_z, z_max, fastCalc);

      //Initialize Aperture Statistics  
      ApertureStatistics apertureStatistics(&bispectrum);


#if PARALLEL_RADII
#pragma omp parallel for collapse(3)
  //Calculate <MapMapMap>(theta1, theta2, theta3) 
  //This does the calculation only for theta1<=theta2<=theta3, but because of
  //the properties of omp collapse, the for-loops are defined starting from 0
      for (int j=0; j<N; j++)
	{   
	  for (int k=0; k<N; k++)
	    {
	      for(int l=0; l<N; l++)
		{
		  if(k>=j && l>=k) //Only do calculation for theta1<=theta2<=theta3
		    {
		      //Thetas are defined here, because auf omp collapse
		      double theta1=thetas.at(j)*3.1416/180./60; //Conversion to rad
		      double theta2=thetas.at(k)*3.1416/180./60.;
		      double theta3=thetas.at(j)*3.1416/180./60.;
		      double thetas_calc[3]={theta1, theta2, theta3};

		      double MapMapMap=apertureStatistics.MapMapMap(thetas_calc); //Do calculation
		      // Do assigment (including permutations)
		      MapMapMaps[i][j*N*N+k*N+l]=MapMapMap;
		      MapMapMaps[i][j*N*N+l*N+k]=MapMapMap;
		      MapMapMaps[i][k*N*N+j*N+l]=MapMapMap;
		      MapMapMaps[i][k*N*N+l*N+j]=MapMapMap;
		      MapMapMaps[i][l*N*N+j*N+k]=MapMapMap;
		      MapMapMaps[i][l*N*N+k*N+j]=MapMapMap;
		};
	    };
	};
    };
#else
  //Needed for monitoring
  int Ntotal=N*(N+1)*(N+2)/6.; //Total number of bins that need to be calculated, = (N+3+1) ncr 3
  int step=0;

  //Calculate <MapMapMap>(theta1, theta2, theta3) in three loops
  // Calculation only for theta1<=theta2<=theta3, other combinations are assigned
  for (int j=0; j<N; j++)
    {
      double theta1=thetas.at(j)*3.1416/180./60; //Conversion to rad
      for (int k=j; k<N; k++)
	{
	  double theta2=thetas.at(k)*3.1416/180./60.;
	  for(int l=k; l<N; l++)
	    {
	      double theta3=thetas.at(l)*3.1416/180./60.;
	      double thetas_calc[3]={theta1, theta2, theta3};
	      //Progress for the impatient user
	      step+=1;
	      std::cout<<step<<"/"<<Ntotal<<": Thetas:"<<theta1<<" "<<theta2<<" "<<theta3<<std::endl;

	      double MapMapMap=apertureStatistics.MapMapMap(thetas_calc); //Do calculation
	      
	      // Do assigment (including permutations)
	      MapMapMaps[i][j*N*N+k*N+l]=MapMapMap;
	      MapMapMaps[i][j*N*N+l*N+k]=MapMapMap;
	      MapMapMaps[i][k*N*N+j*N+l]=MapMapMap;
	      MapMapMaps[i][k*N*N+l*N+j]=MapMapMap;
	      MapMapMaps[i][l*N*N+j*N+k]=MapMapMap;
	      MapMapMaps[i][l*N*N+k*N+j]=MapMapMap;
	    };
	};
    };
#endif
  // Stop measuring time and calculate the elapsed time
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  
  std::cout<<"Time needed:"<<elapsed.count()*1e-9;
    };
      
  
  // Calculation of Derivatives
  int Nderivs=int(Ncosmos/4); ///<Number of derivatives
  double derivs_MapMapMaps[Nderivs][N*N*N]; ///<Array which will contain MapMapMap calculated

#pragma omp parallel for collapse(2)
  for(int i=0; i<Nderivs; i++)
    {
      for(int j=0; j<N*N*N; j++)
	{
	  // Stencil calculation: df/dx = [f(x-2h)-8f(x-h)+8f(x+h)-f(x+2h)]/(12h)
	  derivs_MapMapMaps[i][j]=(MapMapMaps[4*i][j]-8*MapMapMaps[4*i+1][j]+8*MapMapMaps[4*i+2][j]-MapMapMaps[4*i+3][j])/(12.*h);
	  
	}
    }


  // Output (Cannot be parallelized!!)
  std::ofstream out;
  out.open(outfn.c_str());
  std::cout<<"Writing results to "<<outfn<<std::endl;
  for(int i=0; i<Nderivs; i++)
    {
      for(int j=0; j<N*N*N; j++)
	{
	  out<<derivs_MapMapMaps[i][j]<<" ";
	}
      out<<std::endl;
    }

      
}
    
  
