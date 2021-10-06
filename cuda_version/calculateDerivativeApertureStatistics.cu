#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono> //For time measurements


/**
 * @file calculateDerivativeApertureStatistics.cpp
 * This executable calculates the derivative of <MapMapMap> wrt to
 * the cosmological parameters \f$h$\f, \f$\sigma_8$\f, \f$\Omega_b$\f, 
 * \f$n_s$\f, \f$w$\f, \f$\Omega_m$\f, and \f$\Omega_\Lambda$\f 
 * for predefined thetas from the Takahashi+ Bispectrum
 * Uses 5-point stencil with stepsize h
 * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 * @warning thetas currently hardcoded
 * @warning Main cosmology is hardcoded (Set to either MS or SLICS), check constant slics in bispectrum.cuh for which cosmology is used! 
 * @todo Thetas should be read from command line
 * @todo cosmology should be read from command line
 */
int main(int argc, char* argv[])
{
  // Set up cosmology (at which derivative is calculated)
  struct cosmology cosmo; ///<cosmology at which derivative is calculated


  bool five_point=false;
  if(five_point) std::cerr<<"Using five-point stencil"<<std::endl;
  
  if(argc!=3)
    {
      std::cerr<<"calculateDerivativeApertureStatistics.x: Need to specify output and stencil stepsize"<<std::endl;
      exit(1);
    };
  
  std::string outfn=argv[1]; ///< Outputfilename
  double h=std::stod(argv[2]); ///<Stepsize of Stencil

  
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
    }

  std::ofstream out;
  out.open(outfn.c_str());
  if(!out.is_open())
    {
      std::cerr<<"Couldn't open "<<outfn<<std::endl;
      exit(1);
    };


  double z_max=1.1; //maximal redshift
  if(slics) z_max=3.;
  double dz = z_max/((double) n_redshift_bins); //redshift binsize
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A96,&A96,48*sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_W96,&W96,48*sizeof(double)));

  // Set up thetas for which ApertureStatistics are calculated
  std::vector<double> thetas{0.5, 1, 2, 4, 8, 16, 32}; //Thetas in arcmin
  int N=thetas.size();

  
  // Borders of integral
  double phiMin=0.0;
  double phiMax=6.28319;
  double lMin=1;

  // Set up cosmologies at which Map^3 is calculated
  // This can probably be done smarter


  std::vector<cosmology> cosmos; ///<container for all cosmologies
  std::vector<double> derivative_parameters; //parameters the derivatives are taken in 

  cosmology newCosmo=cosmo;
  if(five_point)
    {
      newCosmo.h=cosmo.h*(1.-2*h);
      cosmos.push_back(newCosmo);
    };
  newCosmo.h=cosmo.h*(1.-h);
  cosmos.push_back(newCosmo);
  newCosmo.h=cosmo.h*(1.+h);
  cosmos.push_back(newCosmo);
  if(five_point)
    {
      newCosmo.h=cosmo.h*(1.+2*h);
      cosmos.push_back(newCosmo);
    };
    derivative_parameters.push_back(cosmo.h);
  
  newCosmo=cosmo;
  if(five_point)
    {
      newCosmo.sigma8=cosmo.sigma8*(1.-2*h);
      cosmos.push_back(newCosmo);
    };
  
  newCosmo.sigma8=cosmo.sigma8*(1.-h);
  cosmos.push_back(newCosmo);
  newCosmo.sigma8=cosmo.sigma8*(1.+h);
  cosmos.push_back(newCosmo);
  if(five_point)
    {
      newCosmo.sigma8=cosmo.sigma8*(1.+2*h);
      cosmos.push_back(newCosmo);
    };
    derivative_parameters.push_back(cosmo.sigma8);
  
  newCosmo=cosmo;
  if(five_point)
    {
      newCosmo.omb=cosmo.omb*(1-2*h);
      newCosmo.omc=newCosmo.om-newCosmo.omb;
      cosmos.push_back(newCosmo);
    };
  
  newCosmo.omb=cosmo.omb*(1-h);
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  newCosmo.omb=cosmo.omb*(1+h);
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  if(five_point)
    {
      newCosmo.omb=cosmo.omb*(1+2*h);
      newCosmo.omc=newCosmo.om-newCosmo.omb;
      cosmos.push_back(newCosmo);
    };
    derivative_parameters.push_back(cosmo.omb);
  
  newCosmo=cosmo;
  if(five_point)
    {
      newCosmo.ns=cosmo.ns*(1-2*h);
      cosmos.push_back(newCosmo);
    };
  newCosmo.ns=cosmo.ns*(1-h);
  cosmos.push_back(newCosmo);
  newCosmo.ns=cosmo.ns*(1+h);
  cosmos.push_back(newCosmo);
  if(five_point)
    {
      newCosmo.ns=cosmo.ns*(1+2*h);
      cosmos.push_back(newCosmo);
    };
  derivative_parameters.push_back(cosmo.ns);

  newCosmo=cosmo;
  if(five_point)
    {
      newCosmo.w=cosmo.w*(1-2*h);
      cosmos.push_back(newCosmo);
    };
  newCosmo.w=cosmo.w*(1-h);
  cosmos.push_back(newCosmo);
  newCosmo.w=cosmo.w*(1+h);
  cosmos.push_back(newCosmo);
  if(five_point)
    {
      newCosmo.w=cosmo.w*(1+2*h);
      cosmos.push_back(newCosmo);
    };
  derivative_parameters.push_back(cosmo.w);

  newCosmo=cosmo;
  if(five_point)
    {
      newCosmo.om=cosmo.om*(1-2*h);
      newCosmo.omc=newCosmo.om-newCosmo.omb;
      cosmos.push_back(newCosmo);
    };
  newCosmo.om=cosmo.om*(1-h);
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  newCosmo.om=cosmo.om*(1+h);
  newCosmo.omc=newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  if(five_point)
    {
      newCosmo.om=cosmo.om*(1+2*h);
      newCosmo.omc=newCosmo.om-newCosmo.omb;
      cosmos.push_back(newCosmo);
    };
    derivative_parameters.push_back(cosmo.om);

  newCosmo=cosmo;
  if(five_point)
    {
      newCosmo.ow=cosmo.ow*(1-2*h);
      cosmos.push_back(newCosmo);
    };
				  
  newCosmo.ow=cosmo.ow*(1-h);
  cosmos.push_back(newCosmo);
  newCosmo.ow=cosmo.ow*(1+h);
  cosmos.push_back(newCosmo);
  if(five_point)
    {
      newCosmo.ow=cosmo.ow*(1+2*h);
      cosmos.push_back(newCosmo);
    };
  derivative_parameters.push_back(cosmo.ow);

  int Ncosmos=cosmos.size();///<Number of cosmologies

  // Calculation of Map^3
  double MapMapMaps[Ncosmos][N*N*N]; ///<Array which will contain MapMapMap calculated

  
  
  for(int i=0; i<Ncosmos; i++)
    {
      std::cout<<"Doing calculations for cosmology "<<i<<" of "<<Ncosmos<<std::endl;
      auto begin=std::chrono::high_resolution_clock::now(); //Begin time measurement
      // Initialize Bispectrum
      cosmos[i].zmax=z_max;
      set_cosmology(cosmos[i], dz);


      //Needed for monitoring
      int Ntotal=N*(N+1)*(N+2)/6.; //Total number of bins that need to be calculated, = (N+3-1) ncr 3
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
		  //Progress for the impatient user (Thetas in arcmin)
		  step+=1;
		  std::cout<<step<<"/"<<Ntotal<<": Thetas:"<<thetas.at(j)<<" "<<thetas.at(k)<<" "<<thetas.at(l)<<" \r"; //\r is so that only one line is shown
		  std::cout.flush();

		  double Map3=MapMapMap(thetas_calc, phiMin, phiMax, lMin); //Do calculation
	      
		  // Do assigment (including permutations)
		  MapMapMaps[i][j*N*N+k*N+l]=Map3;
		  MapMapMaps[i][j*N*N+l*N+k]=Map3;
		  MapMapMaps[i][k*N*N+j*N+l]=Map3;
		  MapMapMaps[i][k*N*N+l*N+j]=Map3;
		  MapMapMaps[i][l*N*N+j*N+k]=Map3;
		  MapMapMaps[i][l*N*N+k*N+j]=Map3;
		};
	    };
	};

      // Stop measuring time and calculate the elapsed time
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  
      std::cout<<"Time needed for last cosmology:"<<elapsed.count()*1e-9<<std::endl;
    };
    
  
  // Calculation of Derivatives
  int Nderivs=int(Ncosmos/2);
  if(five_point) Nderivs=int(Ncosmos/4); ///<Number of derivatives
  double derivs_MapMapMaps[Nderivs][N*N*N]; ///<Array which will contain MapMapMap calculated

#pragma omp parallel for collapse(2)
  for(int i=0; i<Nderivs; i++)
    {
      for(int j=0; j<N*N*N; j++)
	{
	  if(five_point)
	    {
	      // Stencil calculation: df/dx = [f(x-2h)-8f(x-h)+8f(x+h)-f(x+2h)]/(12h)
	      derivs_MapMapMaps[i][j]=(MapMapMaps[4*i][j]-8*MapMapMaps[4*i+1][j]+8*MapMapMaps[4*i+2][j]-MapMapMaps[4*i+3][j])/(12.*h*derivative_parameters.at(i));
	    }
	  else
	    {
	      derivs_MapMapMaps[i][j]=(MapMapMaps[2*i+1][j]-MapMapMaps[2*i][j])/2/h/derivative_parameters.at(i);
	    }
	}
    }


  // Output (Cannot be parallelized!!)
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
