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

  int n_z=400; //Number of redshift bins for grids
  double z_max=1.1; //maximal redshift

  bool fastCalc=false; //whether calculations should be sped up
  BispectrumCalculator bispectrum(cosmo, n_z, z_max, fastCalc);

  //Initialize Aperture Statistics  
  ApertureStatistics apertureStatistics(&bispectrum);

  // Set up thetas for which ApertureStatistics are calculated
  std::vector<double> thetas{0.5, 1, 2, 4, 8, 16, 32}; //Thetas in arcmin
  int N=thetas.size();
  
  // Set up vector for aperture statistics
  #if ONLY_DIAGONAL 
  std::vector<double> Cov_MapMapMaps(pow(N,3));
  #else 
  std::vector<double> Cov_MapMapMaps(pow(N,6));
  #endif //ONLY_DIAGONAL
  

int completed_steps = 0;
int Ntotal;
if(ONLY_DIAGONAL) Ntotal = N*(N+1)*(N+2)/6;
else Ntotal = pow(N*(N+1)*(N+2)/6,2);


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
	      if(ONLY_DIAGONAL)
                {
		  double thetas_456[3]={theta1, theta2, theta3};
		  
		  double MapMapMap=apertureStatistics.MapMapMap_covariance_Gauss(thetas_123,thetas_456,survey_area); //Do calculation
		  
#if CONSTANT_POWERSPECTRUM
		  MapMapMap*=P*P*P;
#endif
		  // Do assigment (including permutations)
		  Cov_MapMapMaps.at(i*N*N+j*N+k)=MapMapMap;
		  Cov_MapMapMaps.at(i*N*N+k*N+j)=MapMapMap;
		  Cov_MapMapMaps.at(j*N*N+i*N+k)=MapMapMap;
		  Cov_MapMapMaps.at(j*N*N+k*N+i)=MapMapMap;
		  Cov_MapMapMaps.at(k*N*N+i*N+j)=MapMapMap;
		  Cov_MapMapMaps.at(k*N*N+j*N+i)=MapMapMap;
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

			      double MapMapMap=apertureStatistics.MapMapMap_covariance_Gauss(thetas_123,thetas_456,survey_area); //Do calculation
#if CONSTANT_POWERSPECTRUM
			      MapMapMap*=P*P*P;
#endif
			      // Do assigment (including permutations)
			      int index_123[3] = {i,j,k};
			      int index_456[3] = {ii,jj,kk};

			      std::sort(index_123,index_123+3);
			      std::sort(index_456,index_456+3);
			      do{
				do{
				  Cov_MapMapMaps.at(index_123[0]*pow(N,5)+index_123[1]*pow(N,4)+index_123[2]*pow(N,3)+index_456[0]*N*N+index_456[1]*N+index_456[2])=MapMapMap;
				}
				while(std::next_permutation(index_123,index_123+3));
			      }
			      while(std::next_permutation(index_456,index_456+3));
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
  
  //Output
  std::string outfn;
  std::ofstream out;
  
#if test_analytical
  if(ONLY_DIAGONAL) outfn="../results_analytical/MapMapMap_cov_diag.dat";
  else outfn="../results_analytical/MapMapMap_cov.dat";
#elif slics
  if(ONLY_DIAGONAL) outfn="../results_SLICS/MapMapMap_cov_diag.dat";
  else outfn="../results_SLICS/MapMapMap_cov.dat";
#else
  if(ONLY_DIAGONAL) outfn="../results_MR/MapMapMap_cov_diag.dat";
  else outfn="../results_MR/MapMapMap_cov.dat";
#endif
#if CONSTANT_POWERSPECTRUM
  char sigma_str[10];
  char n_str[10];
  char A_str[10];
  sprintf(sigma_str, "%.1f", sigma);
  sprintf(n_str, "%.3f", n);
  sprintf(A_str, "%d", 240);
  outfn="../../Covariance_randomField/results/covariance_ccode_"+std::string(sigma_str)+"_"+std::string(n_str)+"_240.dat";
#endif
  
  std::cout<<"Writing results to "<<outfn<<std::endl;
  out.open(outfn.c_str());
  
  //Print out ==> Should not be parallelized!!!
  if(ONLY_DIAGONAL)
    {
    for (int i=0; i<N; i++)
      {
        for(int j=0; j<N; j++)
	  {
            for(int k=0; k<N; k++)
	      {
		out<<thetas[i]<<" "
		   <<thetas[j]<<" "
		   <<thetas[k]<<" "
		   <<Cov_MapMapMaps.at(k*N*N+i*N+j)<<" "
		   <<std::endl;
	      };
	  };
      };
    }
  else
    {
      for (int i=0; i<N; i++)
	{
	  for(int j=0; j<N; j++)
	    {
	      for(int k=0; k<N; k++)
		{
		  for(int ii=0; ii<N; ii++)
		    {
		      for(int jj=0; jj<N; jj++)
			{
			  for(int kk=0; kk<N; kk++)
			    {
			      out<<thetas[i]<<" "
				 <<thetas[j]<<" "
				 <<thetas[k]<<" "
				 <<thetas[ii]<<" "
				 <<thetas[jj]<<" "
				 <<thetas[kk]<<" "
				 <<Cov_MapMapMaps.at(i*pow(N,5)+j*pow(N,4)+k*pow(N,3)+ii*N*N+jj*N+kk)<<" "
				 <<std::endl;
			    }
			}
		    }
		};
	    };
	};
      
    }

    

  return 0;
}