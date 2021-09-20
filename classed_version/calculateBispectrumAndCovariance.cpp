#include "bispectrum.hpp"
#include <fstream>

int main()
{
  // Set Up Cosmology
  struct cosmology cosmo;

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


    double survey_area = pow(4*M_PI/180.,2);
  //Initialize Bispectrum

  int n_z=400; //Number of redshift bins for grids
  double z_max=1.1; //maximal redshift

  bool fastCalc=false; //whether calculations should be sped up
  BispectrumCalculator bispectrum(cosmo, n_z, z_max, fastCalc);

    double ell_min = 100;
    double ell_max = 100000;

    double lell_min = log(ell_min);
    double lell_max = log(ell_max);

    int n_ell = 30;

    double* bispectrum_covariance_array = new double[n_ell*n_ell*n_ell];
    double* bispectrum_array = new double[n_ell*n_ell*n_ell];
    double* powerspectrum_array = new double[n_ell];

  double ell_array[n_ell];
  for(int i=0;i<n_ell;i++)
  {
      ell_array[i] = exp(lell_min + (lell_max-lell_min)/n_ell*(i+0.5));
      powerspectrum_array[i] = bispectrum.convergence_power_spectrum(ell_array[i]);
  }
  for(int i=0;i<n_ell;i++)
  {
      double ell1 = ell_array[i];
      for(int j=i;j<n_ell;j++)
      {
          double ell2 = ell_array[j];
          for(int k=j;k<n_ell;k++)
          {
              double ell3 = ell_array[k];
              double temp,temp2;
              if(is_triangle(ell1,ell2,ell3))
              {
                temp = bispectrum.bispectrumCovariance(ell1, ell2, ell3, ell1, ell2, ell3, 
                0.13*ell1, 0.13*ell2, 0.13*ell3, 0.13*ell1, 0.13*ell2, 0.13*ell3, survey_area);
                temp2 = bispectrum.bkappa(ell1, ell2, ell3);
              }
              else
              {
                  temp = 0;
                  temp2 = 0;
              }
            bispectrum_covariance_array[i*n_ell*n_ell+j*n_ell+k] = temp;
            bispectrum_covariance_array[i*n_ell*n_ell+k*n_ell+j] = temp;
            bispectrum_covariance_array[j*n_ell*n_ell+i*n_ell+k] = temp;
            bispectrum_covariance_array[j*n_ell*n_ell+k*n_ell+i] = temp;
            bispectrum_covariance_array[k*n_ell*n_ell+i*n_ell+j] = temp;
            bispectrum_covariance_array[k*n_ell*n_ell+j*n_ell+i] = temp;

            bispectrum_array[i*n_ell*n_ell+j*n_ell+k] = temp2;
            bispectrum_array[i*n_ell*n_ell+k*n_ell+j] = temp2;
            bispectrum_array[j*n_ell*n_ell+i*n_ell+k] = temp2;
            bispectrum_array[j*n_ell*n_ell+k*n_ell+i] = temp2;
            bispectrum_array[k*n_ell*n_ell+i*n_ell+j] = temp2;
            bispectrum_array[k*n_ell*n_ell+j*n_ell+i] = temp2;
          }
      }
  }

  std::string outfn = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/model_bispectrum.dat";
  std::ofstream out;
  std::string outfn2 = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/model_bispectrum_gaussian_covariance.dat";
  std::ofstream out2;
  std::string outfn3 = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/model_powerspectrum.dat";
  std::ofstream out3;


std::cout<<"Writing results to "<<outfn<<std::endl;
out.open(outfn.c_str());
out2.open(outfn2.c_str());
out3.open(outfn3.c_str());

    //Print out ==> Should not be parallelized!!!
    for(int i=0; i<n_ell; i++)
      {
        out3<<ell_array[i] << " " << powerspectrum_array[i] << std::endl;
	for(int j=0; j<n_ell; j++)
	  {
	    for(int k=0; k<n_ell; k++)
	      {
		out<<ell_array[i]<<" "
		   <<ell_array[j]<<" "
		   <<ell_array[k]<<" "
		   <<bispectrum_array[i*n_ell*n_ell+j*n_ell+k]<<" "
		   <<std::endl;
        out2<<ell_array[i]<<" "
		   <<ell_array[j]<<" "
		   <<ell_array[k]<<" "
		   <<bispectrum_covariance_array[i*n_ell*n_ell+j*n_ell+k]<<" "
		   <<std::endl;
	      };
	  };
  };


}