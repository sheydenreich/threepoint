#include "apertureStatistics.hpp"
#include "bispectrum.hpp"
#include <fstream>

int main()
{
  // Set Up Cosmology
  struct cosmology cosmo;
  double survey_area;
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
      survey_area = pow(10.*M_PI/180.,2);
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
      survey_area = pow(4.*M_PI/180.,2);
    }

    if(CONSTANT_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM || ANALYTICAL_POWERSPECTRUM_V2)
    {
      printf("Using analytic survey area: 10x10 deg!");
      survey_area = pow(10.*M_PI/180.,2);
    }
    int n_pix = 4096;
  //Initialize Bispectrum

  int n_z=100; //Number of redshift bins for grids
  double z_max=1.1; //maximal redshift
  if(slics) z_max=3.;

  bool fastCalc=false; //whether calculations should be sped up
  BispectrumCalculator bispectrum(cosmo, n_z, z_max, fastCalc);

    // double ell_min = 100;
    // double ell_max = 100000;

    // double lell_min = log(ell_min);
    // double lell_max = log(ell_max);

    int n_ell = 30;

    double* bispectrum_covariance_array = new double[n_ell*n_ell*n_ell];
    double* bispectrum_array = new double[n_ell*n_ell*n_ell];
    double* powerspectrum_array = new double[n_ell];

  double ell_array[30] = {100.0, 126.89610031679221, 161.02620275609394, 204.33597178569417, 259.2943797404667, 329.03445623126674, 417.53189365604004, 529.8316906283708, 672.3357536499335, 853.1678524172805, 1082.636733874054, 1373.8237958832638, 1743.3288221999874, 2212.21629107045, 2807.2162039411755, 3562.2478902624443, 4520.35365636024, 5736.152510448682, 7278.953843983146, 9236.708571873865, 11721.022975334794, 14873.521072935118, 18873.918221350996, 23950.26619987486, 30391.95382313195, 38566.20421163472, 48939.00918477499, 62101.694189156166, 78804.62815669904, 100000.0};
  for(int i=0;i<n_ell;i++)
  {
      // ell_array[i] = exp(lell_min + (lell_max-lell_min)/n_ell*(i+0.5));
      // std::cout << ell_array[i] << std::endl;
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
                if(CONSTANT_POWERSPECTRUM)
                {
                  temp *= pow(0.3*0.3/(2.*n_pix*n_pix/survey_area),3);
                }
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
  #if CONSTANT_POWERSPECTRUM

  std::string outfn = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/constant_powerspectrum/model_bispectrum";
  std::ofstream out;
  std::string outfn2 = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/constant_powerspectrum/model_bispectrum_gaussian_covariance";
  std::ofstream out2;
  std::string outfn3 = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/constant_powerspectrum/model_powerspectrum";
  std::ofstream out3;

  #elif ANALYTICAL_POWERSPECTRUM

  std::string outfn = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/analytical_powerspectrum/model_bispectrum";
  std::ofstream out;
  std::string outfn2 = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/analytical_powerspectrum/model_bispectrum_gaussian_covariance";
  std::ofstream out2;
  std::string outfn3 = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/analytical_powerspectrum/model_powerspectrum";
  std::ofstream out3;

  #elif ANALYTICAL_POWERSPECTRUM_V2

  std::string outfn = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/analytical_powerspectrum_v2/model_bispectrum";
  std::ofstream out;
  std::string outfn2 = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/analytical_powerspectrum_v2/model_bispectrum_gaussian_covariance";
  std::ofstream out2;
  std::string outfn3 = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/gaussian_random_field/analytical_powerspectrum_v2/model_powerspectrum";
  std::ofstream out3;

  #else

  std::string outfn = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_SLICS/model_bispectrum_2";
  std::ofstream out;
  std::string outfn2 = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_SLICS/model_bispectrum_gaussian_covariance_2";
  std::ofstream out2;
  std::string outfn3 = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_SLICS/model_powerspectrum_2";
  std::ofstream out3;

  #endif

#if TREAT_DEGENERATE_TRIANGLES
  outfn2.append("_approx_degenerate_triangles");
#endif
outfn.append(".dat");
outfn2.append(".dat");
outfn3.append(".dat");


std::cout<<"Writing results to "<<outfn2<<std::endl;
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