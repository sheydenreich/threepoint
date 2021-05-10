/**
 * @file calculateGamma.cpp
 * This executable calculates <Gamma^0> for predefined sidelengths from the
 * Takahashi+ Bispectrum
 * @warning Make sure test_analytical in bispectrum.hpp is set to false 
 * @author Sven Heydenreich
 * @warning thetas currently hardcoded
 * @warning output is written in file "<shared folder>/results_SLICS/Gamma0.dat or <shared folder>/results_MR/Gamma0.dat"
 * @todo Thetas should be read from command line
 * @todo Outputfilename should be read from command line
 */



#include "bispectrum.hpp"
#include "gamma.hpp"
#include "helper.hpp"
#include <omp.h>



int main()
{
    
    if(test_analytical)
    {
        std::cerr << "**************************************************************" << std::endl;
        std::cerr << "WARNING: test_analytical is active, not using real bispectrum!" << std::endl;
        std::cerr << "**************** Disable it in bispectrum.hpp ****************" << std::endl;
        std::cerr << "**************************************************************" << std::endl;
    }
  struct cosmology cosmo;

  double z_max;
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

      z_max = 3;
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
      z_max = 1.5;
    }

    int steps = 10;
    int usteps = 10;
    int vsteps = 10;

    std::string infile;
    // Reading the triangle configurations
    if(slics)
    {
        infile = "../necessary_files/triangles_slics.dat";
    }
    else
    {
        infile = "../necessary_files/triangles_millennium_new.dat";
    }

    GammaCalculator class_gamma(cosmo, 0.1, 3.5, false, 400, z_max);

    std::vector<treecorr_bin> triangle_configurations;
    read_triangle_configurations(infile, triangle_configurations);

    std::complex<double> result_gamma0[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    std::complex<double> result_gamma1[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    std::complex<double> result_gamma2[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    std::complex<double> result_gamma3[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    printf("Computing 3pcf. This might take a while ...\n");

    // #pragma omp parallel for collapse(3)
    printf("Computing gammas...[100%%]");
    for(int i=0;i<steps;i++){
        for(int j=0;j<usteps;j++){
        	for(int k=0;k<vsteps;k++){
                printf("\b\b\b\b\b\b[%3d%%]",static_cast<int>(100*(1.*i*usteps*vsteps+j*vsteps+k)/(usteps*vsteps*steps)));
        		fflush(stdout);

                    double r,u,v;
                    int id_x = i*usteps*vsteps+j*vsteps+k;
                    r = triangle_configurations[id_x].r;
                    u = triangle_configurations[id_x].u;
                    v = triangle_configurations[id_x].v;


                    double r2 = r; //THIS IS THE BINNING BY JARVIS. FROM THE WEBSITE, NOT THE PAPER.
                    double r3 = r2*u;
                    double r1 = v*r3+r2;
                    // std::complex<double> res_temp = class_gamma.gamma0(M_PI/180./60.*r1, M_PI/180./60.*r2, M_PI/180./60.*r3);
                    // if(isnan(real(res_temp)) || isnan(imag(res_temp))){
                    //     printf("%lf %lf %lf %lf %lf \n",r1,r2,r3,real(res_temp),imag(res_temp));
                    //     res_temp = std::complex<double>(0.0,0.0);
                    // }
                    // assert(!isnan(real(res_temp)) && !isnan(imag(res_temp)));
                    if(r1!=0 && r2!=0 && r3!=0) 
                    {
                        result_gamma0[id_x] = class_gamma.gamma0(M_PI/180./60.*r1, M_PI/180./60.*r2, M_PI/180./60.*r3);
                        result_gamma1[id_x] = class_gamma.gamma1(M_PI/180./60.*r1, M_PI/180./60.*r2, M_PI/180./60.*r3);
                        result_gamma2[id_x] = class_gamma.gamma2(M_PI/180./60.*r1, M_PI/180./60.*r2, M_PI/180./60.*r3);
                        result_gamma3[id_x] = class_gamma.gamma3(M_PI/180./60.*r1, M_PI/180./60.*r2, M_PI/180./60.*r3);
                    }
             }
         }
     }

     printf("\b\b\b\b\b\b[100%%]...Done. Writing results ...");
     fflush(stdout);

     FILE *fp;

     if(test_analytical)
     {
        fp = fopen("/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/Gamma0_0p1_to_120.dat","w");
     }
     if(slics)
     {
         fp = fopen("/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_SLICS/Gamma0.dat","w");
     }
     else
     {
         fp = fopen("/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/Gammas_0p1_to_120.dat","w");
     }
     for(int i=0;i<steps;i++){
         for(int j=0;j<usteps;j++){
             for(int k=0;k<vsteps;k++){
                 int id_x;
                 id_x = i*usteps*vsteps+j*vsteps+k;

                 fprintf(fp,"%d %d %d %e %e %e %e %e %e %e %e \n",i ,j ,k ,real(result_gamma0[id_x]),imag(result_gamma0[id_x]),
                 real(result_gamma1[id_x]),imag(result_gamma1[id_x]),real(result_gamma2[id_x]),imag(result_gamma2[id_x]),
                 real(result_gamma3[id_x]),imag(result_gamma3[id_x]));
             }
         }
     }
     fclose(fp);
     printf("Done.\n");
}