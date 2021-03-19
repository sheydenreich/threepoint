/**
 * @file testGamma_analytical_bispectrum.cpp
 * This executable calculates <Gamma^0> for predefined sidelengths from an
 * analytical bispectrum, and the corresponding values that Gamma^0 should take
 * @warning Make sure test_analytical in bispectrum.hpp is set to true 
 * @author Sven Heydenreich
 * @warning output is written to console
 */


#include "gamma.hpp"
#include <omp.h>

int main()
{
    
    if(!test_analytical)
    {
        std::cerr << "*********************************************************" << std::endl;
        std::cerr << "WARNING: test_analytical is false, using real bispectrum!" << std::endl;
        std::cerr << "************** Enable it in bispectrum.hpp **************" << std::endl;
        std::cerr << "*********************************************************" << std::endl;
    }

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


    int steps = 10;
    int usteps = 10;
    int vsteps = 11;
   	double r_array[steps] = {0.65763, 1.1376, 1.968, 3.4044, 5.8893, 10.188, 17.624, 30.488, 52.741, 91.237};
   	double u_array[usteps] = {0.05  ,  0.15 ,   0.25  ,  0.39229 , 0.45  ,  0.55  ,  0.63257,  0.79057,  0.89457,  0.95};
   	double v_array[vsteps] = {0.045455 ,  0.13636 ,  0.22727 ,  0.31818 ,  0.40909  , 0.5     ,  0.59091 , 0.68182  , 0.77273 ,  0.86364 , 0.95455};


    GammaCalculator class_gamma(cosmo, 0.05, 3.5, false, 10, 2);

    // #pragma omp parallel for collapse(3)
    for(int i=0;i<steps;i++){
        for(int j=0;j<usteps;j++){
            for(int k=0;k<vsteps;k++){
                double r,u,v;

                r = r_array[i]; 
                u = u_array[j];
                v = v_array[k];

                    double r2 = r*M_PI/(60*180.); //THIS IS THE BINNING BY JARVIS. FROM THE WEBSITE, NOT THE PAPER.
                    double r3 = r2*u;
                    double r1 = v*r3+r2;

                    std::complex<double> res_temp = class_gamma.gamma0_from_cubature(r1, r2, r3);
                    if(isnan(real(res_temp)) || isnan(imag(res_temp))){
                        printf("%lf %lf %lf %lf %lf \n",r1,r2,r3,real(res_temp),imag(res_temp));
                        res_temp = std::complex<double>(0,0);
                    }
                    std::complex<double> X(r1,0);
                    double height = (r1+r2+r3)*(r2+r3-r1)*(r1-r2+r3)*(r1+r2-r3);
                    height = sqrt(height)/(2.*r1);
                    double rest_of_r1 = sqrt(r2*r2-height*height);
                    std::complex<double> Y(rest_of_r1,height);
                    std::complex<double> comp_temp;
                    comp_temp = class_gamma.ggg(Y,X);

                    printf("[%d, %d, %d, %.4e, %.4e, %.4e, %.4e],",i,j,k,
                            real(res_temp),imag(res_temp),real(comp_temp),imag(comp_temp));

                }
            }
        }
    printf("\n Done. \n");
}
