/**
 * @file testGamma_analytical_bispectrum.cpp
 * This executable calculates <Gamma^0> for predefined sidelengths from an
 * analytical bispectrum, and the corresponding values that Gamma^0 should take
 * @warning Make sure test_analytical in bispectrum.hpp is set to true 
 * @author Sven Heydenreich
 */


#include "bispectrum.hpp"
#include "gamma.hpp"
#include <omp.h>

#define RECOMPUTE_GAMMA true
#define DO_SANITY_CHECKS false
#define DO_PERMUTATION_CHECKS false

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


    // int steps = 10;
    // int usteps = 10;
    // int vsteps = 11;
   	// double r_array[steps] = {0.65763, 1.1376, 1.968, 3.4044, 5.8893, 10.188, 17.624, 30.488, 52.741, 91.237};
   	// double u_array[usteps] = {0.05  ,  0.15 ,   0.25  ,  0.39229 , 0.45  ,  0.55  ,  0.63257,  0.79057,  0.89457,  0.95};
   	// double v_array[vsteps] = {0.045455 ,  0.13636 ,  0.22727 ,  0.31818 ,  0.40909  , 0.5     ,  0.59091 , 0.68182  , 0.77273 ,  0.86364 , 0.95455};
    int steps = 15;
    int usteps = 15;
    int vsteps = 15;
    double* r_array = new double[steps];
    double* u_array = new double[usteps];
    double* v_array = new double[vsteps];

    double rmin = 0.1;
    double rmax = 300.;
    double umin = 0;
    double umax = 1;
    double vmin = 0;
    double vmax = 1;

    double lrmin = log(rmin);
    double lrmax = log(rmax);

    for(int i=0;i<steps;i++)
    {
      double temp = lrmin+(lrmax-lrmin)/steps*(i+0.5);
      r_array[i] = exp(temp);
    }
    for(int i=0;i<usteps;i++)
    {
      u_array[i] = umin+(umax-umin)/usteps*(i+0.5);
    }
    for(int i=0;i<vsteps;i++)
    {
      v_array[i] = vmin+(vmax-vmin)/vsteps*(i+0.5);
    }

    BispectrumCalculator Bispectrum(cosmo,10,1.,false);
    GammaCalculator class_gamma(&Bispectrum, 0.01, 3.5, "centroid");

    static std::complex<double>* result_gamma0 = new std::complex<double>[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    static std::complex<double>* result_gamma1 = new std::complex<double>[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    static std::complex<double>* result_gamma2 = new std::complex<double>[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    static std::complex<double>* result_gamma3 = new std::complex<double>[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    static std::complex<double>* result_ggg = new std::complex<double>[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    static std::complex<double>* result_gstargg = new std::complex<double>[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    static std::complex<double>* result_ggstarg = new std::complex<double>[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
    static std::complex<double>* result_gggstar = new std::complex<double>[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")

    // #pragma omp parallel for collapse(3)
    printf("Computing gammas...[100%%]");
    for(int i=0;i<steps;i++){
        for(int j=0;j<usteps;j++){
            for(int k=0;k<vsteps;k++){
            printf("\b\b\b\b\b\b[%3d%%]",static_cast<int>(100*(1.*i*usteps*vsteps+j*vsteps+k)/(usteps*vsteps*(steps))));
        		fflush(stdout);


                double r,u,v;

                r = r_array[i]; 
                u = u_array[j];
                v = v_array[k];

                    int id_x = i*usteps*vsteps+j*vsteps+k;

                    double r2t = r*M_PI/(60*180.); //THIS IS THE BINNING BY JARVIS. FROM THE WEBSITE, NOT THE PAPER.
                    double r3t = r2t*u;
                    double r1t = v*r3t+r2t;

                    double r1 = r1t;
                    double r2 = r2t;
                    double r3 = r3t;

                    if(RECOMPUTE_GAMMA)
                    {
                      result_gamma0[id_x] = class_gamma.gamma0(r1,r2,r3);
                      result_gamma1[id_x] = class_gamma.gamma1(r1,r2,r3);
                      result_gamma2[id_x] = class_gamma.gamma2(r1,r2,r3);
                      result_gamma3[id_x] = class_gamma.gamma3(r1,r2,r3);
                    }


                    result_ggg[id_x] = class_gamma.ggg(r1,r2,r3);
                    result_gstargg[id_x] = class_gamma.gstargg(r1,r2,r3);
                    result_ggstarg[id_x] = class_gamma.ggstarg(r1,r2,r3);
                    result_gggstar[id_x] = class_gamma.gggstar(r1,r2,r3);

                    // printf("[%d, %d, %d, %.4e, %.4e, %.4e, %.4e],",i,j,k,
                    //         real(res_temp),imag(res_temp),real(comp_temp),imag(comp_temp));

                }
            }
        }
    printf("\n Done. Writing results. \n");

    FILE *fp;

    if(RECOMPUTE_GAMMA)
    {
    
      if(test_analytical)
      {
        fp = fopen("/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/Gamma_15.dat","w");
      }

      for(int i=0;i<steps;i++){
          for(int j=0;j<usteps;j++){
              for(int k=0;k<vsteps;k++){
                  int id_x;
                  id_x = i*usteps*vsteps+j*vsteps+k;

                  fprintf(fp,"%d %d %d %e %e %e %e %e %e %e %e \n",i ,j ,k ,real(result_gamma0[id_x]),imag(result_gamma0[id_x]),real(result_gamma1[id_x]),imag(result_gamma1[id_x]),
                  real(result_gamma2[id_x]),imag(result_gamma2[id_x]),real(result_gamma3[id_x]),imag(result_gamma3[id_x]));
              }
          }
      }
      fclose(fp);
    }

      if(test_analytical)
     {
        fp = fopen("/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_analytic/ggg_15.dat","w");
     }

      for(int i=0;i<steps;i++){
         for(int j=0;j<usteps;j++){
             for(int k=0;k<vsteps;k++){
                 int id_x;
                 id_x = i*usteps*vsteps+j*vsteps+k;

                 fprintf(fp,"%d %d %d %e %e %e %e %e %e %e %e \n",i ,j ,k ,real(result_ggg[id_x]),imag(result_ggg[id_x]),real(result_gstargg[id_x]),imag(result_gstargg[id_x]),
                 real(result_ggstarg[id_x]),imag(result_ggstarg[id_x]),real(result_gggstar[id_x]),imag(result_gggstar[id_x]));
             }
         }
     }
     fclose(fp);


    if(DO_SANITY_CHECKS)
    {
      printf("Performing test calculations.  \n");
      double r = r_array[static_cast<int>(0.7*steps)]*M_PI/(60*180.);
      for(int i=0;i<2;i++)
      {
        double r2 = (0.5+i)*r;
        std::complex<double> gamma0 = class_gamma.gamma0(r,r,r2);
        // printf("blub0 \n");
        std::complex<double> gamma1 = class_gamma.gamma1(r,r,r2);
        // printf("blub1 \n");
        std::complex<double> gamma2 = class_gamma.gamma2(r,r,r2);
        // printf("blub2 \n");
        std::complex<double> gamma3 = class_gamma.gamma3(r,r,r2);
        // printf("blub3 \n");

        std::complex<double> ggg = class_gamma.ggg(r,r,r2);
        std::complex<double> gstargg = class_gamma.gstargg(r,r,r2);
        std::complex<double> ggstarg = class_gamma.ggstarg(r,r,r2);
        std::complex<double> gggstar = class_gamma.gggstar(r,r,r2);

        std::cout << "These should be zero: Imag(Gamma0, Gamma3, ggg, gggstar): \n" << imag(gamma0) << ", " << imag(gamma3) << ", " << imag(ggg) << ", " << imag(gggstar) << std::endl;
        std::cout << "These should have opposite sign and equal magnitude: Imag(Gamma1, Gamma2, gstargg, ggstarg): \n" << imag(gamma1) << ", " << imag(gamma2) << ", " << imag(gstargg) << ", " << imag(ggstarg) << std::endl;
      }
    }
    if(DO_PERMUTATION_CHECKS)
    {
      {
        double r1 = r_array[7]*M_PI/(60*180.);
        double r2 = r1*0.8;
        double r3 = r1*0.6;

        std::complex<double> gamma01 = class_gamma.gamma0(r1,r2,r3);
        std::complex<double> gamma02 = class_gamma.gamma0(r2,r3,r1);
        std::complex<double> gamma03 = class_gamma.gamma0(r3,r1,r2);
        std::complex<double> ggg1 = class_gamma.ggg(r1,r2,r3);
        std::complex<double> ggg2 = class_gamma.ggg(r2,r3,r1);
        std::complex<double> ggg3 = class_gamma.ggg(r3,r1,r2);

        std::cout << "Gamma^0(x1,x2,x3);Gamma^0(x2,x3,x1);Gamma^0(x3,x1,x2): \n" <<
        real(gamma01) << "+" << imag(gamma01) << "i; " <<
        real(gamma02) << "+" << imag(gamma02) << "i; " <<
        real(gamma03) << "+" << imag(gamma03) << "i; \n" <<
        "ggg(x1,x2,x3);ggg(x2,x3,x1);ggg(x3,x1,x2): \n" <<
        real(ggg1) << "+" << imag(ggg1) << "i; " <<
        real(ggg2) << "+" << imag(ggg2) << "i; " <<
        real(ggg3) << "+" << imag(ggg3) << "i; \n" << std::endl;

        std::complex<double> gamma1 = class_gamma.gamma1(r1,r2,r3);
        std::complex<double> gamma2 = class_gamma.gamma2(r3,r1,r2);
        std::complex<double> gamma3 = class_gamma.gamma3(r2,r3,r1);

        std::complex<double> gstargg = class_gamma.gstargg(r1,r2,r3);
        std::complex<double> ggstarg = class_gamma.ggstarg(r3,r1,r2);
        std::complex<double> gggstar = class_gamma.gggstar(r2,r3,r1);

        std::cout << "Gamma^1(x1,x2,x3);Gamma^2(x3,x1,x2);Gamma^3(x2,x3,x1): \n" <<
        real(gamma1) << "+" << imag(gamma1) << "i; " <<
        real(gamma2) << "+" << imag(gamma2) << "i; " <<
        real(gamma3) << "+" << imag(gamma3) << "i; \n" <<
        "gstargg(x1,x2,x3);ggstarg(x3,x1,x2);gggstar(x2,x3,x1): \n" <<
        real(gstargg) << "+" << imag(gstargg) << "i; " <<
        real(ggstarg) << "+" << imag(ggstarg) << "i; " <<
        real(gggstar) << "+" << imag(gggstar) << "i; \n" << std::endl;
      }

      {
        double r1 = r_array[7]*M_PI/(60*180.);
        double r2 = r1*0.95;
        double r3 = r1*0.1;

        std::complex<double> gamma1 = class_gamma.gamma1(r1,r2,r3);
        std::complex<double> gamma2 = class_gamma.gamma1(r2,r3,r1);
        std::complex<double> gamma3 = class_gamma.gamma1(r3,r1,r2);
        std::complex<double> ggg1 = class_gamma.gstargg(r1,r2,r3);
        std::complex<double> ggg2 = class_gamma.gstargg(r2,r3,r1);
        std::complex<double> ggg3 = class_gamma.gstargg(r3,r1,r2);

        std::cout << "Gamma^1(x1,x2,x3);gstargg(x1,x2,x3)\n" <<
        real(gamma1) << "+" << imag(gamma1) << "i; \n" <<
        real(ggg1) << "+" << imag(ggg1) << "i; \n" << 
        "Gamma^1(x2,x3,x1);gstargg(x3,x1,x2)\n" <<
        real(gamma2) << "+" << imag(gamma2) << "i; \n" <<
        real(ggg2) << "+" << imag(ggg2) << "i; \n" << 
        "Gamma^1(x3,x1,x2);gstargg(x3,x1,x2)\n" <<
        real(gamma3) << "+" << imag(gamma3) << "i; \n" <<
        real(ggg3) << "+" << imag(ggg3) << "i; \n" << 
        std::endl;

        gamma1 = class_gamma.gamma1(r1*1.001,r2*0.999,r3*0.999);
        ggg1 = class_gamma.gstargg(r1*1.001,r2*0.999,r3*0.999);

        std::cout << "Gamma^1(x1,x2,x3);gstargg(x1,x2,x3)\n" <<
        real(gamma1) << "+" << imag(gamma1) << "i; \n" <<
        real(ggg1) << "+" << imag(ggg1) << "i; \n" << 
        std::endl;

        gamma1 = class_gamma.gamma1(r1*0.999,r2*1.001,r3*0.999);
        ggg1 = class_gamma.gstargg(r1*0.999,r2*1.001,r3*0.999);

        std::cout << "Gamma^1(x1,x2,x3);gstargg(x1,x2,x3)\n" <<
        real(gamma1) << "+" << imag(gamma1) << "i; \n" <<
        real(ggg1) << "+" << imag(ggg1) << "i; \n" << 
        std::endl;


      }

    }




}
