/**
 * @file testGamma_integrand.cpp
 * This executable calculates several tests for the integrand of the Gamma^0 integration.
 * @author Sven Heydenreich
 * @warning depending on the tests, output is written to console
 */

#include "gamma.hpp"
#include "helper.hpp"
#include <omp.h>

int main()
{

  struct cosmology cosmo;

  if (slics)
  {
    printf("using SLICS cosmology...");
    cosmo.h = 0.6898;               // Hubble parameter
    cosmo.sigma8 = 0.826;           // sigma 8
    cosmo.omb = 0.0473;             // Omega baryon
    cosmo.omc = 0.2905 - cosmo.omb; // Omega CDM
    cosmo.ns = 0.969;               // spectral index of linear P(k)
    cosmo.w = -1.0;
    cosmo.om = cosmo.omb + cosmo.omc;
    cosmo.ow = 1 - cosmo.om;
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
    cosmo.om = cosmo.omc + cosmo.omb;
    cosmo.ow = 1. - cosmo.om;
  }

  // BispectrumCalculator class_bkappa(cosmo,200,2,false);
  // double phi = M_PI;
  // double ell1 = 100;
  // double ell2 = 100;
  // double ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));

  // printf("data1 = [");
  // while(ell3>=1.)
  // {
  //     ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));
  //     printf("[%.4e,%.4e,%.4e],",phi,ell3,class_bkappa.bkappa(ell1,ell2,ell3));
  //     phi /= 1.05;
  // }
  // printf("\b ] \n");

  //  phi = M_PI;
  //  ell1 = 1000;
  //  ell2 = 1000;
  //  ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));

  // printf("data2 = [");
  // while(ell3>=1.)
  // {
  //     ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));
  //     printf("[%.4e,%.4e,%.4e],",phi,ell3,class_bkappa.bkappa(ell1,ell2,ell3));
  //     phi /= 1.05;
  // }
  // printf("\b ] \n");

  //  phi = M_PI;
  //  ell1 = 10000;
  //  ell2 = 10000;
  //  ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));

  // printf("data3 = [");
  // while(ell3>=1.)
  // {
  //     ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));
  //     printf("[%.4e,%.4e,%.4e],",phi,ell3,class_bkappa.bkappa(ell1,ell2,ell3));
  //     phi /= 1.05;
  // }
  // printf("\b ] \n");

  // double phi = 2*M_PI/198;
  // printf("data_bkappa = [");
  // for(int i=0;i<400;i++)
  // {
  //     double ell1 = pow(10,(2+i*2./400));
  //     double ell2 = ell1;
  //     double ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));
  //     printf("[%.4e,%.4e],",ell1,class_bkappa.bkappa(ell1,ell2,ell3));
  // }
  // printf("\b ] \n");

  // double ell = 100;
  // printf("data_bkappa_phi = [");
  // for(int i=0;i<400;i++)
  // {
  //     double phi = (i+0.5)*2*M_PI/400;
  //     double ell1 = ell;
  //     double ell2 = ell1;
  //     double ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));
  //     printf("[%.4e,%.4e],",phi,class_bkappa.bkappa(ell1,ell2,ell3));
  // }
  // printf("\b ] \n");

  // GammaCalculator class_gamma(cosmo, 0.05, 3.5, false, 200, 2);
  // double x1 = 10.*M_PI/(60*180.);
  // double x2 = x1;
  // double x3 = x1/2;
  // double psi = M_PI/4.;
  // printf("data_squeezed = [");
  // for(double phi = M_PI;phi>1.0e-6;phi/=1.05){
  //         // double phi = (i+0.5)*2*M_PI/(96*2.);
  //         double l3 = sqrt(1-cos(phi));
  //         double temp = class_gamma.r_integral(phi,psi,x1,x2,x3)+class_gamma.r_integral(phi,psi,x2,x3,x1)+class_gamma.r_integral(phi,psi,x3,x1,x2);
  //         printf("[%.4e, %.4e, %.4e],",l3,phi,temp);
  // }
  // printf("\b ] \n");

  // GammaCalculator class_gamma(cosmo, 0.05, 3.5, false, 200, 2);
  // double results[96*2][96*2];
  // double x1 = 10.*M_PI/(60*180.);
  // double x2 = x1;
  // double x3 = x1/2;
  // #pragma omp parallel for collapse(2)
  // for(int i=0;i<96*2;i++){
  //     for(int j=0;j<96*2;j++){
  //         double phi = (i+0.5)*2*M_PI/(96*2.);
  //         double psi = (j+0.5)*M_PI/2./(96*2.);
  //         // printf("%d %d \n",i,j);
  //         double temp = class_gamma.r_integral(phi,psi,x1,x2,x3)+class_gamma.r_integral(phi,psi,x2,x3,x1)+class_gamma.r_integral(phi,psi,x3,x1,x2);

  //         results[i][j] = temp;
  //     }
  // }

  // FILE *fp;
  // fp = fopen("../tests/integrated_bispectrum_10_squeezed_tree.dat","w");
  // for(int i=0;i<96*2;i++){
  //     for(int j=0;j<96*2;j++){
  //         fprintf(fp, "%.4e\t",results[i][j]);
  //     }
  //     fprintf(fp,"\n");
  // }
  // fclose(fp);

  // GammaCalculator class_gamma(cosmo, 0.05, 3.5, false, 200, 2);
  // std::complex<double> results[96*2][96*2];
  // double x1 = 10.*M_PI/(60*180.);
  // double x2 = x1;
  // double x3 = x1/2;
  // #pragma omp parallel for collapse(2)
  // for(int i=0;i<96*2;i++){
  //     for(int j=0;j<96*2;j++){
  //         double phi = (i+0.5)*2*M_PI/(96*2.);
  //         double psi = (j+0.5)*M_PI/2./(96*2.);
  //         // printf("%d %d \n",i,j);
  //         results[i][j] = class_gamma.integrand_phi_psi(phi,psi,x1,x2,x3)+class_gamma.integrand_phi_psi(phi,psi,x2,x3,x1)+class_gamma.integrand_phi_psi(phi,psi,x3,x1,x2);
  //     }
  // }

  // FILE *fp;
  // fp = fopen("../tests/integrand_phi_psi_squeezed_3_gauss_10_real.dat","w");
  // for(int i=0;i<96*2;i++){
  //     for(int j=0;j<96*2;j++){
  //         fprintf(fp, "%.4e\t",real(results[i][j]));
  //     }
  //     fprintf(fp,"\n");
  // }
  // fclose(fp);

  // fp = fopen("../tests/integrand_phi_psi_squeezed_3_gauss_10_imag.dat","w");
  // for(int i=0;i<96*2;i++){
  //     for(int j=0;j<96*2;j++){
  //         fprintf(fp, "%.4e\t ",imag(results[i][j]));
  //     }
  //     fprintf(fp,"\n");
  // }
  // fclose(fp);

  // fp = fopen("../tests/A3_squeezed_2.dat","w");
  // for(int i=0;i<96*2;i++){
  //     for(int j=0;j<96*2;j++){
  //         double phi = (i+0.5)*2*M_PI/(96*2.);
  //         double psi = (j+0.5)*M_PI/2./(96*2.);
  //         fprintf(fp, "%.4e\t ",class_gamma.A(psi,x1,x2,phi,class_gamma.varpsifunc(x1,x2,x3)));
  //     }
  //     fprintf(fp,"\n");
  // }
  // fclose(fp);
}