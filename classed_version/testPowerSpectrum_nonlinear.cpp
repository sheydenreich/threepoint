/**
 * @file testPowerSpectrum_nonlinear.cpp
 * This executable gives the linear and non-linear power spectra 
 * used for debugging
 * @author Sven Heydenreich
 * @warning output is printed to console
 */

#include "bispectrum.hpp"
#include <fstream>

double integrate_limber(double ell, double dz, double z_max, BispectrumCalculator bispectrum, bool nonlinear)
{
  double integration_result = 0;
  for(double z=z_max;z>=0;z-=dz)
  {
    integration_result += bispectrum.limber_integrand_power_spectrum(ell, z, nonlinear)*dz;
  }
  return integration_result;
}


int main()
{
  // Set up Cosmology
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

  int n_z=200; //Number of redshift bins for grids
  double z_max=2; //maximal redshift
  bool fastCalc=false; //whether calculations should be sped up
  BispectrumCalculator bispectrum(cosmo, n_z, z_max, fastCalc);

    printf("[");
    for(double k=1e-5;k<10;k*=1.1)
    {
        printf("[%.4e,%.4e,%.4e],",k,bispectrum.linear_pk_at_z(k,1.),bispectrum.P_k_nonlinear(k,1.));
    }
    printf("\b]\n");

    printf("[");
    for(double ell=1e+1;ell<=1e+5;ell*=1.1)
    {
        printf("[%.4e,%.4e,%.4e],",ell,ell*(ell+1)*integrate_limber(ell, 0.025, z_max, bispectrum, false)/(2*M_PI),
                                  ell*(ell+1)*integrate_limber(ell, 0.025, z_max, bispectrum, true)/(2*M_PI));
    }
    printf("\b]\n");
    // integrate_limber(10, 0.025, z_max, bispectrum, true)/(2*M_PI);
    // bispectrum.P_k_nonlinear(0.1, 0);


    return 0;
}