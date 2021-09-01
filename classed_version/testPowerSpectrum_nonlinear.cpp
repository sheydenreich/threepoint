/**
 * @file testPowerSpectrum_nonlinear.cpp
 * This executable gives the linear and non-linear power spectra 
 * used for debugging
 * @author Sven Heydenreich
 * @warning output is printed to console
 */

#include "bispectrum.hpp"
#include <fstream>

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
  double z_max=4; //maximal redshift
  bool fastCalc=false; //whether calculations should be sped up
  BispectrumCalculator bispectrum(cosmo, n_z, z_max, fastCalc);

    printf("[");
    for(double k=1e-5;k<10;k*=1.1)
    {
        printf("[%.4e,%.4e,%.4e],",k,bispectrum.P_k_nonlinear(k,3.),bispectrum.linear_pk_at_z(k,3.));
    }
    printf("\b]\n");
    return 0;
}