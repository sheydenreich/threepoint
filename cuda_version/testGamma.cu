#include "gamma.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#define slics false


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
    };

  if(slics) z_max = 3.;
  else z_max = 1.1;
  double dz = z_max / ((double) n_redshift_bins);

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A96,&A96,48*sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_W96,&W96,48*sizeof(double)));

  cosmo.zmax=z_max;
  set_cosmology(cosmo, dz);

  compute_weights_bessel();

  double r_arr[10] = {0.15423, 0.34705, 0.69582, 1.408, 2.8577, 5.8013, 11.775, 23.875, 48.156, 95.819};
  double u = 0.95;
  double v = 0.05;

  for(int i=0;i<10;i++)
  {

    double r2 = r_arr[i]; //THIS IS THE BINNING BY JARVIS. FROM THE WEBSITE, NOT THE PAPER.
    double r3 = r2*u;
    double r1 = v*r3+r2;

    std::cout << r1 << ", " << r2 << ", " << r3 << ", " << std::endl;

    auto begin=std::chrono::high_resolution_clock::now(); //Begin time measurement

    std::complex<double> gamma = gamma1(M_PI/180./60.*r1, M_PI/180./60.*r2, M_PI/180./60.*r3,z_max);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    std::cout<<"Time needed for last point:"<<elapsed.count()*1e-9<<std::endl;
    std::cout
    <<real(gamma)<<" "
    <<imag(gamma)<<std::endl;

  }

}
