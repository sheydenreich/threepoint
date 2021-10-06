#include "gamma.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#define slics false


int main(int argc, char** argv)
{
  std::cout << "Executing " << argv[0] << " ";
  if(argc>=2)
    {
      int deviceNumber = atoi(argv[1]);
      std::cout << "on GPU " << deviceNumber << std::endl;
      cudaSetDevice(deviceNumber);
    }
  else
    {
      std::cout << "on default GPU";
    };
  

  std::string cosmo_paramfile, outfn;
  if(slics)
    {
      // Set Up Cosmology
      cosmo_paramfile="SLICS_cosmo.dat";
      // Set output file
      outfn="../../results_SLICS/Gammas_0p1_to_60_10_15_15_bins.dat";
    }
  else
    {
      // Set Up Cosmology
      cosmo_paramfile="MR_cosmo.dat";
      // Set output file
      outfn="../../results_MR/Gammas_0p1_to_60_10_15_15_bins.dat";
    };
  
  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);
  
  // Check output file
  std::ofstream out;
  out.open(outfn.c_str());
  if(!out.is_open())
    {
      std::cerr<<"Couldn't open "<<outfn<<std::endl;
      exit(1);
    };

  // User output
  std::cerr<<"Using cosmology:"<<std::endl;
  std::cerr<<cosmo;
  std::cerr<<"Writing to:"<<outfn<<std::endl;



  // Binning
  int steps = 10;
  int usteps = 15;
  int vsteps = 15;

  double rmin = 0.1;
  double rmax = 70.;
  double umin = 0;
  double umax = 1;
  double vmin = 0;
  double vmax = 1;

  double dz = cosmo.zmax / ((double) n_redshift_bins);

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A96,&A96,48*sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_W96,&W96,48*sizeof(double)));

  cosmo.zmax=z_max;
  set_cosmology(cosmo, dz);

  compute_weights_bessel();

  //Calculation + Output in one
  double lrmin = log(rmin);
  double lrmax = log(rmax);
  
  auto begin=std::chrono::high_resolution_clock::now(); //Begin time measurement

  
  for(int i=0; i<steps; i++)
    {
      double r=exp(lrmin+(lrmax-lrmin)/steps*(i+0.5));
      for(int j=0; j<usteps; j++)
	{
	  double u = umin+(umax-umin)/usteps*(j+0.5);
    // double u;
    // if(j<10) u = 0.2/10*(j+0.5);
    // else u = 1./10*((j-8)+0.5);

	  for(int k=0; k<vsteps; k++)
	    {
	      double v= vmin+(vmax-vmin)/vsteps*(k+0.5);

	      double r2 = r*M_PI/180./60.; //THIS IS THE BINNING BY JARVIS. FROM THE WEBSITE, NOT THE PAPER.
	      double r3 = r2*u;
	      double r1 = v*r3+r2;
	      
	      std::complex<double> _gamma0 = gamma0(r1, r2, r3,z_max);
	      std::complex<double> _gamma1 = gamma1(r1, r2, r3,z_max);
	      std::complex<double> _gamma2 = gamma2(r1, r2, r3,z_max);
	      std::complex<double> _gamma3 = gamma3(r1, r2, r3,z_max);
	      auto end = std::chrono::high_resolution_clock::now();
	      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        int completed_steps = i*usteps*vsteps+j*vsteps+k+1;
        int total_steps = steps*usteps*vsteps;
        double progress = (completed_steps*1.)/(total_steps);

	      printf("\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step.",
        static_cast<int>(progress*100),
        elapsed.count()*1e-9/3600,
        (total_steps-completed_steps)*elapsed.count()*1e-9/3600/completed_steps,
        elapsed.count()*1e-9/completed_steps);
	      out
        <<i<<" "
        <<j<<" "
        <<k<<" "
        <<real(_gamma0)<<" "
        <<imag(_gamma0)<<" "
        <<real(_gamma1)<<" "
        <<imag(_gamma1)<<" "
        <<real(_gamma2)<<" "
        <<imag(_gamma2)<<" "
        <<real(_gamma3)<<" "
        <<imag(_gamma3)<<" "
        <<r<<" "
        <<u<<" "
        <<v<<     
        std::endl;
	    };
	};
    };


  
  
}
