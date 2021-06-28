#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"
#include "cubature.h"

#include <iostream>
#include <algorithm>

__device__ double uHat(double eta)
{
  double temp=0.5*eta*eta;
  return temp*exp(-temp);
}



__global__ void integrand_Map3(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double* value)
{
  // index of thread
  int thread_index=blockIdx.x*blockDim.x + threadIdx.x;

  //Grid-Stride loop, so I get npts evaluations
  for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
      double l1=vars[i*ndim];
      double l2=vars[i*ndim+1];
      double phi=vars[i*ndim+2];

      double l3=sqrt(l1*l1+l2*l2+2*l1*l2*cos(phi));
      value[i]= l1*l2*bkappa(l1, l2, l3)*uHat(l1*theta1)*uHat(l2*theta2)*uHat(l3*theta3); 
    }
}


int integral_Map3(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim != 1)
    {
      std::cerr<<"integrand: Wrong number of function dimensions"<<std::endl;
      exit(1);
    };

  // Read data for integration
  ApertureStatisticsContainer* container = (ApertureStatisticsContainer*) thisPtr;

  double theta1 = container-> theta1;
  double theta2 = container-> theta2;
  double theta3 = container-> theta3;

  // Allocate memory on device for integrand values
  double* dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

  // Copy integration variables to device
  double* dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

  // Calculate values
  integrand_Map3<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta1, theta2, theta3, dev_value);

  cudaFree(dev_vars); //Free variables
  
  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); //Free values
  
  return 0; //Success :)  
  

}


double MapMapMap(double* thetas, const double& phiMin, const double& phiMax, const double& lMin)
{
  //Set maximal l value such, that theta*l <= 10
  double thetaMin=std::min({thetas[0], thetas[1], thetas[2]});
  double lMax=10./thetaMin;


  ApertureStatisticsContainer container;
  container.theta1=thetas[0];
  container.theta2=thetas[1];
  container.theta3=thetas[2];  
  double result,error;

  double vals_min[3]={lMin, lMin, phiMin};
  double vals_max[3]={lMax, lMax, phiMax};
  
  hcubature_v(1, integral_Map3, &container, 3, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);

  
  return result/248.050213442;//Divided by (2*pi)³
}

  

