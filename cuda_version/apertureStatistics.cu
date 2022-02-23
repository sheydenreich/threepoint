#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "cubature.h"

#include <iostream>
#include <algorithm>

__device__ double uHat(double eta)
{
  double temp=0.5*eta*eta;
  return temp*exp(-temp);
}

__device__ double uHat_product(const double& l1, const double& l2, const double& l3, double* thetas)
{
  return uHat(l1*thetas[0])*uHat(l2*thetas[1])*uHat(l3*thetas[2]);
}

__device__ double uHat_product_permutations(const double& l1, const double& l2, const double& l3, double* thetas)
{
  double result;
  result = uHat_product(l1,l2,l3,thetas);
  result += uHat_product(l2,l3,l1,thetas);
  result += uHat_product(l3,l1,l2,thetas);
  result += uHat_product(l1,l3,l2,thetas);
  result += uHat_product(l3,l2,l1,thetas);
  result += uHat_product(l2,l1,l3,thetas);
  return result;
}

__device__ double dev_integrand_Map2(const double& ell, const double& z, double theta)
{
  return ell*pow(uHat(ell*theta),2)*(limber_integrand(ell,z)+0.5*dev_sigma*dev_sigma/dev_n/dev_z_max);
}

__global__ void integrand_Map2(const double* vars, unsigned ndim, int npts, double theta, double* value)
{
  // index of thread
  int thread_index=blockIdx.x*blockDim.x + threadIdx.x;

  //Grid-Stride loop, so I get npts evaluations
  for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
      double ell=vars[i*ndim];
      double z=vars[i*ndim+1];

      value[i] = dev_integrand_Map2(ell,z,theta);
    }
}

int integrand_Map2(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim != 1)
    {
      std::cerr<<"integrand: Wrong number of function dimensions"<<std::endl;
      exit(1);
    };
    if(ndim != 2)
    {
      std::cerr<<"integrand: Wrong number of variable dimensions"<<std::endl;
      exit(1);
    };

  // Read data for integration
  ApertureStatisticsContainer* container = (ApertureStatisticsContainer*) thisPtr;

  double theta = container-> thetas.at(0);

  // Allocate memory on device for integrand values
  double* dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

  // Copy integration variables to device
  double* dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //allocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

  // Calculate values
  integrand_Map2<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta, dev_value);
  CudaCheckError();
  
  cudaFree(dev_vars); //Free variables

  
  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));


  cudaFree(dev_value); //Free values
  
  return 0; //Success :)  
}

double Map2(double theta)
{
  //Set maximal l value such, that theta*l <= 10
  double lMax=10./theta;
  double lMin = 1.;

  ApertureStatisticsContainer container;
  std::vector<double> thetas{theta};
  container.thetas=thetas;

  double result,error;

  double vals_min[2]={lMin, 0};
  double vals_max[2]={lMax, z_max};

  hcubature_v(1, integrand_Map2, &container, 2, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
  
  return result/2./M_PI;
}





__global__ void integrand_Map3_kernel(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double* value)
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
//      printf("%lf, %lf, %lf, %lf\n", l1, l2, l3, bkappa(l1, l2, l3));
    }
}


int integrand_Map3(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim != 1)
    {
      std::cerr<<"integrand: Wrong number of function dimensions"<<std::endl;
      exit(1);
    };
  // Read data for integration
  ApertureStatisticsContainer* container = (ApertureStatisticsContainer*) thisPtr;

  double theta1 = container-> thetas.at(0);
  double theta2 = container-> thetas.at(1);
  double theta3 = container-> thetas.at(2);

  // Allocate memory on device for integrand values
  double* dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

  // Copy integration variables to device
  double* dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

  // Calculate values
  integrand_Map3_kernel<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta1, theta2, theta3, dev_value);

  cudaFree(dev_vars); //Free variables
  
  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); //Free values
  
  return 0; //Success :)  
  

}


double MapMapMap(const std::vector<double>& thetas, const double& phiMin, const double& phiMax, const double& lMin)
{
  //Set maximal l value such, that theta*l <= 10
  double thetaMin=std::min({thetas[0], thetas[1], thetas[2]});
  double lMax=10./thetaMin;


  ApertureStatisticsContainer container;
  container.thetas=thetas;
  double result,error;

  double vals_min[3]={lMin, lMin, phiMin};
  double vals_max[3]={lMax, lMax, phiMax};

 
  hcubature_v(1, integrand_Map3, &container, 3, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);

  
  return result/8/M_PI/M_PI/M_PI;//Divided by (2*pi)³
}
