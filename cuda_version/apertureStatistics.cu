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

__device__ double dev_integrand_Gaussian_Aperture_Covariance(const double& l1, const double& l2, const double& phi, const double& z, 
  double* thetas_123, double* thetas_456, const double& shapenoise_contribution)
{
double l3 = sqrt(l1*l1+l2*l2+2*l1*l2*cos(phi));
double result = uHat_product(l1, l2, l3, thetas_123);
result *= uHat_product_permutations(l1, l2, l3, thetas_456);
result *= limber_integrand_triple_power_spectrum(l1,l2,l3,z,shapenoise_contribution);
result *= l1*l2;

// printf("%.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e: %.3e\n",l1,l2,phi,
// thetas_123[0],thetas_123[1],thetas_123[2],
// thetas_456[0],thetas_456[1],thetas_456[2],
// result);
return result;
}

__global__ void integrand_Gaussian_Map3_Covariance(const double* vars, unsigned ndim, int npts, double* thetas_123, double* thetas_456, double* value, double shapenoise_contribution)
{
  // index of thread
  int thread_index=blockIdx.x*blockDim.x + threadIdx.x;

  //Grid-Stride loop, so I get npts evaluations
  for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
      // printf("%d, %d \n",npts*ndim-i*ndim-3,npts-i);
      double l1=vars[i*ndim];
      double l2=vars[i*ndim+1];
      double phi=vars[i*ndim+2];
      double z=vars[i*ndim+3];

      value[i] = dev_integrand_Gaussian_Aperture_Covariance(l1,l2,phi,z,thetas_123,thetas_456,shapenoise_contribution);
      // printf("%.3e, %.3e, %.3f, %.3f: %.3e \n",l1,l2,phi,z,value[i]);
    }
  //   printf("p %.3e %.3e %.3e %.3e %.3e %.3e: %.3e \n",
  //   thetas_123[0],thetas_123[1],thetas_123[2],
  //   thetas_456[0],thetas_456[1],thetas_456[2],
  //   // dev_integrand_Gaussian_Aperture_Covariance(1e+4, 1e+4, 1, 1, thetas_123, thetas_456)
  //   // limber_integrand_triple_power_spectrum(1e+4,1e+4,1e+4,1.)
  //   // dev_E(1.)
  //   // limber_integrand_prefactor(1, 0.5)
  //   // P_k_nonlinear(0.1, 1.5)
  //   // dev_linear_pk(0.1)
  // );
}

int integral_Gaussian_Map3_Covariance(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim != 1)
    {
      std::cerr<<"integrand: Wrong number of function dimensions"<<std::endl;
      exit(1);
    };
    if(ndim != 4)
    {
      std::cerr<<"integrand: Wrong number of variable dimensions"<<std::endl;
      exit(1);
    };

  // Read data for integration
  ApertureStatisticsCovarianceContainer* container = (ApertureStatisticsCovarianceContainer*) thisPtr;

  double* thetas_123 = container-> thetas_123;
  double* thetas_456 = container-> thetas_456;
  double shapenoise_powerspectrum = container -> shapenoise_powerspectrum;

  // Allocate memory on device for integrand values
  double* dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

  // Copy integration variables to device
  double* dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //allocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

  double* dev_thetas_123;
  double* dev_thetas_456;
  CUDA_SAFE_CALL(cudaMalloc(&dev_thetas_123, 3*sizeof(double))); //allocate memory
  CUDA_SAFE_CALL(cudaMalloc(&dev_thetas_456, 3*sizeof(double))); //allocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_thetas_123, thetas_123, 3*sizeof(double), cudaMemcpyHostToDevice)); //copying
  CUDA_SAFE_CALL(cudaMemcpy(dev_thetas_456, thetas_456, 3*sizeof(double), cudaMemcpyHostToDevice)); //copying


  // Calculate values
  integrand_Gaussian_Map3_Covariance<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, dev_thetas_123, dev_thetas_456, dev_value,shapenoise_powerspectrum);
  // std::cerr << "test " << npts << std::endl;
  CudaCheckError();
  
  cudaFree(dev_vars); //Free variables
  cudaFree(dev_thetas_123); //Free variables
  cudaFree(dev_thetas_456); //Free variables

  
  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

  // std::cerr << value[5] << std::endl;

  cudaFree(dev_value); //Free values
  
  return 0; //Success :)  
  

}


double Gaussian_MapMapMap_Covariance(double* thetas_123, double* thetas_456, const covarianceParameters covPar, bool shapenoise)
{
  //Set maximal l value such, that theta*l <= 10
  double thetaMin_123=std::min({thetas_123[0], thetas_123[1], thetas_123[2]});
  double thetaMin_456=std::min({thetas_456[0], thetas_456[1], thetas_456[2]});
  double thetaMin=std::max({thetaMin_123,thetaMin_456}); //should increase runtime, if either theta_123 or theta_456 is zero, so is their product
  double lMax=10./thetaMin;
  double lMin = 1.;
  double phiMin = 0;
  double phiMax = 2.*M_PI;

  ApertureStatisticsCovarianceContainer container;
  container.thetas_123=thetas_123;
  container.thetas_456=thetas_456;
  double shapenoise_powerspectrum;
  if(shapenoise)
    shapenoise_powerspectrum = covPar.power_spectrum_contribution;
  else
    shapenoise_powerspectrum = 0.;

  container.shapenoise_powerspectrum = shapenoise_powerspectrum;

  double result,error;

  double vals_min[4]={lMin, lMin, phiMin, 0};
  double vals_max[4]={lMax, lMax, phiMax/2., z_max}; //use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

  hcubature_v(1, integral_Gaussian_Map3_Covariance, &container, 4, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
  
  double survey_area = covPar.survey_area*pow(M_PI/180.,2);
  return 2.*result/survey_area/8/M_PI/M_PI/M_PI;
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

  

