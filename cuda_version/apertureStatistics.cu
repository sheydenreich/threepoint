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

__device__ double dev_integrand_Map2(const double& ell, const double& z, double theta, const double& shapenoise_contribution)
{
  return ell*pow(uHat(ell*theta),2)*(limber_integrand(ell,z)+shapenoise_contribution/dev_z_max);
}

__global__ void integrand_Map2(const double* vars, unsigned ndim, int npts, double theta, double* value, double shapenoise_contribution)
{
  // index of thread
  int thread_index=blockIdx.x*blockDim.x + threadIdx.x;

  //Grid-Stride loop, so I get npts evaluations
  for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
      double ell=vars[i*ndim];
      double z=vars[i*ndim+1];

      value[i] = dev_integrand_Map2(ell,z,theta,shapenoise_contribution);
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
  ApertureStatisticsCovarianceContainer* container = (ApertureStatisticsCovarianceContainer*) thisPtr;

  double theta = container-> theta_1;
  double shapenoise_powerspectrum = container -> shapenoise_powerspectrum;

  // Allocate memory on device for integrand values
  double* dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

  // Copy integration variables to device
  double* dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //allocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

  // Calculate values
  integrand_Map2<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta, dev_value,shapenoise_powerspectrum);
  // std::cerr << "test " << npts << std::endl;
  CudaCheckError();
  
  cudaFree(dev_vars); //Free variables

  
  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

  // std::cerr << value[5] << std::endl;

  cudaFree(dev_value); //Free values
  
  return 0; //Success :)  
}

double Map2(double theta, const covarianceParameters covPar, bool shapenoise)
{
  //Set maximal l value such, that theta*l <= 10
  double lMax=10./theta;
  double lMin = 1.;

  ApertureStatisticsCovarianceContainer container;
  container.theta_1=theta;

  double shapenoise_powerspectrum;
  if(shapenoise)
    shapenoise_powerspectrum = covPar.power_spectrum_contribution;
  else
    shapenoise_powerspectrum = 0.;

  container.shapenoise_powerspectrum = shapenoise_powerspectrum;

  double result,error;

  double vals_min[2]={lMin, 0};
  double vals_max[2]={lMax, z_max};

  hcubature_v(1, integrand_Map2, &container, 2, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
  
  return result/2./M_PI;
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


__device__ double dev_integrand_Gaussian_Map2_covariance(const double& ell, const double& z, 
  const double& theta_1, const double& theta_2, const double& shapenoise_contribution)
  {
    double result = ell*pow(uHat(ell*theta_1),2)*pow(uHat(ell*theta_2),2);
    result *= pow((limber_integrand(ell,z)+shapenoise_contribution/dev_z_max),2);
    return result;
  }

__global__ void integrand_Gaussian_Map2_Covariance(const double* vars, unsigned ndim, int npts, double theta_1, double theta_2,
                                                    double* value, double shapenoise_contribution)
{
  // index of thread
  int thread_index=blockIdx.x*blockDim.x + threadIdx.x;

  //Grid-Stride loop, so I get npts evaluations
  for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
      // printf("%d, %d \n",npts*ndim-i*ndim-3,npts-i);
      double ell=vars[i*ndim];
      double z=vars[i*ndim+1];

      value[i] = dev_integrand_Gaussian_Map2_covariance(ell,z,theta_1,theta_2,shapenoise_contribution);
    }
}

__device__ double G_A(double ell_x, double ell_y, double theta_max)
{
  if(ell_x*theta_max<1e-4 || ell_y*theta_max<1e-4) return 1;
  // return 1;
  double result = sin(ell_x*theta_max/2)*sin(ell_y*theta_max/2)/(ell_x*ell_y*theta_max*theta_max);
  return 16.*pow(result,2);
}

__device__ double dev_integrand_1_for_L2(double ell, double z, double theta_1, double theta_2, double shapenoise_contribution)
{
  return ell*(limber_integrand(ell,z)+shapenoise_contribution/dev_z_max)*uHat(ell*theta_1)*uHat(ell*theta_2);
}

__global__ void integrand_1_for_L2(const double* vars, unsigned ndim, int npts, double theta_1, double theta_2,
  double* value, double shapenoise_contribution)
{
  // index of thread
  int thread_index=blockIdx.x*blockDim.x + threadIdx.x;

  //Grid-Stride loop, so I get npts evaluations
  for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
  {
    // printf("%d, %d \n",npts*ndim-i*ndim-3,npts-i);
    double ell=vars[i*ndim];
    double z=vars[i*ndim+1];

    value[i] = dev_integrand_1_for_L2(ell,z,theta_1,theta_2,shapenoise_contribution);
  }
}

int integrand_1_for_L2(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
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
  ApertureStatisticsCovarianceContainer* container = (ApertureStatisticsCovarianceContainer*) thisPtr;

  double theta_1 = container-> theta_1;
  double theta_2 = container-> theta_2;
  // double theta_max = container -> theta_max;
  double shapenoise_powerspectrum = container -> shapenoise_powerspectrum;

  // Allocate memory on device for integrand values
  double* dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

  // Copy integration variables to device
  double* dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //allocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

  // Calculate values
  integrand_1_for_L2<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta_1, theta_2, dev_value,shapenoise_powerspectrum);
  // std::cerr << "test " << npts << std::endl;
  CudaCheckError();
  
  cudaFree(dev_vars); //Free variables

  
  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); //Free values
  
  return 0; //Success :)  
}


__device__ double integrand_2_for_L2(double ell_x, double ell_y, double theta_1, double theta_2, double theta_max, double shapenoise_contribution)
{
  double ell = sqrt(ell_x*ell_x+ell_y*ell_y);
  return (dev_p_kappa(ell)+shapenoise_contribution)*uHat(ell*theta_1)*uHat(ell*theta_2)*G_A(ell_x,ell_y,theta_max);
}

__global__ void integrand_2_for_L2(const double* vars, unsigned ndim, int npts, double theta_1, double theta_2, double theta_max,
  double* value, double shapenoise_contribution)
{
  // index of thread
  int thread_index=blockIdx.x*blockDim.x + threadIdx.x;
  // if(thread_index==0)
  // {
  //   for(double ell=1e+2;ell<1e+5;ell*=1.1)
  //   {
  //     printf("%f: %.4e \n",ell,ell*(ell+1)/M_PI*dev_p_kappa(ell));
  //   }
  // }
  //Grid-Stride loop, so I get npts evaluations
  for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
  {
    // printf("%d, %d \n",npts*ndim-i*ndim-3,npts-i);
    double ell_x=vars[i*ndim];
    double ell_y=vars[i*ndim+1];
    // double z=vars[i*ndim+2];

    value[i] = integrand_2_for_L2(ell_x,ell_y,theta_1,theta_2,theta_max,shapenoise_contribution);
  }
}

int integrand_2_for_L2(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
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

  // std::cout << npts << std::endl;
  // Read data for integration
  ApertureStatisticsCovarianceContainer* container = (ApertureStatisticsCovarianceContainer*) thisPtr;

  double theta_1 = container-> theta_1;
  double theta_2 = container-> theta_2;
  double theta_max = container -> theta_max;
  double shapenoise_powerspectrum = container -> shapenoise_powerspectrum;

  // Allocate memory on device for integrand values
  double* dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

  // Copy integration variables to device
  double* dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //allocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

  // Calculate values
  integrand_2_for_L2<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta_1, theta_2, theta_max, dev_value,shapenoise_powerspectrum);
  // std::cerr << "test " << npts << std::endl;
  CudaCheckError();
  
  cudaFree(dev_vars); //Free variables

  
  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); //Free values
  
  return 0; //Success :)  
}

double L2(double theta_1, double theta_2, double theta_3, double theta_4, double theta_5, double theta_6,
  const covarianceParameters covPar, bool shapenoise)
{
  //Set maximal l value such, that theta*l <= 10
  double lMin = 1.;
  double shapenoise_powerspectrum;
  double thetaMin, lMax;
  double result,error;
  double result_L2;

  if(shapenoise)
    shapenoise_powerspectrum = covPar.power_spectrum_contribution;
  else
    shapenoise_powerspectrum = 0.;

  ApertureStatisticsCovarianceContainer container;
  container.shapenoise_powerspectrum = shapenoise_powerspectrum;
  container.theta_max = sqrt(covPar.survey_area)*M_PI/180;
  
  // check if L2 has been called for the first time, warn about implementation of theta_max
  static bool runOnce = true;
  if(runOnce)
  {
    std::cerr << "Warning: Assuming a sqare field of sidelength " << container.theta_max*180./M_PI << " deg." << std::endl;
    runOnce = false;
  }


  // Calculate the first term of Eq.(136)
  thetaMin=std::min({theta_1,theta_2}); //should increase runtime, if either theta_123 or theta_456 is zero, so is their product
  lMax=10./thetaMin;

  container.theta_1=theta_1;
  container.theta_2=theta_2;


  double vals_min[2]={lMin, 0};
  double vals_max[2]={lMax, z_max}; //use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

  hcubature_v(1, integrand_1_for_L2, &container, 2, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
  result_L2 = result;


  // Calculate the third term of Eq.(136)
  thetaMin=std::min({theta_5,theta_6}); //should increase runtime, if either theta_123 or theta_456 is zero, so is their product
  lMax=10./thetaMin;

  container.theta_1=theta_5;
  container.theta_2=theta_6;


  double vals_min_2[2]={lMin, 0};
  double vals_max_2[2]={lMax, z_max}; //use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

  hcubature_v(1, integrand_1_for_L2, &container, 2, vals_min_2, vals_max_2, 0, 0, 1e-4, ERROR_L1, &result, &error);
  result_L2 *= result;


  // Calculate the second term of Eq.(136)
  thetaMin=std::min({theta_3,theta_4}); //should increase runtime, if either theta_123 or theta_456 is zero, so is their product
  lMax=10./thetaMin;

  container.theta_1=theta_3;
  container.theta_2=theta_4;


  double vals_min_3[3]={lMin, lMin};
  double vals_max_3[3]={lMax, lMax}; //use symmetry, integrate only from 0 to infty and multiply result by 4 in the end

  hcubature_v(1, integrand_2_for_L2, &container, 2, vals_min_3, vals_max_3, 0, 0, 1e-4, ERROR_L1, &result, &error);
  result_L2 *= 4*result;
  
  return result_L2/pow(2*M_PI,4);
}

double Gaussian_MapMapMap_Covariance_term2(double* thetas_123, double* thetas_456, const covarianceParameters covPar, bool shapenoise)
{
  double theta_1 = thetas_123[0];
  double theta_2 = thetas_123[1];
  double theta_3 = thetas_123[2];
  double theta_4 = thetas_456[0];
  double theta_5 = thetas_456[1];
  double theta_6 = thetas_456[2];

  double result = 0;
  result += L2(theta_1,theta_2,theta_3,theta_4,theta_5,theta_6,covPar,shapenoise);
  result += L2(theta_1,theta_2,theta_3,theta_5,theta_4,theta_6,covPar,shapenoise);
  result += L2(theta_1,theta_2,theta_3,theta_6,theta_4,theta_5,covPar,shapenoise);

  result += L2(theta_1,theta_3,theta_2,theta_4,theta_5,theta_6,covPar,shapenoise);
  result += L2(theta_1,theta_3,theta_2,theta_5,theta_4,theta_6,covPar,shapenoise);
  result += L2(theta_1,theta_3,theta_2,theta_6,theta_4,theta_5,covPar,shapenoise);

  result += L2(theta_2,theta_3,theta_1,theta_4,theta_5,theta_6,covPar,shapenoise);
  result += L2(theta_2,theta_3,theta_1,theta_5,theta_4,theta_6,covPar,shapenoise);
  result += L2(theta_2,theta_3,theta_1,theta_6,theta_4,theta_5,covPar,shapenoise);

  return result;
}



int integral_Gaussian_Map2_Covariance(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
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
  ApertureStatisticsCovarianceContainer* container = (ApertureStatisticsCovarianceContainer*) thisPtr;

  double theta_1 = container-> theta_1;
  double theta_2 = container-> theta_2;
  double shapenoise_powerspectrum = container -> shapenoise_powerspectrum;

  // Allocate memory on device for integrand values
  double* dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

  // Copy integration variables to device
  double* dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //allocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

  // Calculate values
  integrand_Gaussian_Map2_Covariance<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta_1, theta_2, dev_value,shapenoise_powerspectrum);
  // std::cerr << "test " << npts << std::endl;
  CudaCheckError();
  
  cudaFree(dev_vars); //Free variables

  
  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

  // std::cerr << value[5] << std::endl;

  cudaFree(dev_value); //Free values
  
  return 0; //Success :)  
}

double Gaussian_Map2_Covariance(double theta_1, double theta_2, const covarianceParameters covPar, bool shapenoise)
{
  //Set maximal l value such, that theta*l <= 10
  double thetaMin=std::min({theta_1,theta_2}); //should increase runtime, if either theta_123 or theta_456 is zero, so is their product
  double lMax=10./thetaMin;
  double lMin = 1.;

  ApertureStatisticsCovarianceContainer container;
  container.theta_1=theta_1;
  container.theta_2=theta_2;
  double shapenoise_powerspectrum;

  if(shapenoise)
    shapenoise_powerspectrum = covPar.power_spectrum_contribution;
  else
    shapenoise_powerspectrum = 0.;

  container.shapenoise_powerspectrum = shapenoise_powerspectrum;

  double result,error;

  double vals_min[4]={lMin, 0};
  double vals_max[4]={lMax, z_max}; //use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

  hcubature_v(1, integral_Gaussian_Map2_Covariance, &container, 2, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
  
  double survey_area = covPar.survey_area*pow(M_PI/180.,2);
  return result/survey_area/M_PI;
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
  
  hcubature_v(1, integral_Map3, &container, 3, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

  
  return result/248.050213442;//Divided by (2*pi)³
}

  

