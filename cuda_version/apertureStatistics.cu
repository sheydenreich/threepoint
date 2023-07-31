#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "cubature.h"
// #include "halomodel.cuh"
#include "cuba.h"

#include <math.h>
#include <iostream>
#include <algorithm>

__device__ double uHat(double eta)
{
  double temp = 0.5 * eta * eta;
  return temp * exp(-temp);
}

__device__ double uHat_product(const double &l1, const double &l2, const double &l3, double *thetas)
{
  return uHat(l1 * thetas[0]) * uHat(l2 * thetas[1]) * uHat(l3 * thetas[2]);
}

__global__ void integrand_Map2_kernel(const double *vars, unsigned ndim, int npts, double theta, int zbin1, int zbin2, double *dev_g, double *dev_p, int Ntomo, double * dev_sigma_epsilon, double * dev_ngal, double *value)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double ell = vars[i * ndim];
    double z = vars[i * ndim + 1];
    if (T17_CORRECTION)
    {
      double correction = 1 + pow(ell / (1.6 * 8192.), 2);
      value[i] = ell * pow(uHat(ell * theta), 2) * dev_limber_integrand_power_spectrum(ell, z, zbin1, zbin2, dev_g, dev_p, Ntomo) / correction;
      if (zbin1 == zbin2)
      {
        value[i] += ell * pow(uHat(ell * theta)* dev_sigma_epsilon[zbin1], 2) * 0.5 / dev_ngal[zbin1] * pow(2.9088820866e-4, 2) / dev_z_max;
      }
    }
    else
    {
      value[i] = ell * pow(uHat(ell * theta), 2) * dev_limber_integrand_power_spectrum(ell, z, zbin1, zbin2, dev_g, dev_p, Ntomo);
      if (zbin1 == zbin2)
      {
        value[i] += ell * pow(uHat(ell * theta)* dev_sigma_epsilon[zbin1], 2) * 0.5 / dev_ngal[zbin1] * pow(2.9088820866e-4, 2) / dev_z_max;
      }
    }
  }
}

int integrand_Map2(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "integrand: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
  if (ndim != 2)
  {
    std::cerr << "integrand: Wrong number of variable dimensions" << std::endl;
    exit(1);
  };

  // Read data for integration
  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  double theta = container->thetas.at(0);

  int zbin1 = container->zbins.at(0);
  int zbin2 = container->zbins.at(1);

  double *dev_g = container->dev_g;
  double *dev_p = container->dev_p;
  int Ntomo = container->Ntomo;

  double *dev_sigma_epsilon = container->dev_sigma_epsilon;
  double *dev_ngal = container->dev_ngal;

  // Allocate memory on device for integrand values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // allocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

  // Calculate values
  integrand_Map2_kernel<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta, zbin1, zbin2, dev_g, dev_p, Ntomo, dev_sigma_epsilon, dev_ngal, dev_value);
  CudaCheckError();

  cudaFree(dev_vars); // Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Free values

  return 0; // Success :)
}

double Map2(double theta, const std::vector<int> &zbins, double *dev_g, double *dev_p, int Ntomo, double * dev_sigma_epsilon, double * dev_ngal)
{
  // Set maximal l value such, that theta*l <= 10
  double lMax = 10. / theta;
  double lMin = 1.;

  ApertureStatisticsContainer container;
  std::vector<double> thetas{theta};
  container.thetas = thetas;
  container.zbins = zbins;

  container.dev_g = dev_g;
  container.dev_p = dev_p;
  container.Ntomo = Ntomo;

  container.dev_sigma_epsilon = dev_sigma_epsilon;
  container.dev_ngal = dev_ngal;

  double result, error;

  double vals_min[2] = {lMin, 0};
  double vals_max[2] = {lMax, z_max};

  hcubature_v(1, integrand_Map2, &container, 2, vals_min, vals_max, 0, 0, 1e-5, ERROR_L1, &result, &error);

  return result / 2. / M_PI;
}

__global__ void integrand_Map3_kernel(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, int zbin1, int zbin2, int zbin3, double *dev_g, double *dev_p, int Ntomo, double *value)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double l1 = vars[i * ndim];
    double l2 = vars[i * ndim + 1];
    double phi = vars[i * ndim + 2];

    double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));
    value[i] = l1 * l2 * bkappa(l1, l2, l3, zbin1, zbin2, zbin3, dev_g, dev_p, Ntomo) * uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3);
  }
}

int integrand_Map3(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "integrand: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
  // printf("%d\n", npts);
  // Read data for integration
  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  double theta1 = container->thetas.at(0);
  double theta2 = container->thetas.at(1);
  double theta3 = container->thetas.at(2);

  int zbin1 = container->zbins.at(0);
  int zbin2 = container->zbins.at(1);
  int zbin3 = container->zbins.at(2);

  double *dev_g = container->dev_g;
  double *dev_p = container->dev_p;
  int Ntomo = container->Ntomo;

  // Allocate memory on device for integrand values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

  // Calculate values
  integrand_Map3_kernel<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta1, theta2, theta3, zbin1, zbin2, zbin3, dev_g, dev_p, Ntomo, dev_value);


  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
  
  cudaFree(dev_vars); // Free variables

  cudaFree(dev_value); // Free values

  return 0; // Success :)
}

double MapMapMap(const std::vector<double> &thetas, const std::vector<int> &zbins, double *dev_g, double *dev_p, int Ntomo, const double &phiMin, const double &phiMax, const double &lMin)
{
  // Check if correct number of aperture radii is given
  if (thetas.size() != 3) // To Do: This should throw an exception
  {
    std::cerr << "Wrong number of thetas given to MapMapMap" << std::endl;
    exit(-1);
  };

  // Set maximal l value such, that theta*l <= 10
  double thetaMin = std::min({thetas[0], thetas[1], thetas[2]});
  double lMax = 10. / thetaMin;

  ApertureStatisticsContainer container;
  container.thetas = thetas;
  container.zbins = zbins;

  container.dev_g = dev_g;
  container.dev_p = dev_p;
  container.Ntomo = Ntomo;

  double result, error;

  double vals_min[3] = {lMin, lMin, phiMin};
  double vals_max[3] = {lMax, lMax, phiMax};

  hcubature_v(1, integrand_Map3, &container, 3, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

  return result / 8 / M_PI / M_PI / M_PI; // Divided by (2*pi)³
}
