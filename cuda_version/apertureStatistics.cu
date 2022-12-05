#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "cubature.h"
#include "halomodel.cuh"
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

__global__ void integrand_Map2_kernel(const double *vars, unsigned ndim, int npts, double theta, double *value)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double ell = vars[i * ndim];
    double z = vars[i * ndim + 1];

    value[i] = ell * pow(uHat(ell * theta), 2) * (limber_integrand_power_spectrum(ell, z) + 0.5 * dev_sigma * dev_sigma / dev_n / dev_z_max);
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

  // Allocate memory on device for integrand values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // allocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

  // Calculate values
  integrand_Map2_kernel<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta, dev_value);


  cudaFree(dev_vars); // Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Free values

  return 0; // Success :)
}

double Map2(double theta)
{
  // Set maximal l value such, that theta*l <= 10
  double lMax = 10. / theta;
  double lMin = 1.;

  ApertureStatisticsContainer container;
  std::vector<double> thetas{theta};
  container.thetas = thetas;

  double result, error;

  double vals_min[2] = {lMin, 0};
  double vals_max[2] = {lMax, z_max};

  hcubature_v(1, integrand_Map2, &container, 2, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

  return result / 2. / M_PI;
}

__global__ void integrand_Map3_kernel(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double *value)
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
    value[i] = l1 * l2 * bkappa(l1, l2, l3) * uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3);
  }
}

int integrand_Map3(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "integrand: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
  // Read data for integration
  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;

  double theta1 = container->thetas.at(0);
  double theta2 = container->thetas.at(1);
  double theta3 = container->thetas.at(2);

  // Allocate memory on device for integrand values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

  // Calculate values
  integrand_Map3_kernel<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta1, theta2, theta3, dev_value);

  cudaFree(dev_vars); // Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Free values

  return 0; // Success :)
}

double MapMapMap(const std::vector<double> &thetas, const double &phiMin, const double &phiMax, const double &lMin)
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
  double result, error;

  double vals_min[3] = {lMin, lMin, phiMin};
  double vals_max[3] = {lMax, lMax, phiMax};

  hcubature_v(1, integrand_Map3, &container, 3, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);

  return result / 8 / M_PI / M_PI / M_PI; // Divided by (2*pi)³
}

__global__ void integrand_Map4_kernel(const double *vars, unsigned ndim, int npts,
                                      double theta1, double theta2, double theta3, double theta4, double *value,
                                      double lMin, double lMax, double phiMin, double phiMax,
                                      double mMin, double mMax, double zMin, double zMax)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  double deltaEll = log(lMax) - log(lMin);
  double deltaPhi = phiMax - phiMin;
  double deltaM = log(mMax) - log(mMin);
  double deltaZ = zMax - zMin;

  // Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double l1 = exp(vars[i * ndim] * deltaEll) * lMin;
    double l2 = exp(vars[i * ndim + 1] * deltaEll) * lMin;
    double l3 = exp(vars[i * ndim + 2] * deltaEll) * lMin;
    double phi1 = vars[i * ndim + 3] * deltaPhi + phiMin;
    double phi2 = vars[i * ndim + 4] * deltaPhi + phiMin;
    double phi3 = vars[i * ndim + 5] * deltaPhi + phiMin;
    double m = exp(vars[i * ndim + 6] * deltaM) * mMin;
    double z = vars[i * ndim + 7] * deltaZ + zMin;

    double l4 = l1 * l1 + l2 * l2 + l3 * l3 + 2 * l1 * l2 * cos(phi2 - phi1) + 2 * l2 * l3 * cos(phi2 - phi3) + 2 * l1 * l3 * cos(phi3 - phi1);
    double result;
    if (l4 > 0)
    {
      l4 = sqrt(l4);
      result = l1 * l2 * l3 * uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l4 * theta4) * trispectrum_integrand(m, z, l1, l2, l3, l4);

      result *= l1 * l2 * l3 * m;
    }
    else // Set the result to 0 if l4<lMin
    {
      result = 0;
    };

    value[i] = result;
  }
}

static int integrand_Map4(const int *ndim, const double *xx,
                          const int *ncomp, double *ff, void *userdata, const int *nvec)
{

  if (*ncomp != 1) // TO DO: throw exception here
  {
    std::cerr << "Wrong number of function dimensions in Map4 integration" << std::endl;
    exit(1);
  }

  // Read data for integration
  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)userdata;

  double theta1 = container->thetas.at(0);
  double theta2 = container->thetas.at(1);
  double theta3 = container->thetas.at(2);
  double theta4 = container->thetas.at(3);
  double lMin = container->lMin;
  double lMax = container->lMax;
  double phiMin = container->phiMin;
  double phiMax = container->phiMax;
  double mMin = container->mMin;
  double mMax = container->mMax;
  double zMin = container->zMin;
  double zMax = container->zMax;

  // Allocate memory on device for integrand values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, *ncomp * *nvec * sizeof(double)));

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, *ndim * *nvec * sizeof(double)));                            // alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, xx, *ndim * *nvec * sizeof(double), cudaMemcpyHostToDevice)); // copying

  // Calculate values
  integrand_Map4_kernel<<<BLOCKS, THREADS>>>(dev_vars, *ndim, *nvec,
                                             theta1, theta2, theta3, theta4, dev_value,
                                             lMin, lMax, phiMin, phiMax, mMin, mMax, zMin, zMax);

  cudaFree(dev_vars); // Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(ff, dev_value, *ncomp * *nvec * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Free values

  return 0; // Success :)
}

double Map4(const std::vector<double> &thetas, const double &phiMin, const double &phiMax, const double &lMin)
{
  // Check if correct number of aperture radii is given
  if (thetas.size() != 4) // To Do: This should throw an exception
  {
    std::cerr << "Wrong number of thetas given to Map4" << std::endl;
    exit(-1);
  };

  // Set maximal l value such, that theta*l <= 10
  double thetaMin = std::min({thetas[0], thetas[1], thetas[2], thetas[3]});
  double lMax = 10. / thetaMin;

  // Create container
  ApertureStatisticsContainer container;
  container.thetas = thetas;

  // Set integral boundaries
  container.lMin = lMin;
  container.lMax = lMax;
  container.phiMin = 0;
  container.phiMax = 2 * M_PI;
  container.mMin = pow(10, logMmin);
  container.mMax = pow(10, logMmax);
  container.zMin = 0;
  container.zMax = z_max;

  double deltaEll = log(container.lMax) - log(container.lMin);
  double deltaPhi = container.phiMax - container.phiMin;
  double deltaM = log(container.mMax) - log(container.mMin);
  double deltaZ = container.zMax - container.zMin;

  // allocate necessary variables
  int neval, fail, nregions;
  double integral[1], error[1], prob[1];

  // Internal parameters of the integration
  int NDIM = 8;  // dimensions of integration parameters
  int NCOMP = 1; // dimensions of function

  int NVEC = 1048576; // maximum value of parallel executions (adjust so that GPU memory can not overload)
  // now: 2^20

  double EPSREL = 1e-2; // accuracy parameters
  double EPSABS = 0;

  int VERBOSE = 2; // verbosity
  int LAST = 4;

  int SEED = 0;             // random seed. =0: Sobol quasi-random number, >0 pseudo-random numbers
  int MINEVAL = 1000;       // minimum number of evaluations
  int MAXEVAL = 1000000000; // maximum number of evaluations, if integral is not converged by then it throws an error

  int NNEW = 1000;
  int NMIN = 2;
  double FLATNESS = 25; // describes the "flatness" of integration function in the unit cube. try different values, see what happens

  const char *STATEFILE = NULL; // possibility to save integration and resume at later stage
  void *SPIN = NULL;            // something to do with the parallel processes, only necessary if parallelized manually

  // GO!
  Suave(NDIM, NCOMP, (integrand_t)integrand_Map4, &container, NVEC,
        EPSREL, EPSABS, VERBOSE | LAST, SEED,
        MINEVAL, MAXEVAL, NNEW, NMIN, FLATNESS,
        STATEFILE, SPIN,
        &nregions, &neval, &fail, integral, error, prob);

  if (VERBOSE)
  {
    printf("SUAVE RESULT:\tnregions %d\tneval %d\tfail %d\n",
           nregions, neval, fail);

    for (int comp = 0; comp < NCOMP; comp++)
      printf("SUAVE RESULT:\t%.8f +- %.8f\t p = %.3f\n", integral[comp], error[comp], prob[comp]);
  }

  if (fail != 0) // TO DO: These should throw exceptions
  {
    if (fail > 0)
    {
      std::cerr << "Integral did not converge after " << neval << "evaluations." << std::endl;
      exit(1); // program is cancelled if integration does not converge. alternative: return 0 or nan
    }
    if (fail < 0)
    {
      std::cerr << "An error occured in the integration." << std::endl;
      exit(1);
    }
  }

  return integral[0] * (pow(deltaEll, 3) * pow(deltaPhi, 3) * deltaM * deltaZ) / pow(2 * M_PI, 6); // Divided by (2*pi)^6 and adjust for variable transform.
}

__global__ void integrand_Map6_kernel(const double *vars, unsigned ndim, int npts,
                                      double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value,
                                      double lMin, double lMax, double phiMin, double phiMax,
                                      double mMin, double mMax, double zMin, double zMax)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  double deltaEll = log(lMax) - log(lMin);
  double deltaPhi = phiMax - phiMin;
  double deltaM = log(mMax) - log(mMin);
  double deltaZ = zMax - zMin;

  // Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {

    double l1 = exp(vars[i * ndim] * deltaEll) * lMin;
    double l2 = exp(vars[i * ndim + 1] * deltaEll) * lMin;
    double l3 = exp(vars[i * ndim + 2] * deltaEll) * lMin;
    double l4 = exp(vars[i * ndim + 3] * deltaEll) * lMin;
    double l5 = exp(vars[i * ndim + 4] * deltaEll) * lMin;
    double phi1 = vars[i * ndim + 5] * deltaPhi + phiMin;
    double phi2 = vars[i * ndim + 6] * deltaPhi + phiMin;
    double phi3 = vars[i * ndim + 7] * deltaPhi + phiMin;
    double phi4 = vars[i * ndim + 8] * deltaPhi + phiMin;
    double phi5 = vars[i * ndim + 9] * deltaPhi + phiMin;
    double m = exp(vars[i * ndim + 10] * deltaM) * mMin;
    double z = vars[i * ndim + 11] * deltaZ + zMin;

    double l6 = l1 * l1 + l2 * l2 + l3 * l3 + l4 * l4 + l5 * l5;
    l6 += 2 * l1 * l2 * cos(phi2 - phi1) + 2 * l1 * l3 * cos(phi3 - phi1) + 2 * l1 * l4 * cos(phi4 - phi1);
    l6 += 2 * l1 * l5 * cos(phi5 - phi1) + 2 * l2 * l3 * cos(phi3 - phi2) + 2 * l2 * l4 * cos(phi4 - phi2);
    l6 += 2 * l2 * l5 * cos(phi5 - phi2) + 2 * l3 * l4 * cos(phi4 - phi3) + 2 * l3 * l5 * cos(phi5 - phi3);
    l6 += 2 * l4 * l5 * cos(phi5 - phi4);

    double result;
    if (l6 > 0)
    {
      l6 = sqrt(l6);
      result = l1 * l2 * l3 * l4 * l5 * pentaspectrum_integrand(m, z, l1, l2, l3, l4, l5, l6);
      result *= uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3);
      result *= uHat(l4 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);
      result *= l1 * l2 * l3 * l4 * l5 * m;
    }
    else
    {
      result = 0;
    };
    value[i] = result;
  }
}

static int integrand_Map6(const int *ndim, const double *xx,
                          const int *ncomp, double *ff, void *userdata, const int *nvec)
{

  if (*ndim != 12) // TO DO: throw exception here
  {
    std::cerr << "Wrong number of argument dimension in Map6 integration" << std::endl;
    std::cerr << "Given:" << *ndim << " Needed:12" << std::endl;
    exit(1);
  };

  if (*ncomp != 1) // TO DO: throw exception here
  {
    std::cerr << "Wrong number of function dimensions in Map4 integration" << std::endl;
    exit(1);
  }

  // Read data for integration
  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)userdata;

  double theta1 = container->thetas.at(0);
  double theta2 = container->thetas.at(1);
  double theta3 = container->thetas.at(2);
  double theta4 = container->thetas.at(3);
  double theta5 = container->thetas.at(4);
  double theta6 = container->thetas.at(5);
  double lMin = container->lMin;
  double lMax = container->lMax;
  double phiMin = container->phiMin;
  double phiMax = container->phiMax;
  double mMin = container->mMin;
  double mMax = container->mMax;
  double zMin = container->zMin;
  double zMax = container->zMax;

  // Allocate memory on device for integrand values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, *ncomp * *nvec * sizeof(double)));

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, *ndim * *nvec * sizeof(double)));                            // alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, xx, *ndim * *nvec * sizeof(double), cudaMemcpyHostToDevice)); // copying

  // Calculate values
  integrand_Map6_kernel<<<BLOCKS, THREADS>>>(dev_vars, *ndim, *nvec,
                                             theta1, theta2, theta3, theta4, theta5, theta6, dev_value,
                                             lMin, lMax, phiMin, phiMax, mMin, mMax, zMin, zMax);

  cudaFree(dev_vars); // Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(ff, dev_value, *ncomp * *nvec * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Free values

  return 0; // Success :)
}

double Map6(const std::vector<double> &thetas, const double &phiMin, const double &phiMax, const double &lMin)
{
  // Check number of given aperture radii
  if (thetas.size() != 6) // To Do: This should throw an exception
  {
    std::cerr << "Wrong number of thetas given to Map6" << std::endl;
    exit(-1);
  };

  // Set maximal l value such, that theta*l <= 10
  double thetaMin = std::min({thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5]});
  double lMax = 10. / thetaMin;

  // Create container
  ApertureStatisticsContainer container;
  container.thetas = thetas;

  // Set integral boundaries
  container.lMin = lMin;
  container.lMax = lMax;
  container.phiMin = 0;
  container.phiMax = 2 * M_PI;
  container.mMin = pow(10, logMmin);
  container.mMax = pow(10, logMmax);
  container.zMin = 0;
  container.zMax = z_max;

  double deltaEll = log(container.lMax) - log(container.lMin);
  double deltaPhi = container.phiMax - container.phiMin;
  double deltaM = log(container.mMax) - log(container.mMin);
  double deltaZ = container.zMax - container.zMin;

  // allocate necessary variables
  int neval, fail, nregions;
  double integral[1], error[1], prob[1];

  // Internal parameters of the integration
  int NDIM = 12; // dimensions of integration parameters
  int NCOMP = 1; // dimensions of function

  int NVEC = 1048576; // maximum value of parallel executions (adjust so that GPU memory can not overload)
  // now: 2^vars[i*ndim]

  double EPSREL = 1e-2; // accuracy parameters
  double EPSABS = 0;

  int VERBOSE = 2; // verbosity
  int LAST = 4;

  int SEED = 0;             // random seed. =0: Sobol quasi-random number, >0 pseudo-random numbers
  int MINEVAL = 1000;       // minimum number of evaluations
  int MAXEVAL = 1000000000; // maximum number of evaluations, if integral is not converged by then it throws an error

  int NNEW = 1000;
  int NMIN = 2;
  double FLATNESS = 25; // 25; // describes the "flatness" of integration function in the unit cube. try different values, see what happens

  const char *STATEFILE = NULL; // possibility to save integration and resume at later stage
  void *SPIN = NULL;            // something to do with the parallel processes, only necessary if parallelized manually

  // GO!
  Suave(NDIM, NCOMP, (integrand_t)integrand_Map6, &container, NVEC,
        EPSREL, EPSABS, VERBOSE | LAST, SEED,
        MINEVAL, MAXEVAL, NNEW, NMIN, FLATNESS,
        STATEFILE, SPIN,
        &nregions, &neval, &fail, integral, error, prob);

  if (VERBOSE)
  {
    printf("SUAVE RESULT:\tnregions %d\tneval %d\tfail %d\n",
           nregions, neval, fail);

    for (int comp = 0; comp < NCOMP; comp++)
      printf("SUAVE RESULT:\t%.8f +- %.8f\t (ratio:%.3f) p = %.3f\n", integral[comp], error[comp], error[comp] / integral[comp], prob[comp]);
  }

  if (fail != 0) // TO DO: These should throw exceptions
  {
    if (fail > 0)
    {
      std::cerr << "Integral did not converge after " << neval << "evaluations." << std::endl;
      exit(1); // program is cancelled if integration does not converge. alternative: return 0 or nan
    }
    if (fail < 0)
    {
      std::cerr << "An error occured in the integration." << std::endl;
      exit(1);
    }
  }

  return integral[0] * (pow(deltaEll, 5) * pow(deltaPhi, 5) * deltaM * deltaZ) / pow(2 * M_PI, 10); // Divided by (2*pi)^10 and adjust for variable transform.
}