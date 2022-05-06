#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "cubature.h"
#include "halomodel.cuh"
#include "cuba.h"

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

__device__ double uHat_product_permutations(const double &l1, const double &l2, const double &l3, double *thetas)
{
  double result;
  result = uHat_product(l1, l2, l3, thetas);
  result += uHat_product(l2, l3, l1, thetas);
  result += uHat_product(l3, l1, l2, thetas);
  result += uHat_product(l1, l3, l2, thetas);
  result += uHat_product(l3, l2, l1, thetas);
  result += uHat_product(l2, l1, l3, thetas);
  return result;
}

__device__ double dev_integrand_Map2(const double &ell, const double &z, double theta)
{
  return ell * pow(uHat(ell * theta), 2) * (limber_integrand(ell, z) + 0.5 * dev_sigma * dev_sigma / dev_n / dev_z_max);
}

__global__ void integrand_Map2(const double *vars, unsigned ndim, int npts, double theta, double *value)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  //Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double ell = vars[i * ndim];
    double z = vars[i * ndim + 1];

    value[i] = dev_integrand_Map2(ell, z, theta);
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
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              //allocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); //copying

  // Calculate values
  integrand_Map2<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta, dev_value);
  CudaCheckError();

  cudaFree(dev_vars); //Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); //Free values

  return 0; //Success :)
}

double Map2(double theta)
{
  //Set maximal l value such, that theta*l <= 10
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

// __device__ double dev_integrand_Gaussian_Map2_covariance(const double& ell, const double& z,
//   const double& theta_1, const double& theta_2, const double& shapenoise_contribution)
//   {
//     double result = ell*pow(uHat(ell*theta_1),2)*pow(uHat(ell*theta_2),2);
//     result *= pow((limber_integrand(ell,z)+shapenoise_contribution/dev_z_max),2);
//     return result;
//   }

// __global__ void integrand_Gaussian_Map2_Covariance(const double* vars, unsigned ndim, int npts, double theta_1, double theta_2,
//                                                     double* value, double shapenoise_contribution)
// {
//   // index of thread
//   int thread_index=blockIdx.x*blockDim.x + threadIdx.x;

//   //Grid-Stride loop, so I get npts evaluations
//   for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
//     {
//       // printf("%d, %d \n",npts*ndim-i*ndim-3,npts-i);
//       double ell=vars[i*ndim];
//       double z=vars[i*ndim+1];

//       value[i] = dev_integrand_Gaussian_Map2_covariance(ell,z,theta_1,theta_2,shapenoise_contribution);
//     }
// }

// int integral_Gaussian_Map2_Covariance(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
// {
//   if(fdim != 1)
//     {
//       std::cerr<<"integrand: Wrong number of function dimensions"<<std::endl;
//       exit(1);
//     };
//     if(ndim != 2)
//     {
//       std::cerr<<"integrand: Wrong number of variable dimensions"<<std::endl;
//       exit(1);
//     };

//   // Read data for integration
//   ApertureStatisticsCovarianceContainer* container = (ApertureStatisticsCovarianceContainer*) thisPtr;

//   double theta_1 = container-> theta_1;
//   double theta_2 = container-> theta_2;
//   double shapenoise_powerspectrum = container -> shapenoise_powerspectrum;

//   // Allocate memory on device for integrand values
//   double* dev_value;
//   CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

//   // Copy integration variables to device
//   double* dev_vars;
//   CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //allocate memory
//   CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

//   // Calculate values
//   integrand_Gaussian_Map2_Covariance<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta_1, theta_2, dev_value,shapenoise_powerspectrum);
//   // std::cerr << "test " << npts << std::endl;
//   CudaCheckError();

//   cudaFree(dev_vars); //Free variables

//   // Copy results to host
//   CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

//   // std::cerr << value[5] << std::endl;

//   cudaFree(dev_value); //Free values

//   return 0; //Success :)
// }

// double Gaussian_Map2_Covariance(double theta_1, double theta_2, const covarianceParameters covPar, bool shapenoise)
// {
//   //Set maximal l value such, that theta*l <= 10
//   double thetaMin=std::min({theta_1,theta_2}); //should increase runtime, if either theta_123 or theta_456 is zero, so is their product
//   double lMax=10./thetaMin;
//   double lMin = 1.;

//   ApertureStatisticsCovarianceContainer container;
//   container.theta_1=theta_1;
//   container.theta_2=theta_2;
//   double shapenoise_powerspectrum;

//   if(shapenoise)
//     shapenoise_powerspectrum = covPar.power_spectrum_contribution;
//   else
//     shapenoise_powerspectrum = 0.;

//   container.shapenoise_powerspectrum = shapenoise_powerspectrum;

//   double result,error;

//   double vals_min[4]={lMin, 0};
//   double vals_max[4]={lMax, z_max}; //use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

//   hcubature_v(1, integral_Gaussian_Map2_Covariance, &container, 2, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

//   double survey_area = covPar.survey_area*pow(M_PI/180.,2);
//   return result/survey_area/M_PI;
// }

__global__ void integrand_Map3_kernel(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double *value)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  //Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double l1 = vars[i * ndim];
    double l2 = vars[i * ndim + 1];
    double phi = vars[i * ndim + 2];

    double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));
    value[i] = l1 * l2 * bkappa(l1, l2, l3) * uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3);
    //      printf("%lf, %lf, %lf, %lf\n", l1, l2, l3, bkappa(l1, l2, l3));
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
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              //alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); //copying

  // Calculate values
  integrand_Map3_kernel<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta1, theta2, theta3, dev_value);

  cudaFree(dev_vars); //Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); //Free values

  return 0; //Success :)
}

double MapMapMap(const std::vector<double> &thetas, const double &phiMin, const double &phiMax, const double &lMin)
{
  //Set maximal l value such, that theta*l <= 10
  double thetaMin = std::min({thetas[0], thetas[1], thetas[2]});
  double lMax = 10. / thetaMin;

  ApertureStatisticsContainer container;
  container.thetas = thetas;
  double result, error;

  double vals_min[3] = {lMin, lMin, phiMin};
  double vals_max[3] = {lMax, lMax, phiMax};

  hcubature_v(1, integrand_Map3, &container, 3, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);

  return result / 8 / M_PI / M_PI / M_PI; //Divided by (2*pi)³
}

__global__ void integrand_Map4_kernel(const double *vars, unsigned ndim, int npts, 
  double theta1, double theta2, double theta3, double theta4, double *value,
  double lMin, double lMax, double phiMin, double phiMax, 
  double mMin, double mMax, double zMin, double zMax)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  double deltaEll = lMax - lMin;
  double deltaPhi = phiMax - phiMin;
  double deltaM = mMax - mMin;
  double deltaZ = zMax - zMin;

  //Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double l1 = vars[i * ndim]*deltaEll+lMin;
    double l2 = vars[i * ndim + 1]*deltaEll+lMin;
    double l3 = vars[i * ndim + 2]*deltaEll+lMin;
    double phi1 = vars[i * ndim + 3]*deltaPhi+phiMin;
    double phi2 = vars[i * ndim + 4]*deltaPhi+phiMin;
    double phi3 = vars[i * ndim + 5]*deltaPhi+phiMin;
    double m = vars[i * ndim + 6]*deltaM+mMin;
    double z = vars[i*ndim+7]*deltaZ+zMin;

    double l4 = l1 * l1 + l2 * l2 + l3 * l3 + 2 * l1 * l2 * cos(phi2 - phi1) + 2 * l2 * l3 * cos(phi2 - phi3) + 2 * l1 * l3 * cos(phi3 - phi1);
    double result;
    if (l4 > 0)
    {
      l4 = sqrt(l4);
      result = l1 * l2 * l3 * trispectrum_integrand(m, z, l1, l2, l3, l4) * uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l4 * theta4);
    }
    else
    {
      result=0;
    };
    
    value[i]=result;
  }
}

static int integrand_Map4(const int *ndim, const double* xx,
  const int *ncomp, double* ff, void *userdata, const int* nvec)
{

  if (*ndim != 8) //TO DO: throw exception here
  {
      std::cerr << "Wrong number of argument dimension in Map4 integration" << std::endl;
      exit(1);
  }

  if (*ncomp != 1) //TO DO: throw exception here
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
  double phiMin=container->phiMin;
  double phiMax=container->phiMax;
  double mMin=container->mMin;
  double mMax=container->mMax;
  double zMin=container->zMin;
  double zMax=container->zMax;

  std::cerr<<lMin<<" "<<lMax<<std::endl;
  std::cerr<<phiMin<<" "<<phiMax<<std::endl;
  std::cerr<<mMin<<" "<<mMax<<std::endl;
  std::cerr<<zMin<<" "<<zMax<<std::endl;

  // Allocate memory on device for integrand values
  double *dev_value;

  std::cerr<<*ncomp<<" "<<*nvec<<" "<<*ndim<<std::endl;

  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value,  *ncomp* *nvec*sizeof(double)));
   
 

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, *ndim * *nvec * sizeof(double)));                              //alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, xx, *ndim * *nvec * sizeof(double), cudaMemcpyHostToDevice)); //copying

  // Calculate values
  integrand_Map4_kernel<<<BLOCKS, THREADS>>>(dev_vars, *ndim, *nvec, 
    theta1, theta2, theta3, theta4, dev_value,
    lMin, lMax, phiMin, phiMax, mMin, mMax, zMin, zMax);

  cudaFree(dev_vars); //Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(ff, dev_value, *ncomp * *nvec * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); //Free values

  return 0; //Success :)
}

double Map4(const std::vector<double> &thetas, const double &phiMin, const double &phiMax, const double &lMin)
{
  //Set maximal l value such, that theta*l <= 10
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

  double deltaEll = container.lMax - container.lMin;
  double deltaPhi = container.phiMax - container.phiMin;
  double deltaM = container.mMax - container.mMin;
  double deltaZ = container.zMax - container.zMin;

  // allocate necessary variables
  int neval, fail, nregions;
  double integral[1], error[1], prob[1];

  // Internal parameters of the integration
  int NDIM = 8;  // dimensions of integration parameters
  int NCOMP = 1; // dimensions of function

  int NVEC = 1048576; // maximum value of parallel executions (adjust so that GPU memory can not overload)
  // now: 2^20

  double EPSREL = 1e-4; // accuracy parameters
  double EPSABS = 0;

  int VERBOSE = 2; // verbosity
  int LAST = 4;    // WHAT IS THAT?

  int SEED = 0;             // random seed. =0: Sobol quasi-random number, >0 pseudo-random numbers
  int MINEVAL = 1000;       // minimum number of evaluations
  int MAXEVAL = 1000000000; // maximum number of evaluations, if integral is not converged by then it throws an error

  int NNEW = 1000;
  int NMIN = 2;
  double FLATNESS = 25; // describes the "flatness" of integration function in the unit cube. try different values, see what happens

  const char *STATEFILE = NULL; // possibility to save integration and resume at later stage
  void *SPIN = NULL;            // something to do with the parallel processes, only necessary if parallelized manually

  //cubacores(0,0);
  //cubaaccel(0,0);

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

  if (fail != 0) //TO DO: These should throw exceptions
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

  return integral[0] * (pow(deltaEll, 3)*pow(deltaPhi,3)*deltaM*deltaZ) / pow(2 * M_PI, 6); //Divided by (2*pi)^6 and adjust for variable transform.

}

__global__ void integrand_Map6_kernel(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  //Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double l1 = vars[i * ndim];
    double l2 = vars[i * ndim + 1];
    double l3 = vars[i * ndim + 2];
    double l4 = vars[i * ndim + 3];
    double l5 = vars[i * ndim + 4];
    double phi1 = vars[i * ndim + 5];
    double phi2 = vars[i * ndim + 6];
    double phi3 = vars[i * ndim + 7];
    double phi4 = vars[i * ndim + 8];
    double phi5 = vars[i * ndim + 9];
    double m = vars[i * ndim + 10];

    double l6 = l1 * l1 + l2 * l2 + l3 * l3 + l4 * l4 + l5 * l5;
    l6 += 2 * l1 * l2 * cos(phi2 - phi1) + 2 * l1 * l3 * cos(phi3 - phi1) + 2 * l1 * l4 * cos(phi4 - phi1);
    l6 += 2 * l1 * l5 * cos(phi5 - phi1) + 2 * l2 * l3 * cos(phi3 - phi2) + 2 * l2 * l4 * cos(phi4 - phi2);
    l6 += 2 * l2 * l5 * cos(phi5 - phi2) + 2 * l3 * l4 * cos(phi4 - phi3) + 2 * l3 * l5 * cos(phi5 - phi3);
    l6 += 2 * l4 * l5 * cos(phi5 - phi4);

    if (l6 > 0)
    {
      l6 = sqrt(l6);
    }
    else
    {
      l6 = 0;
    };
    //printf("zmax: %lf", dev_z_max);
    if (l6 <= 0)
    {
      value[i] = 0;
    }
    else
    {
      value[i] = l1 * l2 * l3 * l4 * l5 * pentaspectrum_limber_integrated(0, dev_z_max, m, l1, l2, l3, l4, l5, l6) * uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l4 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);
    };
  }
}

int integrand_Map6(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "integrand: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
  // Read data for integration
  ApertureStatisticsContainer *container = (ApertureStatisticsContainer *)thisPtr;
  printf("%d\n", npts);

  if (npts > 5e7)
  {
    std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    return 1;
  };

  double theta1 = container->thetas.at(0);
  double theta2 = container->thetas.at(1);
  double theta3 = container->thetas.at(2);
  double theta4 = container->thetas.at(3);
  double theta5 = container->thetas.at(4);
  double theta6 = container->thetas.at(5);

  // Allocate memory on device for integrand values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              //alocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); //copying

  // Calculate values
  integrand_Map6_kernel<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, theta1, theta2, theta3, theta4, theta5, theta6, dev_value);

  cudaFree(dev_vars); //Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); //Free values

  return 0; //Success :)
}

double Map6(const std::vector<double> &thetas, const double &phiMin, const double &phiMax, const double &lMin)
{
  //Set maximal l value such, that theta*l <= 10
  double thetaMin = std::min({thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5]});
  double lMax = 10. / thetaMin;

  ApertureStatisticsContainer container;
  container.thetas = thetas;
  double result, error;

  double mmin = pow(10, logMmin);
  double mmax = pow(10, logMmax);
  double vals_min[11] = {lMin, lMin, lMin, lMin, lMin, phiMin, phiMin, phiMin, phiMin, phiMin, mmin};
  double vals_max[11] = {lMax, lMax, lMax, lMax, lMax, phiMax, phiMax, phiMax, phiMax, phiMax, mmax};

  hcubature_v(1, integrand_Map6, &container, 11, vals_min, vals_max, 0, 0, 1e-1, ERROR_L1, &result, &error);

  return result / pow(2 * M_PI, 10); //Divided by (2*pi)^10
}