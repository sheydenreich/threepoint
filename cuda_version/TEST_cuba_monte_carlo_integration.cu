#include "apertureStatisticsCovariance.cuh"
#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"

#include "cubature.h"
#include "cuba.h"

// #include <cstdlib>
// #include <math.h>
// #include <string>

#include <algorithm>
#include <iostream>
// #include <chrono>
// #include <thread>

int main()
{
    //std::cout << dummy_T7(0.5,0.5,0.5,0.5,0.5,0.5) << std::endl;
}

// double testfunc(double r, double phi)
// {
//     return exp(-r*r/2);
// }

// static int Integrand(const int *ndim, const double* xx,
//     const int *ncomp, double* ff, void *userdata, const int* nvec)
// {   
//     if(*ndim!=8)
//     {
//         std::cerr << "Wrong number of argument dimension" << std::endl;
//         exit(1);
//     }

//     ApertureStatisticsCovarianceContainer* container = (ApertureStatisticsCovarianceContainer*) userdata;

//     std::vector<double> thetas_123 = container->thetas_123;

//     std::cout << thetas_123[0] << std::endl;

//     for(int i=0; i<*nvec; i++)
//     {
//         double r1 = xx[i* *ndim+0];
//         double r2 = xx[i* *ndim+1];
//         double r3 = xx[i* *ndim+2];
//         double r4 = xx[i* *ndim+3];

//         double phi1 = xx[i* *ndim+4];
//         double phi2 = xx[i* *ndim+5];
//         double phi3 = xx[i* *ndim+6];
//         double phi4 = xx[i* *ndim+7];


//         double result = testfunc(r1,phi1)*testfunc(r2,phi2)*testfunc(r3,phi3)*testfunc(r4,phi4);
//         ff[i] = result;
//     }
//     return 0;
// }

// int main()
// {
//     int NDIM = 8;
//     int NCOMP = 1;
//     // void* USERDATA = NULL;

//     int NVEC = 96;

//     double EPSREL = 1e-4;
//     double EPSABS = 0;

//     int VERBOSE = 2;
//     int LAST = 4;

//     int SEED = 0;
//     int MINEVAL = 10;
//     int MAXEVAL = 1000000000;

//     int NNEW = 1000;
//     int NMIN = 2;
//     double FLATNESS = 25;

//     const char* STATEFILE = NULL;
//     void* SPIN = NULL;

//     int neval,fail,nregions;

//     double integral[1],error[1],prob[1];

//     ApertureStatisticsCovarianceContainer container;

//     std::vector<double> thetas_123 = {0.5,0.5,0.5};
//     std::vector<double> thetas_456 = {0.5,0.5,0.5};

//     container.thetas_123 = thetas_123;
//     container.thetas_456 = thetas_456;

//     // void* USERDATA = (void*) container;

//     Suave(NDIM, NCOMP, (integrand_t)Integrand, &container, NVEC,
//         EPSREL, EPSABS, VERBOSE | LAST, SEED,
//         MINEVAL, MAXEVAL, NNEW, NMIN, FLATNESS,
//         STATEFILE, SPIN,
//         &nregions, &neval, &fail, integral, error, prob);
    
//     printf("SUAVE RESULT:\tnregions %d\tneval %d\tfail %d\n",
//         nregions, neval, fail);

//     for(int comp = 0; comp < NCOMP; comp++ )
//         printf("SUAVE RESULT:\t%.8f +- %.8f\t p = %.3f\n",
//           integral[comp], error[comp], prob[comp]);
    
    
// }

