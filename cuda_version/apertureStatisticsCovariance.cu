#include "apertureStatisticsCovariance.cuh"
#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"
#include "halomodel.cuh"

#include "cubature.h"
#include "cuba.h"

#include <math.h>
#include <algorithm>
#include <iostream>

/***************** GENERAL DEFINITIONS **************************************************/

__constant__ double dev_thetaMax;
__constant__ double dev_thetaMax_smaller;
__constant__ double dev_area;

__constant__ double dev_lMin;
__constant__ double dev_lMax;
double lMin;
double thetaMax;
double thetaMax_smaller;
double area;
int type; // 0: circle, 1: square, 2: infinite, 3: rectangular

__constant__ double dev_sigma2_from_windowfunction_array[n_redshift_bins];

void initCovariance()
{
    copyConstants();

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_thetaMax, &thetaMax, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_area, &area, sizeof(double)));

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma, &sigma, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n, &n, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMin, &lMin, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_constant_powerspectrum, &constant_powerspectrum, sizeof(bool)));
    if (type == 3)
    {
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_thetaMax_smaller, &thetaMax_smaller, sizeof(double)));
    };
};

void writeCov(const std::vector<double> &values, const int &N, const std::string &filename)
{
    // Check if there are the correct number of values for an NxN Covariance
    if (values.size() != N * (N + 1) / 2)
    {
        throw std::out_of_range("writeCov: Values has wrong length");
    };

    std::ofstream out(filename); // Open file and check if it can be opened
    if (!out.is_open())
    {
        throw std::runtime_error("writeCov: Could not open " + filename);
    };

    // Read values into an NxN Matrix
    std::vector<double> cov_tmp(N * N);
    int ix = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            cov_tmp.at(i * N + j) = values.at(ix);
            ix += 1;
        }
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            cov_tmp.at(i * N + j) = cov_tmp.at(j * N + i);
        }
    }

    // Do Output
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            out << cov_tmp.at(i * N + j) << " ";
        };
        out << std::endl;
    }
}

void writeCrossCov(const std::vector<double> &values, const int &Ninner, const int &Nouter, const std::string &filename)
{
    // Check if there are the correct number of values for an NxN Covariance
    if (values.size() != Ninner * Nouter)
    {
        throw std::out_of_range("writeCrossCov: Values has wrong length");
    };

    std::ofstream out(filename); // Open file and check if it can be opened
    if (!out.is_open())
    {
        throw std::runtime_error("writeCrossCov: Could not open " + filename);
    };

    // Do Output
    for (int i = 0; i < Nouter; i++)
    {
        for (int j = 0; j < Ninner; j++)
        {
            out << values.at(i * Ninner + j) << " ";
        };
        out << std::endl;
    };
}

__host__ __device__ double G_circle(const double &ell)
{
#ifdef __CUDA_ARCH__
    double tmp = dev_thetaMax * ell;
#else
    double tmp = thetaMax * ell;
#endif
    double result = j1(tmp);
    result *= result;
    result *= 4 / tmp / tmp;

    return result;
}

__host__ __device__ double G_square(const double &ellX, const double &ellY)
{
#ifdef __CUDA_ARCH__
    double tmp1 = 0.5 * ellX * dev_thetaMax;
    double tmp2 = 0.5 * ellY * dev_thetaMax;
#else
    double tmp1 = 0.5 * ellX * thetaMax;
    double tmp2 = 0.5 * ellY * thetaMax;
#endif

    double j01, j02;
    if (abs(tmp1) <= 1e-6)
        j01 = 1;
    else
        j01 = sin(tmp1) / tmp1;

    if (abs(tmp2) <= 1e-6)
        j02 = 1;
    else
        j02 = sin(tmp2) / tmp2;

    return j01 * j01 * j02 * j02;
};

__host__ __device__ double G_rectangle(const double &ellX, const double &ellY)
{
#ifdef __CUDA_ARCH__
    double tmp1 = 0.5 * ellX * dev_thetaMax;
    double tmp2 = 0.5 * ellY * dev_thetaMax_smaller;
#else
    double tmp1 = 0.5 * ellX * thetaMax;
    double tmp2 = 0.5 * ellY * thetaMax_smaller;
#endif

    double j01, j02;
    if (abs(tmp1) <= 1e-6)
        j01 = 1;
    else
        j01 = sin(tmp1) / tmp1;

    if (abs(tmp2) <= 1e-6)
        j02 = 1;
    else
        j02 = sin(tmp2) / tmp2;

    return j01 * j01 * j02 * j02;
};

double sigma2_from_windowFunction(double chi)
{
    double qmin = 0;   //-1e3;
    double qmax = 1e5; // 1e3

    double vals_min[2] = {qmin, qmin};
    double vals_max[2] = {qmax, qmax};
    Sigma2Container container;
    container.chi = chi;

    double result, error;
    hcubature_v(1, integrand_sigma2_from_windowFunction, &container, 2, vals_min, vals_max,
                0, 0, 1e-4, ERROR_L1, &result, &error);

    result /= pow(2 * M_PI, 2);
    result *= 4;

    // std::cout<<chi<<" "<<result<<std::endl;

    return result;
}

int integrand_sigma2_from_windowFunction(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_sigma2_from_windowFunction: wrong function dimension" << std::endl;
        return -1;
    };
    Sigma2Container *container_ = (Sigma2Container *)container;

    // Allocate memory on device for integrand values
    double *dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

    // Copy integration variables to device
    double *dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

    integrand_sigma2_from_windowFunction<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, type, container_->chi, dev_value);

    cudaFree(dev_vars); // Free variables

    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); // Free values

    return 0; // Success :)
}

__global__ void integrand_sigma2_from_windowFunction(const double *vars, unsigned ndim, int npts, int type, double chi, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double q1 = vars[i * ndim];
        double q2 = vars[i * ndim + 1];

        double q = sqrt(q1 * q1 + q2 * q2);
        double Gfactor;

        if (type == 0)
        {
            Gfactor = G_circle(q);
        }
        else if (type == 1)
        {
            Gfactor = G_square(q1, q2);
        }
        else
        {
            printf("Wrong geometry for sigma2_from_windowFunction, %d \n", type);
            return;
        };
        double result;
        if (q / chi < 1e-9)
        {
            result = 0;
        }
        else
        {
            result = Gfactor * linear_pk(q / chi);
            // printf("q, chi, G, P is %e, %e, %e, %e\n",q, chi, Gfactor, linear_pk(q/chi), result);
        };
        value[i] = result;
    }
}

/******************* FOR COVARIANCE OF <Map^3> *****************************************/

double T1_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    // Check if the right number of thetas is given
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T1_total: Wrong number of aperture radii");
    };

    // Read out thetas
    double th0 = thetas_123.at(0);
    double th1 = thetas_123.at(1);
    double th2 = thetas_123.at(2);
    double th3 = thetas_456.at(0);
    double th4 = thetas_456.at(1);
    double th5 = thetas_456.at(2);

    double result; // Will contain final result

    if (th3 == th4)
    {
        if (th4 == th5) // All thetas_456 are equal
        {
            result = 6 * T1(th0, th1, th2, th3, th3, th3);
        }
        else
        {
            result = 2 * T1(th0, th1, th2, th3, th3, th5);
            result += 2 * T1(th0, th1, th2, th3, th5, th3);
            result += 2 * T1(th0, th1, th2, th5, th3, th3);
        }
    }
    else if (th3 == th5)
    {
        result = 2 * T1(th0, th1, th2, th3, th4, th3);
        result += 2 * T1(th0, th1, th2, th3, th3, th4);
        result += 2 * T1(th0, th1, th2, th4, th3, th3);
    }
    else if (th4 == th5)
    {
        result = 2 * T1(th0, th1, th2, th3, th4, th4);
        result += 2 * T1(th0, th1, th2, th4, th3, th4);
        result += 2 * T1(th0, th1, th2, th4, th4, th3);
    }
    else // All thetas_456 are different from each other
    {
        result = T1(th0, th1, th2, th3, th4, th5);
        result += T1(th0, th1, th2, th3, th5, th4);
        result += T1(th0, th1, th2, th4, th3, th5);
        result += T1(th0, th1, th2, th4, th5, th3);
        result += T1(th0, th1, th2, th5, th3, th4);
        result += T1(th0, th1, th2, th5, th4, th3);
    }

    return result;
}

double T2_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (type == 2)
        return 0;

    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T2_total: Wrong number of aperture radii");
    };

    double result;

    double th0 = thetas_123.at(0);
    double th1 = thetas_123.at(1);
    double th2 = thetas_123.at(2);
    double th3 = thetas_456.at(0);
    double th4 = thetas_456.at(1);
    double th5 = thetas_456.at(2);

    if (th0 == th1 && th1 == th2) // All thetas_123 are the same
    {
        if (th3 == th4 && th4 == th5) // All thetas_456 are the same
        {
            result = 9 * T2(th0, th0, th0, th3, th3, th3);
        }
        else if (th3 == th4) // th3=th4 and th4 \neq th5
        {
            result = 6 * T2(th0, th0, th0, th3, th3, th5);
            result += 3 * T2(th0, th0, th0, th5, th3, th3);
        }
        else if (th3 == th5) // th3=th5 and th4 \neq th5
        {
            result = 6 * T2(th0, th0, th0, th3, th4, th3);
            result += 3 * T2(th0, th0, th0, th4, th3, th3);
        }
        else if (th4 == th5) // th4 = th5 and th3 \neq th4
        {
            result = 6 * T2(th0, th0, th0, th4, th3, th4);
            result += 3 * T2(th0, th0, th0, th3, th4, th4);
        }
        else // All thetas_456 are different
        {
            result = 3 * T2(th0, th0, th0, th3, th4, th5);
            result += 3 * T2(th0, th0, th0, th4, th3, th5);
            result += 3 * T2(th0, th0, th0, th5, th3, th4);
        }
    }
    else if (th0 == th1) // th0=th1 and th0 \neq th2
    {
        if (th3 == th4 && th4 == th5)
        {
            result = 6 * T2(th0, th2, th0, th3, th3, th3);
            result += 3 * T2(th0, th0, th2, th3, th3, th3);
        }
        else if (th3 == th4) /// th3=th4 and th4 \neq th5
        {
            result = 2 * T2(th0, th0, th2, th3, th3, th5);
            result += T2(th0, th0, th2, th5, th3, th3);
            result += 4 * T2(th0, th2, th0, th3, th3, th5);
            result += 2 * T2(th0, th2, th0, th5, th3, th3);
        }
        else if (th3 == th5)
        {
            result = 2 * T2(th0, th0, th2, th3, th4, th3);
            result += T2(th0, th0, th2, th4, th3, th3);
            result += 4 * T2(th0, th2, th0, th3, th4, th3);
            result += 2 * T2(th0, th2, th0, th4, th3, th3);
        }
        else if (th4 == th5)
        {
            result = 2 * T2(th0, th0, th2, th4, th3, th4);
            result += T2(th0, th0, th2, th3, th4, th4);
            result += 4 * T2(th0, th2, th0, th4, th3, th4);
            result += 2 * T2(th0, th2, th0, th3, th4, th4);
        }
        else
        {
            result = T2(th0, th0, th2, th3, th4, th5);
            result += T2(th0, th0, th2, th4, th3, th5);
            result += T2(th0, th0, th2, th5, th3, th4);
            result += 2 * T2(th0, th2, th0, th3, th4, th5);
            result += 2 * T2(th0, th2, th0, th4, th3, th5);
            result += 2 * T2(th0, th2, th0, th5, th3, th4);
        }
    }
    else if (th1 == th2) // th1=th2 and th0 \neq th1
    {
        if (th3 == th4 && th4 == th5)
        {
            result = 6 * T2(th0, th1, th1, th3, th3, th3);
            result += 3 * T2(th1, th1, th0, th3, th3, th3);
        }
        else if (th3 == th4)
        {
            result = 2 * T2(th0, th1, th1, th5, th3, th3);
            result += T2(th1, th1, th0, th5, th3, th3);
            result += 4 * T2(th0, th1, th1, th3, th3, th5);
            result += 2 * T2(th1, th1, th0, th3, th3, th5);
        }
        else if (th3 == th5)
        {
            result = 4 * T2(th0, th1, th1, th3, th4, th3);
            result += 2 * T2(th0, th1, th1, th4, th3, th3);
            result += 2 * T2(th1, th1, th0, th3, th4, th3);
            result += T2(th1, th1, th0, th4, th3, th3);
        }
        else if (th4 == th5)
        {
            result = 2 * T2(th0, th1, th1, th3, th4, th4);
            result += T2(th1, th1, th0, th3, th4, th4);
            result += 4 * T2(th0, th1, th1, th4, th3, th4);
            result += 2 * T2(th1, th1, th0, th4, th3, th4);
        }
        else
        {
            result = 2 * T2(th0, th1, th1, th3, th4, th5);
            result += 2 * T2(th0, th1, th1, th4, th3, th5);
            result += 2 * T2(th0, th1, th1, th5, th3, th4);
            result += T2(th1, th1, th0, th3, th4, th5);
            result += T2(th1, th1, th0, th4, th3, th5);
            result += T2(th1, th1, th0, th5, th3, th4);
        }
    }
    else if (th0 == th2) // th0=th2 and th0 \neq th1
    {
        if (th3 == th4 && th4 == th5)
        {
            result = 6 * T2(th0, th1, th0, th3, th3, th3);
            result += 3 * T2(th1, th0, th0, th3, th3, th3);
        }
        else if (th3 == th4)
        {
            result = 2 * T2(th0, th1, th0, th5, th3, th3);
            result += T2(th1, th0, th0, th5, th3, th3);
            result += 4 * T2(th0, th1, th0, th3, th3, th5);
            result += 2 * T2(th1, th0, th0, th3, th3, th5);
        }
        else if (th3 == th5)
        {
            result = 4 * T2(th0, th1, th0, th3, th4, th3);
            result += 2 * T2(th0, th1, th0, th4, th3, th3);
            result += 2 * T2(th1, th0, th0, th3, th4, th3);
            result += T2(th1, th0, th0, th4, th3, th3);
        }
        else if (th4 == th5)
        {
            result = 2 * T2(th0, th1, th0, th3, th4, th4);
            result += T2(th1, th0, th0, th3, th4, th4);
            result += 4 * T2(th0, th1, th0, th4, th3, th4);
            result += 2 * T2(th1, th0, th0, th4, th3, th4);
        }
        else
        {
            result = 2 * T2(th0, th1, th0, th3, th4, th5);
            result += 2 * T2(th0, th1, th0, th4, th3, th5);
            result += 2 * T2(th0, th1, th0, th5, th3, th4);
            result += T2(th1, th0, th0, th3, th4, th5);
            result += T2(th1, th0, th0, th4, th3, th5);
            result += T2(th1, th0, th0, th5, th3, th4);
        }
    }
    else // All thetas_123 are different
    {
        if (th3 == th4 && th4 == th5)
        {
            result = 3 * T2(th0, th1, th2, th3, th3, th3);
            result += 3 * T2(th0, th2, th1, th3, th3, th3);
            result += 3 * T2(th1, th2, th0, th3, th3, th3);
        }
        else if (th3 == th4)
        {
            result = 2 * T2(th0, th1, th2, th3, th3, th5);
            result += T2(th0, th1, th2, th5, th3, th3);
            result += 2 * T2(th0, th2, th1, th3, th3, th5);
            result += T2(th0, th2, th1, th5, th3, th3);
            result += 2 * T2(th1, th2, th0, th3, th3, th5);
            result += T2(th1, th2, th0, th5, th3, th3);
        }
        else if (th3 == th5)
        {
            result = 2 * T2(th0, th1, th2, th3, th4, th3);
            result += T2(th0, th1, th2, th4, th3, th3);
            result += 2 * T2(th0, th2, th1, th3, th4, th3);
            result += T2(th0, th2, th1, th4, th3, th3);
            result += 2 * T2(th1, th2, th0, th3, th4, th3);
            result += T2(th1, th2, th0, th4, th3, th3);
        }
        else if (th4 == th5)
        {
            result = 2 * T2(th0, th1, th2, th4, th3, th4);
            result += T2(th0, th1, th2, th3, th4, th4);
            result += 2 * T2(th0, th2, th1, th4, th3, th4);
            result += T2(th0, th2, th1, th3, th4, th4);
            result += 2 * T2(th1, th2, th0, th4, th3, th4);
            result += T2(th1, th2, th0, th3, th4, th4);
        }
        else // All thetas_456 are different
        {
            result = T2(th0, th1, th2, th3, th4, th5);
            result += T2(th0, th1, th2, th4, th3, th5);
            result += T2(th0, th1, th2, th5, th3, th4);
            result += T2(th0, th2, th1, th3, th4, th5);
            result += T2(th0, th2, th1, th4, th3, th5);
            result += T2(th0, th2, th1, th5, th3, th4);
            result += T2(th1, th2, th0, th3, th4, th5);
            result += T2(th1, th2, th0, th4, th3, th5);
            result += T2(th1, th2, th0, th5, th3, th4);
        }
    }
    return result;
}

double T4_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T4_total: Wrong number of aperture radii");
    };

    double th0 = thetas_123.at(0);
    double th1 = thetas_123.at(1);
    double th2 = thetas_123.at(2);
    double th3 = thetas_456.at(0);
    double th4 = thetas_456.at(1);
    double th5 = thetas_456.at(2);

    double result;

    if (th0 == th1 && th0 == th2)
    {

        if (th3 == th4 && th3 == th5)
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th0, th3, th3);
            result = 9 * T4(th3, th0, th0, th0, th3, th3);
        }
        else if (th3 == th4)
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th0, th3, th5);
            result = 6 * T4(th3, th0, th0, th0, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th0, th0, th3, th3);
            result += 3 * T4(th5, th0, th0, th0, th3, th3);
        }
        else
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th0, th4, th5);
            result = 3 * T4(th3, th0, th0, th0, th4, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th0, th0, th0, th3, th5);
            result += 3 * T4(th4, th0, th0, th0, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th0, th0, th3, th4);
            result += 3 * T4(th5, th0, th0, th0, th3, th4);
        }
    }
    else if (th0 == th1)
    {
        if (th3 == th4 && th3 == th5)
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th0, th3, th3);
            result = 6 * T4(th3, th0, th2, th0, th3, th3);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th2, th3, th3);
            result += 3 * T4(th3, th0, th0, th2, th3, th3);
        }
        else if (th3 == th4)
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th2, th3, th5);
            result = 2 * T4(th3, th0, th0, th2, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th0, th3, th5);
            result += 4 * T4(th3, th0, th2, th0, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th0, th2, th3, th3);
            result += T4(th5, th0, th0, th2, th3, th3);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th2, th0, th3, th3);
            result += 2 * T4(th5, th0, th2, th0, th3, th3);
        }
        else
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th2, th4, th5);
            result = T4(th3, th0, th0, th2, th4, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th0, th4, th5);
            result += 2 * T4(th3, th0, th2, th0, th4, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th0, th0, th2, th3, th5);
            result += T4(th4, th0, th0, th2, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th0, th2, th0, th3, th5);
            result += 2 * T4(th4, th0, th2, th0, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th0, th2, th3, th4);
            result += T4(th5, th0, th0, th2, th3, th4);
            ;
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th2, th0, th3, th4);
            result += 2 * T4(th5, th0, th2, th0, th3, th4);
        }
    }
    else if (th3 == th4)
    {
        if (th3 == th5)
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th1, th2, th3, th3);
            result = 3 * T4(th3, th0, th1, th2, th3, th3);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th1, th3, th3);
            result += 3 * T4(th3, th0, th2, th1, th3, th3);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th1, th2, th0, th3, th3);
            result += 3 * T4(th3, th1, th2, th0, th3, th3);
        }
        else
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th1, th2, th3, th5);
            result = 2 * T4(th3, th0, th1, th2, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th1, th3, th5);
            result += 2 * T4(th3, th0, th2, th1, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th1, th2, th0, th3, th5);
            result += 2 * T4(th3, th1, th2, th0, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th1, th2, th3, th4);
            result += T4(th5, th0, th1, th2, th3, th4);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th1, th2, th0, th3, th3);
            result += T4(th5, th1, th2, th0, th3, th3);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th2, th1, th3, th3);
            result += T4(th5, th0, th2, th1, th3, th3);
        }
    }
    else if (th4 == th5)
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th1, th2, th3, th5);
        result = T4(th3, th0, th1, th2, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th1, th3, th5);
        result += T4(th3, th0, th2, th1, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th1, th2, th0, th3, th5);
        result += T4(th3, th1, th2, th0, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th1, th2, th3, th4);
        result += 2 * T4(th4, th0, th1, th2, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th1, th2, th0, th3, th3);
        result += 2 * T4(th4, th1, th2, th0, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th2, th1, th3, th3);
        result += 2 * T4(th4, th0, th2, th1, th3, th4);
    }
    else
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th1, th2, th4, th5);
        result = T4(th3, th0, th1, th2, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th1, th4, th5);
        result += T4(th3, th0, th2, th1, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th1, th2, th0, th4, th5);
        result += T4(th3, th1, th2, th0, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th0, th1, th2, th3, th5);
        result += T4(th4, th0, th1, th2, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th0, th2, th1, th3, th5);
        result += T4(th4, th0, th2, th1, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th1, th2, th0, th3, th5);
        result += T4(th4, th1, th2, th0, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th1, th2, th3, th4);
        result += T4(th5, th0, th1, th2, th3, th4);
        ;
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th2, th1, th3, th4);
        result += T4(th5, th0, th2, th1, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th1, th2, th0, th3, th4);
        result += T4(th5, th1, th2, th0, th3, th4);
    }
    return result;
}

double T5_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T5_total: Wrong number of aperture radii");
    };

    double th0 = thetas_123.at(0);
    double th1 = thetas_123.at(1);
    double th2 = thetas_123.at(2);
    double th3 = thetas_456.at(0);
    double th4 = thetas_456.at(1);
    double th5 = thetas_456.at(2);

    double result;

    if (th0 == th1)
    {
        if (th0 == th1)
        {
            if (th3 == th4 && th3 == th5)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th3, th3);
                result = 9 * T5(th0, th0, th0, th3, th3, th3);
            }
            else if (th3 == th4)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th3, th5);
                result = 6 * T5(th0, th0, th0, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th5, th3, th3);
                result += 3 * T5(th0, th0, th0, th5, th3, th3);
            }
            else if (th4 == th5)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th4, th4);
                result = 3 * T5(th0, th0, th0, th3, th4, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th4, th3, th4);
                result += 6 * T5(th0, th0, th0, th4, th3, th4);
            }
            else
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th4, th5);
                result = 3 * T5(th0, th0, th0, th3, th4, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th4, th3, th5);
                result += 3 * T5(th0, th0, th0, th4, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th5, th3, th4);
                result += 3 * T5(th0, th0, th0, th5, th3, th4);
            }
        }
        else
        {
            if (th3 == th4 && th3 == th5)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th3, th3);
                result = 6 * T5(th0, th0, th2, th3, th3, th3);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th3, th3, th3);
                result += 3 * T5(th2, th0, th0, th3, th3, th3);
            }
            else if (th3 == th4)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th3, th5);
                result = 4 * T5(th0, th0, th2, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th5, th3, th3);
                result += 2 * T5(th0, th0, th2, th5, th3, th3);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th3, th3, th5);
                result += 2 * T5(th2, th0, th0, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th5, th3, th3);
                result += T5(th2, th0, th0, th5, th3, th3);
            }
            else if (th4 == th5)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th4, th3, th4);
                result = 4 * T5(th0, th0, th2, th4, th3, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th4, th4);
                result += 2 * T5(th0, th0, th2, th3, th4, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th4, th3, th4);
                result += 2 * T5(th2, th0, th0, th4, th3, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th3, th4, th4);
                result += T5(th2, th0, th0, th3, th4, th4);
            }
            else
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th4, th5);
                result = 2 * T5(th0, th0, th2, th3, th4, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th4, th3, th5);
                result += 2 * T5(th0, th0, th2, th4, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th5, th3, th4);
                result += 2 * T5(th0, th0, th2, th5, th3, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th3, th4, th5);
                result += T5(th2, th0, th0, th3, th4, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th4, th3, th5);
                result += T5(th2, th0, th0, th4, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th5, th3, th4);
                result += T5(th2, th0, th0, th5, th3, th4);
            }
        }
    }
    else if (th3 == th4 && th3 == th5)
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th3, th3);
        result = 3 * T5(th0, th1, th2, th3, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th3, th3, th3);
        result += 3 * T5(th1, th0, th2, th3, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th3, th3, th3);
        result += 3 * T5(th2, th0, th1, th3, th3, th3);
    }
    else if (th3 == th4)
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th3, th5);
        result = 2 * T5(th0, th1, th2, th3, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th5, th3, th3);
        result += T5(th0, th1, th2, th5, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th3, th4, th5);
        result += 2 * T5(th1, th0, th2, th3, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th5, th3, th3);
        result += T5(th1, th0, th2, th5, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th3, th3, th5);
        result += 2 * T5(th2, th0, th1, th3, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th5, th3, th3);
        result += T5(th2, th0, th1, th5, th3, th3);
    }
    else if (th4 == th5)
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th4, th4);
        result = T5(th0, th1, th2, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th4, th3, th4);
        result += 2 * T5(th0, th1, th2, th4, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th3, th4, th4);
        result += T5(th1, th0, th2, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th4, th3, th4);
        result += 2 * T5(th1, th0, th2, th4, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th3, th4, th4);
        result += T5(th2, th0, th1, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th4, th3, th4);
        result += 2 * T5(th2, th0, th1, th4, th3, th4);
    }
    else
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th4, th5);
        result = T5(th0, th1, th2, th3, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th4, th3, th5);
        result += T5(th0, th1, th2, th4, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th5, th3, th4);
        result += T5(th0, th1, th2, th5, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th3, th4, th5);
        result += T5(th1, th0, th2, th3, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th4, th3, th5);
        result += T5(th1, th0, th2, th4, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th5, th3, th4);
        result += T5(th1, th0, th2, th5, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th3, th4, th5);
        result += T5(th2, th0, th1, th3, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th4, th3, th5);
        result += T5(th2, th0, th1, th4, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th5, th3, th4);
        result += T5(th2, th0, th1, th5, th3, th4);
    }

    return result;
}

double T6_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (type == 2)
        return 0;

    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T6_total: Wrong number of aperture radii");
    };

    double th0 = thetas_123.at(0);
    double th1 = thetas_123.at(1);
    double th2 = thetas_123.at(2);
    double th3 = thetas_456.at(0);
    double th4 = thetas_456.at(1);
    double th5 = thetas_456.at(2);

    double result;

    if (th0 == th1)
    {
        if (th0 == th2)
        {
            if (th3 == th4 && th3 == th5)
            {
                // A
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th3, th3);
                result = 3 * T6(th0, th0, th0, th3, th3, th3);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th0, th0, th3);
                result += 3 * T6(th3, th3, th0, th0, th0, th3);
            }
            else if (th3 == th4)
            {
                // B
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th3, th5);
                result = 3 * T6(th0, th0, th0, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th0, th0, th5);
                result += T6(th3, th3, th0, th0, th0, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th5, th0, th0, th0, th3);
                result += 2 * T6(th3, th5, th0, th0, th0, th3);
            }
            else if (th4 == th5)
            {
                // C
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th4, th4);
                result = 3 * T6(th0, th0, th0, th3, th4, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th4, th0, th0, th0, th4);
                result += 2 * T6(th3, th4, th0, th0, th0, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th4, th0, th0, th0, th3);
                result += T6(th4, th4, th0, th0, th0, th3);
            }
            else
            {
                // D
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th4, th5);
                result = 3 * T6(th0, th0, th0, th3, th4, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th4, th0, th0, th0, th5);
                result += T6(th3, th4, th0, th0, th0, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th5, th0, th0, th0, th4);
                result += T6(th3, th5, th0, th0, th0, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th5, th0, th0, th0, th3);
                result += T6(th4, th5, th0, th0, th0, th3);
            }
        }
        else
        {
            if (th3 == th4 && th3 == th5)
            {
                // E
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th3, th3);
                result = T6(th0, th0, th2, th3, th3, th3);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th0, th3, th3, th3);
                result += 2 * T6(th0, th2, th0, th3, th3, th3);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th0, th2, th3);
                result += 3 * T6(th3, th3, th0, th0, th2, th3);
            }
            else if (th3 == th4)
            {
                // F
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th3, th5);
                result = T6(th0, th0, th2, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th0, th3, th3, th5);
                result += 2 * T6(th0, th2, th0, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th0, th2, th5);
                result += T6(th3, th3, th0, th0, th2, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th5, th0, th0, th2, th3);
                result += 2 * T6(th3, th5, th0, th0, th2, th3);
            }
            else if (th4 == th5)
            {
                // G
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th4, th4);
                result = T6(th0, th0, th2, th3, th4, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th0, th3, th4, th4);
                result += 2 * T6(th0, th2, th0, th3, th4, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th4, th0, th0, th2, th4);
                result += 2 * T6(th3, th4, th0, th0, th2, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th4, th0, th0, th2, th3);
                result += T6(th4, th4, th0, th0, th2, th3);
            }
            else
            {
                // H
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th4, th5);
                result = T6(th0, th0, th2, th3, th4, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th0, th3, th4, th5);
                result += 2 * T6(th0, th2, th0, th3, th4, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th4, th0, th0, th2, th5);
                result += T6(th3, th4, th0, th0, th2, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th5, th0, th0, th2, th4);
                result += T6(th3, th5, th0, th0, th2, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th5, th0, th0, th2, th3);
                result += T6(th4, th5, th0, th0, th2, th3);
            }
        }
    }
    else if (th3 == th4 && th3 == th5)
    {
        // I
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th3, th3);
        result = T6(th0, th1, th2, th3, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th1, th3, th3, th3);
        result += T6(th0, th2, th1, th3, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th2, th0, th3, th3, th3);
        result += T6(th1, th2, th0, th3, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th1, th2, th3);
        result += 3 * T6(th3, th3, th0, th1, th2, th3);
    }
    else if (th3 == th4)
    {
        // J
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th3, th5);
        result = T6(th0, th1, th2, th3, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th1, th3, th3, th5);
        result += T6(th0, th2, th1, th3, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th2, th0, th3, th3, th5);
        result += T6(th1, th2, th0, th3, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th1, th2, th5);
        result += T6(th3, th3, th0, th1, th2, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th5, th0, th1, th2, th3);
        result += 2 * T6(th3, th5, th0, th1, th2, th3);
    }
    else if (th4 == th5)
    {
        // K
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th4, th4);
        result = T6(th0, th1, th2, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th1, th3, th4, th4);
        result += T6(th0, th2, th1, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th2, th0, th3, th4, th4);
        result += T6(th1, th2, th0, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th4, th0, th1, th2, th4);
        result += 2 * T6(th3, th4, th0, th1, th2, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th4, th0, th1, th2, th3);
        result += T6(th4, th4, th0, th1, th2, th3);
    }
    else
    {
        // L
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th4, th5);
        result = T6(th0, th1, th2, th3, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th1, th3, th4, th5);
        result += T6(th0, th2, th1, th3, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th2, th0, th3, th4, th5);
        result += T6(th1, th2, th0, th3, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th4, th0, th1, th2, th5);
        result += T6(th3, th4, th0, th1, th2, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th5, th0, th1, th2, th4);
        result += T6(th3, th5, th0, th1, th2, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th5, th0, th1, th2, th3);
        result += T6(th4, th5, th0, th1, th2, th3);
    }

    return result;
}

double T7_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T7_total: Wrong number of aperture radii");
    };

    double th0 = thetas_123.at(0);
    double th1 = thetas_123.at(1);
    double th2 = thetas_123.at(2);
    double th3 = thetas_456.at(0);
    double th4 = thetas_456.at(1);
    double th5 = thetas_456.at(2);

    double result;
    result = T7(th0, th1, th2, th3, th4, th5);

    return result;
}

double T7_2h_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (type == 2)
        return 0;

    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T7_2h_total: Wrong number of aperture radii");
    };

    double th0 = thetas_123.at(0);
    double th1 = thetas_123.at(1);
    double th2 = thetas_123.at(2);
    double th3 = thetas_456.at(0);
    double th4 = thetas_456.at(1);
    double th5 = thetas_456.at(2);

    double result;
    result = T7_2h(th0, th1, th2, th3, th4, th5);

    return result;
}

double T1(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
{
    // Set maximal l value such, that theta*l <= 10
    double thetaMin_123 = std::min({theta1, theta2, theta3});
    double thetaMin_456 = std::min({theta4, theta5, theta6});
    double thetaMin = std::max({thetaMin_123, thetaMin_456});
    double lMax = 10. / thetaMin;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));

    // Create container
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
    container.thetas_456 = std::vector<double>{theta4, theta5, theta6};

    // Do integration
    double result, error;
    if (type == 2) // Infinite
    {

        double vals_min[3] = {lMin, lMin, 0};
        double vals_max[3] = {lMax, lMax, M_PI}; // use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

        hcubature_v(1, integrand_T1, &container, 3, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
        result *= 2 / area / pow(2 * M_PI, 3); // Factors: 2 because phi integral goes from 0 to Pi, 1/area because division by area, (2pi)^-3 because 3 integrals in ell-space
    }
    else if (type == 0 || type == 1 || type == 3)
    {
        double vMax = lMax;
        double vals_min[6] = {-vMax, -vMax, -lMax, -lMax, -lMax, -lMax};
        double vals_max[6] = {1.02 * vMax, 1.02 * vMax, 1.02 * lMax, 1.02 * lMax, 1.02 * lMax, 1.02 * lMax};
        hcubature_v(1, integrand_T1, &container, 6, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);

        result /= pow(2 * M_PI, 6); // Factors: (2pi)^-6 because 6 integrals in ell-space
    }
    else
    {
        throw std::logic_error("T1: Wrong survey geometry");
    };

    return result;
}

double T2(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
{
    if (type == 2) // T2 is 0 for infinite survey
        return 0;

    // Integral over ell 1
    std::vector<double> thetas1{theta1, theta2};
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = thetas1;
    double thetaMin = std::min({theta1, theta2, theta3, theta4, theta5});
    double lMax = 10. / thetaMin;

    double result_A1, error_A1;

    double vals_min1[1] = {lMin};
    double vals_max1[1] = {lMax};
    hcubature_v(1, integrand_T2_part1, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_A1, &error_A1);

    result_A1 /= 2 * M_PI; // Division by 2pi because 1 integral in ell-space

    // Integral over ell 3
    std::vector<double> thetas2{theta5, theta6};
    container.thetas_123 = thetas2;
    thetaMin = std::min({theta5, theta6});
    lMax = 10. / thetaMin;

    double result_A2, error_A2;

    hcubature_v(1, integrand_T2_part1, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_A2, &error_A2);

    result_A2 /= 2 * M_PI; // Division by 2pi because 1 integral in ell-space

    // Integral over ell 2
    std::vector<double> thetas3{theta3, theta4};
    container.thetas_123 = thetas3;
    thetaMin = std::min({theta3, theta4});
    lMax = 10. / thetaMin;
    double result_B, error_B;
    if (type == 0)
    {
        hcubature_v(1, integrand_T2_part2, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_B, &error_B);
        result_B /= 2 * M_PI; // Division by 2pi because 1 integral in ell-space
    }
    else if (type == 1 || type == 3)
    {
        double vals_min2[2] = {-lMax, -lMax};
        double vals_max2[2] = {lMax, lMax};
        hcubature_v(1, integrand_T2_part2, &container, 2, vals_min2, vals_max2, 0, 0, 1e-4, ERROR_L1, &result_B, &error_B);
        result_B /= pow(2 * M_PI, 2); // Division by 2pi² because 2 integrals in ell-space
    }
    else
    {
        throw std::logic_error("T2: Wrong survey geometry");
    };

    return result_A1 * result_A2 * result_B;
}

double T4(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
{
    // Set maximal l value such, that theta*l <= 10
    double thetaMin_123 = std::min({theta1, theta2, theta3});
    double thetaMin_456 = std::min({theta4, theta5, theta6});
    double thetaMin = std::max({thetaMin_123, thetaMin_456});
    double lMax = 10. / thetaMin;
    lMin = 1e-5;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));

    // Create container
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
    container.thetas_456 = std::vector<double>{theta4, theta5, theta6};
    // Do integration
    double result, error;
    if (type == 2)
    {
        double vals_min[5] = {log(lMin), log(lMin), log(lMin), 0, 0};
        double vals_max[5] = {log(lMax), log(lMax), log(lMax), 2 * M_PI, 2 * M_PI};

        hcubature_v(1, integrand_T4, &container, 5, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
        result = result / area / pow(2 * M_PI, 5);
    }
    else
    {
        throw std::logic_error("T4: Wrong survey geometry, only coded for infinite survey");
    };

    return result;
}

double T5(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
{
    // Set maximal l value such, that theta*l <= 10
    double thetaMin_123 = std::min({theta1, theta2, theta3});
    double thetaMin_456 = std::min({theta4, theta5, theta6});
    double thetaMin = std::max({thetaMin_123, thetaMin_456});
    double lMax = 10. / thetaMin;
    lMin = 1e-4;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));

    // Create container
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
    container.thetas_456 = std::vector<double>{theta4, theta5, theta6};
    // Do integration
    double result, error;
    if (type == 2)
    {
        double mmin = pow(10, logMmin);
        double mmax = pow(10, logMmax);
        double vals_min[6] = {log(lMin), log(lMin), log(lMin), 0, 0, mmin};
        double vals_max[6] = {log(lMax), log(lMax), log(lMax), 2 * M_PI, 2 * M_PI, mmax};

        hcubature_v(1, integrand_T5, &container, 6, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
        result = result / area / pow(2 * M_PI, 5);
    }
    else
    {
        throw std::logic_error("T5: Wrong survey geometry, only coded for infinite survey");
    };

    return result;
}

double T6(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
{
    if ((type != 1) && (type != 3))
    {
        throw std::logic_error("T6: Wrong survey geometry, only coded for square and rectangular survey");
    };
    // Set maximal l value such, that theta*l <= 10
    double thetaMin_123 = std::min({theta1, theta2, theta3});
    double thetaMin_456 = std::min({theta4, theta5, theta6});
    double thetaMin = std::max({thetaMin_123, thetaMin_456});
    double lMax = 10. / thetaMin;
    lMin = 1e-4;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
    // Integral over ell 1
    std::vector<double> thetas1{theta1, theta2};
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = thetas1;

    double result_A1, error_A1;

    double vals_min1[1] = {lMin};
    double vals_max1[1] = {lMax};
    hcubature_v(1, integrand_T2_part1, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_A1, &error_A1);
    result_A1 /= (2 * M_PI);

    // Integral over ell3 to ell5

    // Create container

    container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
    container.thetas_456 = std::vector<double>{theta4, theta5, theta6};

    // Do integration
    double result, error;

    double mmin = pow(10, logMmin);
    double mmax = pow(10, logMmax);

    double vals_min[7] = {log(lMin), log(lMin), log(lMin), 0, 0, 0, mmin};
    double vals_max[7] = {log(lMax), log(lMax), log(lMax), 2 * M_PI, 2 * M_PI, 2 * M_PI, mmax};

    hcubature_v(1, integrand_T6, &container, 7, vals_min, vals_max, 0, 0, 1e-1, ERROR_L1, &result, &error);

    result /= pow(2 * M_PI, 6);
    return result_A1 * result;
}

double T7(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
{
    // Set maximal l value such, that theta*l <= 10
    double thetaMin_123 = std::min({theta1, theta2, theta3});
    double thetaMin_456 = std::min({theta4, theta5, theta6});
    double thetaMin = std::min({thetaMin_123, thetaMin_456});
    double lMax = 10. / thetaMin;
    lMin = 1e-4;

    // lMax *= 200;
    // lMin /= 2;
    std::cerr << lMin << " " << lMax << std::endl;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));

    // Create container
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
    container.thetas_456 = std::vector<double>{theta4, theta5, theta6};

    // Do integration
    double result, error;
    if (type == 2)
    {
        double mmin = pow(10, logMmin);
        double mmax = pow(10, logMmax);
        double vals_min[7] = {log(lMin), log(lMin), log(lMin), log(lMin), 0, 0, mmin};
        double vals_max[7] = {log(lMax), log(lMax), log(lMax), log(lMax), 2 * M_PI, 2 * M_PI, mmax};

        hcubature_v(1, integrand_T7, &container, 7, vals_min, vals_max, 0, 0, 1e-1, ERROR_L1, &result, &error);
        result = result / area / pow(2 * M_PI, 6);
    }
    else
    {
        throw std::logic_error("T7: Wrong survey geometry, only coded for infinite survey");
    };

    return result;
}

double T7_2h(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
{
    // Set maximal l value such, that theta*l <= 10
    double thetaMin_123 = std::max({theta1, theta2, theta3});
    double thetaMin_456 = std::max({theta4, theta5, theta6});
    double thetaMin = std::max({thetaMin_123, thetaMin_456});
    double lMax = 10. / thetaMin;
    lMin = 1e-4;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));

    // Create container
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
    container.thetas_456 = std::vector<double>{theta4, theta5, theta6};
    container.mMin = pow(10, logMmin);
    container.mMax = pow(10, logMmax);
    container.zMin = 0;
    container.zMax = z_max;

    // Do integration
    double result, error;
    if (type == 1)
    {
        double vals_min[6] = {log(lMin), log(lMin), 0, log(lMin), log(lMin), 0};
        double vals_max[6] = {log(lMax), log(lMax), 2 * M_PI, log(lMax), log(lMax), 2 * M_PI};

        hcubature_v(1, integrand_T7_2h, &container, 6, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
        result = result / pow(2 * M_PI, 6);
    }
    else
    {
        throw std::logic_error("T7_SSC: Wrong survey geometry, only coded for square survey");
    };

    return result;
}


int integrand_T1(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T1: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    // Allocate memory on device for integrand values
    double *dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

    // Copy integration variables to device
    double *dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

    // Calculate values
    if (type == 0)
    {
        integrand_T1_circle<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                 container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                 container_->thetas_456.at(0), container_->thetas_456.at(1),
                                                 container_->thetas_456.at(2), dev_value);
    }
    else if (type == 1)
    {
        integrand_T1_square<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                 container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                 container_->thetas_456.at(0), container_->thetas_456.at(1),
                                                 container_->thetas_456.at(2), dev_value);
    }
    else if (type == 2)
    {
        integrand_T1_infinite<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                   container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                   container_->thetas_456.at(0), container_->thetas_456.at(1),
                                                   container_->thetas_456.at(2), dev_value);
    }
    else if (type == 3)
    {
        integrand_T1_rectangle<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                    container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                    container_->thetas_456.at(0), container_->thetas_456.at(1),
                                                    container_->thetas_456.at(2), dev_value);
    }
    else // This should not happen
    {
        exit(-1);
    };

    cudaFree(dev_vars); // Free variables

    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); // Free values

    return 0; // Success :)
}

int integrand_T2_part1(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T2_part1: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    // Allocate memory on device for integrand values
    double *dev_value;

    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

    // Copy integration variables to device
    double *dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

    // Calculate values
    if (type == 0 || type == 1)
    {
        integrand_T2_part1<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                container_->thetas_123.at(1), dev_value);
    }
    else // This should not happen
    {
        exit(-1);
    };

    cudaFree(dev_vars); // Free variables

    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); // Free values

    return 0; // Success :)
}

int integrand_T2_part2(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T1: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    // Allocate memory on device for integrand values
    double *dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

    // Copy integration variables to device
    double *dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

    // Calculate values
    if (type == 0)
    {
        integrand_T2_part2_circle<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                       container_->thetas_123.at(1), dev_value);
    }
    else if (type == 1)
    {
        integrand_T2_part2_square<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                       container_->thetas_123.at(1), dev_value);
    }
    else if (type == 3)
    {
        integrand_T2_part2_rectangle<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                          container_->thetas_123.at(1), dev_value);
    }
    else // This should not happen
    {
        exit(-1);
    };

    cudaFree(dev_vars); // Free variables

    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); // Free values

    return 0; // Success :)
}

int integrand_T4(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T1: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    int Nmax = 1e5; // Maximal number of points per iteration

    int Niter = int(npts / Nmax) + 1; // Number of iterations

    for (int i = 0; i < Niter; i++)
    {

        int npts_iter;
        if (i == Niter - 1)
        {
            npts_iter = npts - (Niter - 1) * Nmax;
        }
        else
        {
            npts_iter = Nmax;
        };

        double vars_iter[npts_iter * ndim];
        for (int j = 0; j < npts_iter; j++)
        {
            for (int k = 0; k < ndim; k++)
            {
                vars_iter[j * ndim + k] = vars[i * Nmax * ndim + j * ndim + k];
            }
        };

        // Allocate memory on device for integrand values
        double *dev_value_iter;
        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));

        // Copy integration variables to device
        double *dev_vars_iter;
        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying

        // Calculate values
        if (type == 2)
        {
            integrand_T4_infinite<<<BLOCKS, THREADS>>>(dev_vars_iter, ndim, npts_iter, container_->thetas_123.at(0),
                                                       container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                       container_->thetas_456.at(0), container_->thetas_456.at(1),
                                                       container_->thetas_456.at(2), dev_value_iter);
        }
        else // This should not happen
        {
            std::cerr << "Something went wrong in integrand_T4." << std::endl;
            exit(-1);
        };
        cudaFree(dev_vars_iter); // Free variables

        double value_iter[npts_iter];
        // Copy results to host
        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));

        cudaFree(dev_value_iter);

        for (int j = 0; j < npts_iter; j++)
        {
            value[i * Nmax + j] = value_iter[j];
        }
    }
    return 0; // Success :)
}

int integrand_T5(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T5: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    int Nmax = 1e5;
    int Niter = int(npts / Nmax) + 1; // Number of iterations

    for (int i = 0; i < Niter; i++)
    {

        int npts_iter;
        if (i == Niter - 1)
        {
            npts_iter = npts - (Niter - 1) * Nmax;
        }
        else
        {
            npts_iter = Nmax;
        };

        double vars_iter[npts_iter * ndim];
        for (int j = 0; j < npts_iter; j++)
        {
            for (int k = 0; k < ndim; k++)
            {
                vars_iter[j * ndim + k] = vars[i * Nmax * ndim + j * ndim + k];
            }
        };

        // Allocate memory on device for integrand values
        double *dev_value_iter;
        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));

        // Copy integration variables to device
        double *dev_vars_iter;
        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying

        // Calculate values
        if (type == 2)
        {
            integrand_T5_infinite<<<BLOCKS, THREADS>>>(dev_vars_iter, ndim, npts_iter, container_->thetas_123.at(0),
                                                       container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                       container_->thetas_456.at(0), container_->thetas_456.at(1),
                                                       container_->thetas_456.at(2), dev_value_iter);
        }
        else // This should not happen
        {
            std::cerr << "Something went wrong in integrand_T5." << std::endl;
            exit(-1);
        };

        cudaFree(dev_vars_iter); // Free variables

        double value_iter[npts_iter];
        // Copy results to host
        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));

        cudaFree(dev_value_iter); // Free values

        for (int j = 0; j < npts_iter; j++)
        {
            value[i * Nmax + j] = value_iter[j];
        }
    }
    return 0; // Success :)
}

int integrand_T6(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T6: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    int Nmax = 1e5;
    int Niter = int(npts / Nmax) + 1; // Number of iterations

    for (int i = 0; i < Niter; i++)
    {

        int npts_iter;
        if (i == Niter - 1)
        {
            npts_iter = npts - (Niter - 1) * Nmax;
        }
        else
        {
            npts_iter = Nmax;
        };

        double vars_iter[npts_iter * ndim];
        for (int j = 0; j < npts_iter; j++)
        {
            for (int k = 0; k < ndim; k++)
            {
                vars_iter[j * ndim + k] = vars[i * Nmax * ndim + j * ndim + k];
            }
        };

        // Allocate memory on device for integrand values
        double *dev_value_iter;
        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));

        // Copy integration variables to device
        double *dev_vars_iter;
        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying

        // Calculate values
        if (type == 1)
        {
            integrand_T6_square<<<BLOCKS, THREADS>>>(dev_vars_iter, ndim, npts_iter, container_->thetas_123.at(0),
                                                     container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                     container_->thetas_456.at(0), container_->thetas_456.at(1),
                                                     container_->thetas_456.at(2), dev_value_iter);
        }
        else if (type == 3)
        {
            integrand_T6_rectangle<<<BLOCKS, THREADS>>>(dev_vars_iter, ndim, npts_iter, container_->thetas_123.at(0),
                                                        container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                        container_->thetas_456.at(0), container_->thetas_456.at(1),
                                                        container_->thetas_456.at(2), dev_value_iter);
        }
        else // This should not happen
        {
            std::cerr << "Something went wrong in integrand_T6" << std::endl;
            exit(-1);
        };

        cudaFree(dev_vars_iter); // Free variables

        double value_iter[npts_iter];

        // Copy results to host
        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));

        cudaFree(dev_value_iter); // Free values

        for (int j = 0; j < npts_iter; j++)
        {
            value[i * Nmax + j] = value_iter[j];
        }
    }
    return 0; // Success :)
}

int integrand_T7(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T7: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    int Nmax = 1e5;
    int Niter = int(npts / Nmax) + 1; // Number of iterations
    std::cerr << npts << std::endl;
    for (int i = 0; i < Niter; i++)
    {

        int npts_iter;
        if (i == Niter - 1)
        {
            npts_iter = npts - (Niter - 1) * Nmax;
        }
        else
        {
            npts_iter = Nmax;
        };

        double vars_iter[npts_iter * ndim];
        for (int j = 0; j < npts_iter; j++)
        {
            for (int k = 0; k < ndim; k++)
            {
                vars_iter[j * ndim + k] = vars[i * Nmax * ndim + j * ndim + k];
            }
        };

        // Allocate memory on device for integrand values
        double *dev_value_iter;
        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));

        // Copy integration variables to device
        double *dev_vars_iter;
        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying

        // Calculate values
        if (type == 2)
        {
            double mMin = pow(10, logMmin);
            double mMax = pow(10, logMmax);
            integrand_T7_infinite<<<BLOCKS, THREADS>>>(dev_vars_iter, ndim, npts_iter, container_->thetas_123.at(0),
                                                       container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                       container_->thetas_456.at(0), container_->thetas_456.at(1),
                                                       container_->thetas_456.at(2), dev_value_iter, mMin, mMax);
        }
        else // This should not happen
        {
            std::cerr << "Something went wrong in integrand_T7." << std::endl;
            exit(-1);
        };

        cudaFree(dev_vars_iter); // Free variables

        double value_iter[npts_iter];
        // Copy results to host
        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));

        cudaFree(dev_value_iter); // Free values

        for (int j = 0; j < npts_iter; j++)
        {
            value[i * Nmax + j] = value_iter[j];
        }
    }

    return 0; // Success :)
}

int integrand_T7_2h(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T7_2h: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    int Nmax = 1e5;
    int Niter = int(npts / Nmax) + 1; // Number of iterations
    std::cerr << npts << std::endl;
    for (int i = 0; i < Niter; i++)
    {

        int npts_iter;
        if (i == Niter - 1)
        {
            npts_iter = npts - (Niter - 1) * Nmax;
        }
        else
        {
            npts_iter = Nmax;
        };

        double vars_iter[npts_iter * ndim];
        for (int j = 0; j < npts_iter; j++)
        {
            for (int k = 0; k < ndim; k++)
            {
                vars_iter[j * ndim + k] = vars[i * Nmax * ndim + j * ndim + k];
            }
        };

        // Allocate memory on device for integrand values
        double *dev_value_iter;
        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));

        // Copy integration variables to device
        double *dev_vars_iter;
        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying

        integrand_T7_2h<<<BLOCKS, THREADS>>>(dev_vars_iter, ndim, npts_iter,
                                              container_->thetas_123.at(0), container_->thetas_123.at(1), container_->thetas_123.at(2),
                                              container_->thetas_456.at(0), container_->thetas_456.at(1),
                                              container_->thetas_456.at(2), dev_value_iter, container_->mMin, container_->mMax,
                                              container_->zMin, container_->zMax);

        cudaFree(dev_vars_iter); // Free variables

        double value_iter[npts_iter];
        // Copy results to host
        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));

        cudaFree(dev_value_iter); // Free values

        for (int j = 0; j < npts_iter; j++)
        {
            value[i * Nmax + j] = value_iter[j];
        }
    }

    return 0; // Success :)
}

__global__ void integrand_T1_circle(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double a = vars[i * ndim];
        double b = vars[i * ndim + 1];
        double c = vars[i * ndim + 2];
        double d = vars[i * ndim + 3];
        double e = vars[i * ndim + 4];
        double f = vars[i * ndim + 5];

        double ell = sqrt(a * a + b * b);
        double Gfactor = G_circle(ell);

        double ell1 = sqrt((a - c - e) * (a - c - e) + (b - d - f) * (b - d - f));
        double ell2 = sqrt(c * c + d * d);
        double ell3 = sqrt(e * e + f * f);

        if (ell1 <= dev_lMin || ell2 <= dev_lMin || ell3 <= dev_lMin)
        {
            value[i] = 0;
        }
        else
        {
            double result = Pell(ell1) * Pell(ell2) * Pell(ell3);
            result *= uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3 * theta3);
            result *= uHat(ell1 * theta4) * uHat(ell2 * theta5) * uHat(ell3 * theta6);
            result *= Gfactor;

            value[i] = result;
        };
    }
}

__global__ void integrand_T1_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double a = vars[i * ndim];
        double b = vars[i * ndim + 1];
        double c = vars[i * ndim + 2];
        double d = vars[i * ndim + 3];
        double e = vars[i * ndim + 4];
        double f = vars[i * ndim + 5];

        double Gfactor = G_square(a, b);

        double ell1 = sqrt((a - c - e) * (a - c - e) + (b - d - f) * (b - d - f));
        double ell2 = sqrt(c * c + d * d);
        double ell3 = sqrt(e * e + f * f);

        if (ell1 <= 0 || ell2 <= 0 || ell3 <= 0 || ell3 > dev_lMax)
        {
            value[i] = 0;
        }
        else
        {
            double result = Pell(ell1) * Pell(ell2) * Pell(ell3);
            result *= uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3 * theta3);
            result *= uHat(ell1 * theta4) * uHat(ell2 * theta5) * uHat(ell3 * theta6);
            result *= Gfactor;

            value[i] = result;
        };
    }
}

__global__ void integrand_T1_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l1 = vars[i * ndim];
        double l2 = vars[i * ndim + 1];
        double phi = vars[i * ndim + 2];

        double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));

        if (l1 <= dev_lMin || l2 <= dev_lMin || l3 <= dev_lMin || l3 > dev_lMax)
        {
            value[i] = 0;
        }
        else
        {
            double result = Pell(l1) * Pell(l2) * Pell(l3);
            result *= uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3);
            result *= uHat(l1 * theta4) * uHat(l2 * theta5) * uHat(l3 * theta6);
            result *= l1 * l2;
            value[i] = result;
        };
    }
}

__global__ void integrand_T1_rectangle(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double a = vars[i * ndim];
        double b = vars[i * ndim + 1];
        double c = vars[i * ndim + 2];
        double d = vars[i * ndim + 3];
        double e = vars[i * ndim + 4];
        double f = vars[i * ndim + 5];

        double Gfactor = G_rectangle(a, b);

        double ell1 = sqrt((a - c - e) * (a - c - e) + (b - d - f) * (b - d - f));
        double ell2 = sqrt(c * c + d * d);
        double ell3 = sqrt(e * e + f * f);

        if (ell1 <= 0 || ell2 <= 0 || ell3 <= 0 || ell3 > dev_lMax)
        {
            value[i] = 0;
        }
        else
        {
            double result = Pell(ell1) * Pell(ell2) * Pell(ell3);
            result *= uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3 * theta3);
            result *= uHat(ell1 * theta4) * uHat(ell2 * theta5) * uHat(ell3 * theta6);
            result *= Gfactor;

            value[i] = result;
        };
    }
}

__global__ void integrand_T2_part1(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double ell = vars[i * ndim];

        if (ell < dev_lMin)
        {
            value[i] = 0;
        }
        else
        {
            double result = ell * Pell(ell);
            result *= uHat(ell * theta1) * uHat(ell * theta2);
            value[i] = result;
        }
    }
}

__global__ void integrand_T2_part2_circle(const double *vars, unsigned ndim, int npts, double theta1, double theta2,
                                          double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double ell = vars[i * ndim];

        if (ell < dev_lMin)
        {
            value[i] = 0;
        }
        else
        {
            double Gfactor = G_circle(ell);
            double result = Pell(ell);
            result *= ell * uHat(ell * theta1) * uHat(ell * theta2);
            result *= Gfactor;
            value[i] = result;
        };
    }
}

__global__ void integrand_T2_part2_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2,
                                          double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double ellX = vars[i * ndim];
        double ellY = vars[i * ndim + 1];

        double Gfactor = G_square(ellX, ellY);
        double ell = sqrt(ellX * ellX + ellY * ellY);

        if (ell < dev_lMin)
        {
            value[i] = 0;
        }
        else
        {
            double result = Pell(ell);
            result *= uHat(ell * theta1) * uHat(ell * theta2);
            result *= Gfactor;
            value[i] = result;
        };
    }
}

__global__ void integrand_T2_part2_rectangle(const double *vars, unsigned ndim, int npts, double theta1, double theta2,
                                             double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double ellX = vars[i * ndim];
        double ellY = vars[i * ndim + 1];

        double Gfactor = G_rectangle(ellX, ellY);
        double ell = sqrt(ellX * ellX + ellY * ellY);

        if (ell < dev_lMin)
        {
            value[i] = 0;
        }
        else
        {
            double result = Pell(ell);
            result *= uHat(ell * theta1) * uHat(ell * theta2);
            result *= Gfactor;
            value[i] = result;
        };
    }
}

__global__ void integrand_T4_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l1 = exp(vars[i * ndim]);
        double l2 = exp(vars[i * ndim + 1]);
        double l5 = exp(vars[i * ndim + 2]);
        double phi1 = vars[i * ndim + 3];
        double phi2 = vars[i * ndim + 4];

        double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi1));
        double l6 = sqrt(l1 * l1 + l5 * l5 + 2 * l1 * l5 * cos(phi2));

        if (l3 > dev_lMax || l6 > dev_lMax || l3 <= 0 || l6 <= 0)
        {
            value[i] = 0;
        }
        else
        {
            double result = uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l1 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);

            result *= bkappa(l1, l2, l3);
            result *= bkappa(l1, l5, l6);
            result *= l1 * l2 * l5;
            result *= l1 * l2 * l5;

            value[i] = result;
        }
    }
}

__global__ void integrand_T5_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l1 = exp(vars[i * ndim]);
        double l2 = exp(vars[i * ndim + 1]);
        double l5 = exp(vars[i * ndim + 2]);
        double phi1 = vars[i * ndim + 3];
        double phi2 = vars[i * ndim + 4];
        double m = vars[i * ndim + 5];

        double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi1));
        double l6 = sqrt(l1 * l1 + l5 * l5 + 2 * l1 * l5 * cos(phi2));
        double l4 = l1;

        if (l1 <= dev_lMin || l2 <= dev_lMin || l3 <= dev_lMin || l4 <= dev_lMin || l5 <= dev_lMin || l6 <= dev_lMin || l3 > dev_lMax || l6 > dev_lMax)
        {
            value[i] = 0;
        }
        else
        {

            double result = uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l4 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);

            result *= Pell(l1);
            result *= trispectrum_limber_integrated(0, dev_z_max, m, l2, l3, l5, l6);
            result *= l1 * l2 * l5;
            result *= l1 * l2 * l5;

            value[i] = result;
        }
    }
}

__global__ void integrand_T6_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l3 = exp(vars[i * ndim]);
        double l4 = exp(vars[i * ndim + 1]);
        double l5 = exp(vars[i * ndim + 2]);
        double phi3 = vars[i * ndim + 3];
        double phi4 = vars[i * ndim + 4];
        double phi5 = vars[i * ndim + 5];
        double m = vars[i * ndim + 6];

        double l3x = l3 * cos(phi3);
        double l3y = l3 * sin(phi3);
        double l6 = l3 * l3 + l4 * l4 + l5 * l5 + 2 * (l3 * l4 * cos(phi3 - phi4) - l3 * l5 * cos(phi3 - phi5) - l4 * l5 * cos(phi4 - phi5));
        if (l6 > 0)
        {
            l6 = sqrt(l6);
        }
        else
        {
            l6 = 0;
        }

        if (l3 <= dev_lMin || l4 <= dev_lMin || l5 <= dev_lMin || l6 <= dev_lMin || l6 > dev_lMax)
        {
            value[i] = 0;
        }
        else
        {
            double Gfactor = G_square(l3x, l3y);
            double result = uHat(l3 * theta3) * uHat(l4 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);

            double trispec = trispectrum_limber_integrated(0, dev_z_max, m, l3, l4, l5, l6);
            result *= trispec;
            result *= l3 * l4 * l5;
            result *= l3 * l4 * l5;
            result *= Gfactor;

            value[i] = result;
        }
    }
}

__global__ void integrand_T6_rectangle(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l3 = exp(vars[i * ndim]);
        double l4 = exp(vars[i * ndim + 1]);
        double l5 = exp(vars[i * ndim + 2]);
        double phi3 = vars[i * ndim + 3];
        double phi4 = vars[i * ndim + 4];
        double phi5 = vars[i * ndim + 5];
        double m = vars[i * ndim + 6];

        double l3x = l3 * cos(phi3);
        double l3y = l3 * sin(phi3);
        double l6 = l3 * l3 + l4 * l4 + l5 * l5 + 2 * (l3 * l4 * cos(phi3 - phi4) - l3 * l5 * cos(phi3 - phi5) - l4 * l5 * cos(phi4 - phi5));
        if (l6 > 0)
        {
            l6 = sqrt(l6);
        }
        else
        {
            l6 = 0;
        }

        if (l3 <= dev_lMin || l4 <= dev_lMin || l5 <= dev_lMin || l6 <= dev_lMin || l6 > dev_lMax)
        {
            value[i] = 0;
        }
        else
        {
            double Gfactor = G_rectangle(l3x, l3y);
            double result = uHat(l3 * theta3) * uHat(l4 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);

            double trispec = trispectrum_limber_integrated(0, dev_z_max, m, l3, l4, l5, l6);
            result *= trispec;
            result *= l3 * l4 * l5;
            result *= l3 * l4 * l5;
            result *= Gfactor;

            value[i] = result;
        }
    }
}

__global__ void integrand_T7_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double *value, double mMin, double mMax)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l1 = exp(vars[i * ndim]);
        double l2 = exp(vars[i * ndim + 1]);
        double l4 = exp(vars[i * ndim + 2]);
        double l5 = exp(vars[i * ndim + 3]);
        double phi1 = vars[i * ndim + 4];
        double phi2 = vars[i * ndim + 5];
        double m = vars[i * ndim + 6];

        double l3 = (l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi1));
        double l6 = (l4 * l4 + l5 * l5 + 2 * l4 * l5 * cos(phi2));

        if (l3 <= 0 || l6 <= 0)
        {
            value[i] = 0;
        }
        else
        {
            l3 = sqrt(l3);
            l6 = sqrt(l6);
            double result = uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l4 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);

            double pentaspec = pentaspectrum_limber_integrated(0, dev_z_max, m, l1, l2, l3, l4, l5, l6);
            result *= pentaspec;
            result *= l1 * l2 * l4 * l5;
            result *= l1 * l2 * l4 * l5;
            value[i] = result;
        }
    }
}


__global__ void integrand_T7_2h(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3,
                                 double theta4, double theta5, double theta6,
                                 double *value, double mMin, double mMax, double zMin, double zMax)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l1 = exp(vars[i * ndim]);
        double l2 = exp(vars[i * ndim + 1]);
        double phi1 = (vars[i * ndim + 2]);
        double l4 = exp(vars[i * ndim + 3]);
        double l5 = exp(vars[i * ndim + 4]);
        double phi2 = vars[i * ndim + 5];

        double l3 = (l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi1));
        double l6 = (l4 * l4 + l5 * l5 + 2 * l4 * l5 * cos(phi2));

        if (l3 <= 0 || l6 <= 0)
        {
            value[i] = 0;
        }
        else
        {
            l3 = sqrt(l3);
            l6 = sqrt(l6);
            double result = uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l4 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);

            double pentaspec = pentaspectrum_limber_integrated_ssc(zMin, zMax, mMin, mMax, l1, l2, l3, l4, l5, l6);
            result *= pentaspec;
            result *= l1 * l2 * l4 * l5;
            result *= l1 * l2 * l4 * l5;
            value[i] = result;
        }
    }
}

/************************** FOR <Map²> COVARIANCE ***************************************/

double Cov_Map2_Gauss(const double &theta1, const double &theta2)
{
    double thetaMin = std::min({theta1, theta2});

    double lMax = 10. / thetaMin;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));

    CovMap2Container container;
    container.theta1 = theta1;
    container.theta2 = theta2;

    double result, error;
    if (type == 2)
    {
        double vals_min[1] = {lMin};
        double vals_max[1] = {lMax};

        hcubature_v(1, integrand_Cov_Map2_Gauss, &container, 1, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
        result /= (2 * M_PI) * area;
    }
    else if (type == 1)
    {
        double vals_min[4] = {-1.01 * lMax, -1.01 * lMax, -1.01 * lMax, -1.01 * lMax};
        double vals_max[4] = {lMax, lMax, lMax, lMax};
        hcubature_v(1, integrand_Cov_Map2_Gauss, &container, 4, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);
        result /= pow(2 * M_PI, 4);
    }
    else
    {
        throw std::logic_error("Cov_Map2_Gauss: Wrong survey geometry");
    };

    return result;
}

double Cov_Map2_NonGauss(const double &theta1, const double &theta2)
{
    double thetaMin = std::min({theta1, theta2});

    double lMax = 10. / thetaMin;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));

    CovMap2Container container;
    container.theta1 = theta1;
    container.theta2 = theta2;

    double mmin = pow(10, logMmin);
    double mmax = pow(10, logMmax);

    double result, error;
    if (type == 2)
    {
        double vals_min[3] = {lMin, lMin, mmin};
        double vals_max[3] = {lMax, lMax, mmax};

        hcubature_v(1, integrand_Cov_Map2_NonGauss, &container, 3, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

        result /= pow(2 * M_PI, 2) * area;
    }
    else if (type == 1)
    {
        double vals_min[6] = {-1.01 * lMax, -1.01 * lMax, -1.01 * lMax, -1.01 * lMax, -1.01 * lMax, -1.01 * lMax};
        double vals_max[6] = {lMax, lMax, lMax, lMax, lMax, lMax};
        hcubature_v(1, integrand_Cov_Map2_NonGauss, &container, 6, vals_min, vals_max, 1e4, 0, 1e-1, ERROR_L1, &result, &error);
        result /= pow(2 * M_PI, 6);
    }
    else
    {
        throw std::logic_error("Cov_Map2_NonGauss: Wrong survey geometry");
    };

    return result;
}

int integrand_Cov_Map2_Gauss(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_Cov_Map2_Gauss: Wrong function dimension" << std::endl;
        return -1;
    };

    CovMap2Container *container_ = (CovMap2Container *)container;

    // Allocate memory on device for integrand values
    double *dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

    // Copy integration variables to device
    double *dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

    // Calculate values
    if (type == 1)
    {
        integrand_Cov_Map2_Gauss_square<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->theta1,
                                                             container_->theta2, dev_value);
    }
    else if (type == 2)
    {
        integrand_Cov_Map2_Gauss_infinite<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->theta1,
                                                               container_->theta2, dev_value);
    }
    else // This should not happen
    {
        exit(-1);
    };

    cudaFree(dev_vars); // Free variables

    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); // Free values

    return 0; // Success :)
}

int integrand_Cov_Map2_NonGauss(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_Cov_Map2_NonGauss: Wrong function dimension" << std::endl;
        return -1;
    };

    CovMap2Container *container_ = (CovMap2Container *)container;

    // Allocate memory on device for integrand values
    double *dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

    // Copy integration variables to device
    double *dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

    // Calculate values
    if (type == 1)
    {
        integrand_Cov_Map2_NonGauss_square<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->theta1,
                                                                container_->theta2, dev_value);
    }
    else if (type == 2)
    {
        integrand_Cov_Map2_NonGauss_infinite<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->theta1,
                                                                  container_->theta2, dev_value);
    }
    else // This should not happen
    {
        exit(-1);
    };

    cudaFree(dev_vars); // Free variables

    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); // Free values

    return 0; // Success :)
}

__global__ void integrand_Cov_Map2_Gauss_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double ell = vars[i * ndim];
        double result = Pell(ell);
        result *= uHat(ell * theta1) * uHat(ell * theta2);
        result *= result * 2;
        result *= ell;
        value[i] = result;
    }
}

__global__ void integrand_Cov_Map2_Gauss_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double v_x = vars[i * ndim];
        double v_y = vars[i * ndim + 1];
        double ell2_x = vars[i * ndim + 2];
        double ell2_y = vars[i * ndim + 3];

        double Gfactor = G_square(v_x, v_y);

        double ell1_x = v_x - ell2_x;
        double ell1_y = v_y - ell2_y;
        double ell1 = sqrt(ell1_x * ell1_x + ell1_y * ell1_y);
        double ell2 = sqrt(ell2_x * ell2_x + ell2_y * ell2_y);

        double result = Pell(ell1) * Pell(ell2) * Gfactor;
        result *= uHat(ell1 * theta1) * uHat(ell1 * theta2) * uHat(ell2 * theta1) * uHat(ell2 * theta2);
        result *= 2;
        value[i] = result;
    }
}

__global__ void integrand_Cov_Map2_NonGauss_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double ell1 = vars[i * ndim];
        double ell2 = vars[i * ndim + 1];
        double m = vars[i * ndim + 2];

        double result = uHat(ell1 * theta1) * uHat(ell2 * theta2);
        result *= result;

        double trispec = trispectrum_limber_integrated(0, dev_z_max, m, ell1, ell1, ell2, ell2);
        result *= trispec;
        result *= ell1 * ell2;
        value[i] = result;
    }
}

__global__ void integrand_Cov_Map2_NonGauss_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l1x = vars[i * ndim];
        double l1y = vars[i * ndim + 1];
        double l2x = vars[i * ndim + 2];
        double l2y = vars[i * ndim + 3];
        double l3x = vars[i * ndim + 4];
        double l3y = vars[i * ndim + 5];

        double qx = l2x + l1x;
        double qy = l2y + l1y;
        double l4x = l3x + qx;
        double l4y = l3y + qy;

        double ell1 = sqrt(l1x * l1x + l1y * l1y);
        double ell2 = sqrt(l2x * l2x + l2y * l2y);
        double ell3 = sqrt(l3x * l3x + l3y * l3y);
        double ell4 = sqrt(l4x * l4x + l4y * l4y);

        if (ell1 <= dev_lMin || ell2 <= dev_lMin || ell3 <= dev_lMin || ell4 <= dev_lMin)
        {
            value[i] = 0;
        }
        else
        {

            double result = uHat(ell1 * theta1) * uHat(ell2 * theta1) * uHat(ell3 * theta2) * uHat(ell4 * theta2);

            double trispec = 0;
            trispec += trispectrum_2halo(0.001, dev_z_max, pow(10, logMmin), pow(10, logMmax), l1x, l1y, l2x, l2y, l3x, l3y, l4x, l4y);

            double Gfactor = G_square(qx, qy);
            result *= trispec;
            result *= Gfactor;
            value[i] = result;
            // if(result<0) printf("%e %e %e %e %e %e %e \n", result, trispec, Gfactor, ell1, ell2, ell3, ell4);
        };
    }
}

/************************** FOR <Map²-Map3> CROSS-COVARIANCE ***************************************/

double Map2Map3_T2_total(const std::vector<double> &thetas_123, const double &theta_4)
{
    if (thetas_123.size() != 3)
    {
        throw std::invalid_argument("Map2Map3_T2_total: Wrong number of aperture radii");
    };

    double th0 = thetas_123.at(0);
    double th1 = thetas_123.at(1);
    double th2 = thetas_123.at(2);
    double th3 = theta_4;

    double result = 0;

    // TO DO: BY CHECKING IF TWO THETAS ARE THE SAME, THE NUMBER OF INDIVIDUAL PERMUATIONS TO
    // BE CALCULATED CAN BE REDUCED!

    printf("Doing %f, %f, %f, %f \n", th0, th1, th2, th3);
    result += Map2Map3_T2(th0, th1, th2, th3);

    printf("Doing %f, %f, %f, %f \n", th1, th0, th2, th3);
    result += Map2Map3_T2(th1, th0, th2, th3);

    printf("Doing %f, %f, %f, %f \n", th2, th0, th1, th3);
    result += Map2Map3_T2(th2, th0, th1, th3);

    return result;
}

double Map2Map3_T2(const double &theta1, const double &theta2, const double &theta3, const double &theta4)
{
    if (type == 2)
        return 0;

    // Integral over ell 1
    std::vector<double> thetas1{theta2, theta3};
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = thetas1;
    double thetaMin = std::min({theta2, theta3});
    double lMax = 10. / thetaMin;                                        // Determine ell-Boundary
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double))); // Copies the value of lMax to the GPU

    double result_1, error_1;

    double vals_min1[1] = {lMin};
    double vals_max1[1] = {lMax};
    hcubature_v(1, integrand_T2_part1, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_1, &error_1); // Do integration

    result_1 /= 2 * M_PI; // Division by 2pi because 1 integral in ell-space

    // Integral over ell 2 and ell 3
    std::vector<double> thetas2{theta1, theta4};
    container.thetas_123 = thetas2;
    thetaMin = std::max({theta1, theta4});
    lMax = 10. / thetaMin;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));

    // lMax=1e3;
    double result_2, error_2;

    double vals_min2[4] = {-lMax / sqrt(2), -lMax / sqrt(2), -lMax / sqrt(2), -lMax / sqrt(2)};
    double vals_max2[4] = {lMax / sqrt(2), lMax / sqrt(2), lMax / sqrt(2), lMax / sqrt(2)};

    if (type == 1)
    {
        hcubature_v(1, integrand_Map2Map3_T2, &container, 4, vals_min2, vals_max2, 0, 0, 1e-1, ERROR_L1, &result_2, &error_2);
        result_2 /= pow(2 * M_PI, 4); // Division by 2pi^4 because 4 integrals in ell-space
    }
    else
    {
        throw std::logic_error("Map2Map3_T2: Wrong survey geometry");
    }

    return result_1 * result_2;
}

int integrand_Map2Map3_T2(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_Map2Map3_T2: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    // Allocate memory on device for integrand values
    double *dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

    // Copy integration variables to device
    double *dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory for result on GPU
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

    if (type == 1)
    {
        integrand_Map2Map3_T2_square<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                          container_->thetas_123.at(1), dev_value);
    }
    else
    {
        exit(-1);
    };

    cudaFree(dev_vars); // Free variables

    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); // Free values

    return 0; // Success :)
}

__global__ void integrand_Map2Map3_T2_square(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double ell1_x = vars[i * ndim];
        double ell1_y = vars[i * ndim + 1];
        double ell2_x = vars[i * ndim + 2];
        double ell2_y = vars[i * ndim + 3];

        double ell1 = sqrt(ell1_x * ell1_x + ell1_y * ell1_y);
        double ell2 = sqrt(ell2_x * ell2_x + ell2_y * ell2_y);
        double ell3 = sqrt((ell1_x + ell2_x) * (ell1_x + ell2_x) + (ell1_y + ell2_y) * (ell1_y + ell2_y));
        double result;

        double Gfactor = G_square(ell1_x, ell1_y);

        result = bkappa(ell1, ell2, ell3);
        result *= uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3 * theta2);
        result *= Gfactor;

        value[i] = result;
    }
}

double Map2Map3_T3_total(const std::vector<double> &thetas_123, const double &theta_4)
{
    if (thetas_123.size() != 3)
    {
        throw std::invalid_argument("Map2Map3_T2_total: Wrong number of aperture radii");
    };

    double th0 = thetas_123.at(0);
    double th1 = thetas_123.at(1);
    double th2 = thetas_123.at(2);
    double th3 = theta_4;

    double result = 0;

    // TO DO: BY CHECKING IF TWO THETAS ARE THE SAME, THE NUMBER OF INDIVIDUAL PERMUATIONS TO
    // BE CALCULATED CAN BE REDUCED!

    printf("Doing %f, %f, %f, %f \n", th0, th1, th2, th3);
    result += Map2Map3_T3(th0, th1, th2, th3);

    printf("Doing %f, %f, %f, %f \n", th1, th0, th2, th3);
    result += Map2Map3_T3(th1, th0, th2, th3);

    printf("Doing %f, %f, %f, %f \n", th2, th0, th1, th3);
    result += Map2Map3_T3(th2, th0, th1, th3);

    return result;
}

double Map2Map3_T3(const double &theta1, const double &theta2, const double &theta3, const double &theta4)
{

    std::vector<double> thetas{theta1, theta2, theta3, theta4};
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = thetas;
    double thetaMin = std::min({theta1, theta2, theta3, theta4});
    double lMax = 10. / thetaMin;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
    lMin = 1e-5; // Set lMin to a value different from zero, so we can integrate over the log(ell)
    double result, error;

    if (type == 2)
    {
        double vals_min[3] = {log(lMin), log(lMin), 0};
        double vals_max[3] = {log(lMax), log(lMax), 2 * M_PI};
        hcubature_v(1, integrand_Map2Map3_T3, &container, 3, vals_min, vals_max, 0, 0, 1e-1, ERROR_L1, &result, &error);
        result /= pow(2 * M_PI, 2); // Division by 2pi^2 because 2 integral in ell-space
        result /= area;
        result *= 2;
    }
    else
    {
        throw std::logic_error("Map2Map3_T3: Only coded for infinite survey");
    }

    return result;
}

int integrand_Map2Map3_T3(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_Map2Map3_T3: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    // Allocate memory on device for integrand values
    double *dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

    // Copy integration variables to device
    double *dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

    // Calculate values
    if (type == 2)
    {
        integrand_Map2Map3_T3_infinite<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                            container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                            container_->thetas_123.at(3), dev_value);
    }
    else // This should not happen
    {
        exit(-1);
    };

    cudaFree(dev_vars); // Free variables

    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); // Free values

    return 0; // Success :)
}

__global__ void integrand_Map2Map3_T3_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l1 = exp(vars[i * ndim]);
        double l2 = exp(vars[i * ndim + 1]);
        double phi = vars[i * ndim + 2];

        double l3 = l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi);

        if (l3 > dev_lMax || l3 <= 0)
        {
            value[i] = 0;
        }
        else
        {
            l3 = sqrt(l3);

            double result = uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l1 * theta4) * uHat(l1 * theta4);

            result *= bkappa(l1, l2, l3);
            result *= Pell(l1);
            result *= l1 * l2; // one time for the log-integration
            result *= l1 * l2;

            value[i] = result;
        }
    }
}

double Map2Map3_T4_total(const std::vector<double> &thetas_123, const double &theta_4)
{
    if (thetas_123.size() != 3)
    {
        throw std::invalid_argument("Map2Map3_T2_total: Wrong number of aperture radii");
    };

    double th0 = thetas_123.at(0);
    double th1 = thetas_123.at(1);
    double th2 = thetas_123.at(2);
    double th3 = theta_4;

    double result = 0;

    printf("Doing %f, %f, %f, %f \n", th0, th1, th2, th3);
    result += Map2Map3_T4(th0, th1, th2, th3);

    return result;
}

double Map2Map3_T4(const double &theta1, const double &theta2, const double &theta3, const double &theta4)
{

    std::vector<double> thetas{theta1, theta2, theta3, theta4};
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = thetas;
    double thetaMin = std::min({theta1, theta2, theta3, theta4});
    double lMax = 10. / thetaMin;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));

    double lMin = 1e-4;
    double result, error;

    if (type == 2)
    {
        double mmin = pow(10, logMmin);
        double mmax = pow(10, logMmax);

        double vals_min[5] = {log(lMin), log(lMin), log(lMin), 0, mmin};
        double vals_max[5] = {log(lMax), log(lMax), log(lMax), 2 * M_PI, mmax};
        hcubature_v(1, integrand_Map2Map3_T4, &container, 5, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
        result /= pow(2 * M_PI, 3); // Division by 2pi^3 because 3 integral in ell-space
        result /= area;
    }
    else
    {
        throw std::logic_error("Map2Map3_T4: Only coded for infinite survey");
    }

    return result;
}

int integrand_Map2Map3_T4(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_Map2Map3_T4: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    int Nmax = 1e5;
    int Niter = int(npts / Nmax) + 1; // Number of iterations

    for (int i = 0; i < Niter; i++)
    {

        int npts_iter;
        if (i == Niter - 1)
        {
            npts_iter = npts - (Niter - 1) * Nmax;
        }
        else
        {
            npts_iter = Nmax;
        };

        double vars_iter[npts_iter * ndim];
        for (int j = 0; j < npts_iter; j++)
        {
            for (int k = 0; k < ndim; k++)
            {
                vars_iter[j * ndim + k] = vars[i * Nmax * ndim + j * ndim + k];
            }
        };

        // Allocate memory on device for integrand values
        double *dev_value_iter;
        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));

        // Copy integration variables to device
        double *dev_vars_iter;
        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying

        // Calculate values
        if (type == 2)
        {
            double mMin = pow(10, logMmin);
            double mMax = pow(10, logMmax);
            integrand_Map2Map3_T4_infinite<<<BLOCKS, THREADS>>>(dev_vars_iter, ndim, npts_iter, container_->thetas_123.at(0),
                                                                container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                                container_->thetas_123.at(3), dev_value_iter);
        }
        else // This should not happen
        {
            exit(-1);
        };

        cudaFree(dev_vars_iter); // Free variables

        double value_iter[npts_iter];
        // Copy results to host
        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));

        cudaFree(dev_value_iter); // Free values

        for (int j = 0; j < npts_iter; j++)
        {
            value[i * Nmax + j] = value_iter[j];
        }
    }

    return 0; // Success :)
}

__global__ void integrand_Map2Map3_T4_infinite(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, double theta4, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l1 = exp(vars[i * ndim]);
        double l2 = exp(vars[i * ndim + 1]);
        double l4 = exp(vars[i * ndim + 2]);
        double phi = vars[i * ndim + 3];
        double m = vars[i * ndim + 4];

        double l3 = (l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));

        if (l3 <= 0)
        {
            value[i] = 0;
        }
        else
        {
            l3 = sqrt(l3);
            double result = uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) * uHat(l4 * theta4) * uHat(l4 * theta4);

            double tetraspec = tetraspectrum_limber_integrated(0, dev_z_max, m, l1, l2, l3, l4, l4);
            result *= tetraspec;
            result *= l1 * l2 * l4;
            result *= l1 * l2 * l4;

            value[i] = result;
        }
    }
}