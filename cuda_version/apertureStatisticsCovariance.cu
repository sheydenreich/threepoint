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

__constant__ double dev_thetaMax;
__constant__ double dev_lMin;
__constant__ double dev_lMax;
double lMin;
double thetaMax;
int type; // 0: circle, 1: square, 2: infinite

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
        throw std::invalid_argument("T4_total: Wrong number of aperture radii");
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
    if (type == 2)
    {

        double vals_min[3] = {lMin, lMin, 0};
        double vals_max[3] = {lMax, lMax, M_PI}; // use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

        hcubature_v(1, integrand_T1, &container, 3, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

        result *= 2 / thetaMax / thetaMax / pow(2 * M_PI, 3); // Factors: 2 because phi integral goes from 0 to Pi, 1/thetaMax² because division by area, (2pi)^-3 because 3 integrals in ell-space
    }
    else if (type == 0 || type == 1)
    {
        double vMax = lMax;
        double vals_min[6] = {-vMax, -vMax, -lMax, -lMax, -lMax, -lMax};
        double vals_max[6] = {1.02 * vMax, 1.02 * vMax, 1.02 * lMax, 1.02 * lMax, 1.02 * lMax, 1.02 * lMax};
        hcubature_v(1, integrand_T1, &container, 6, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);

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
    else if (type == 1)
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
        result = result / thetaMax / thetaMax / pow(2 * M_PI, 5);
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
        result = result / thetaMax / thetaMax / pow(2 * M_PI, 5);
    }
    else
    {
        throw std::logic_error("T5: Wrong survey geometry, only coded for infinite survey");
    };

    return result;
}

double T6(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
{
    if (type != 1)
    {
        throw std::logic_error("T6: Wrong survey geometry, only coded for square survey");
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
        double vals_min[6] = {log(lMin), log(lMin), log(lMin), log(lMin), 0, 0};
        double vals_max[6] = {log(lMax), log(lMax), log(lMax), log(lMax), 2 * M_PI, 2 * M_PI};

        hcubature_v(1, integrand_T7, &container, 6, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
        result = result / thetaMax / thetaMax / pow(2 * M_PI, 6);
    }
    else
    {
        throw std::logic_error("T7: Wrong survey geometry, only coded for infinite survey");
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
            double result = dev_Pell(ell1) * dev_Pell(ell2) * dev_Pell(ell3);
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
            double result = dev_Pell(ell1) * dev_Pell(ell2) * dev_Pell(ell3);
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
            double result = dev_Pell(l1) * dev_Pell(l2) * dev_Pell(l3);
            result *= uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3);
            result *= uHat(l1 * theta4) * uHat(l2 * theta5) * uHat(l3 * theta6);
            result *= l1 * l2;
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
            double result = ell * dev_Pell(ell);
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
            double result = dev_Pell(ell);
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
            double result = dev_Pell(ell);
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

            result *= dev_Pell(l1);
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

            double pentaspec = pentaspectrum_limber_mass_integrated(0, dev_z_max, log(mMin), log(mMax), l1, l2, l3, l4, l5, l6);
            result *= pentaspec;
            result *= l1 * l2 * l4 * l5;
            result *= l1 * l2 * l4 * l5;

            value[i] = result;
        }
    }
}

__device__ double G_circle(const double &ell)
{
    double tmp = dev_thetaMax * ell;
    double result = j1(tmp);
    result *= result;
    result *= 4 / tmp / tmp;
    return result;
}

__device__ double G_square(const double &ellX, const double &ellY)
{
    double tmp1 = 0.5 * ellX * dev_thetaMax;
    double tmp2 = 0.5 * ellY * dev_thetaMax;

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

void initCovariance()
{
    copyConstants();
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_thetaMax, &thetaMax, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma, &sigma, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n, &n, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMin, &lMin, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_constant_powerspectrum, &constant_powerspectrum, sizeof(bool)));
};

__global__ void integrand_T4_testBispec_analytical(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3,
                                                   double theta4, double theta5, double theta6, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l1 = vars[i * ndim];
        double l2 = vars[i * ndim + 1];
        double l5 = vars[i * ndim + 2];

        double l1sq = l1 * l1;
        double l2sq = l2 * l2;
        double l5sq = l5 * l5;

        double th1sq = theta1 * theta1;
        double th2sq = theta2 * theta2;
        double th3sq = theta3 * theta3;
        double th4sq = theta4 * theta4;
        double th5sq = theta5 * theta5;
        double th6sq = theta6 * theta6;

        double alpha = 1e6;

        double result = l1sq * l1sq * l1sq * l1sq * l1 * l2sq * l2sq * l2 * l5sq * l5sq * l5;
        double tmp = -0.5 * l1sq * (th1sq + th3sq + th4sq + th6sq + 8 / alpha) - 0.5 * l2sq * (th2sq + th3sq + 4 / alpha) - 0.5 * l5sq * (th5sq + th6sq + 4 / alpha);

        double tmp2 = cyl_bessel_i0(l1 * l2 * (th3sq + 2 / alpha));
        result *= exp(tmp);
        if (result != 0)
        {
            result *= (l1sq + l2sq) * cyl_bessel_i0(l1 * l2 * (th3sq + 2 / alpha)) - 2 * l1 * l2 * cyl_bessel_i1(l1 * l2 * (th3sq + 2 / alpha));
            result *= (l1sq + l5sq) * cyl_bessel_i0(l1 * l5 * (th6sq + 2 / alpha)) - 2 * l1 * l5 * cyl_bessel_i1(l1 * l5 * (th6sq + 2 / alpha));
        };
        if (!isfinite(result))
        {
            printf("%e, %e, %e, %e, %e, %e, %e\n", l1, l2, l5, result, tmp, exp(tmp), tmp2);
        };

        value[i] = result;
    }
}

int integrand_T4_testBispec_analytical(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T4_testBispec_analytical: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
        return 1;
    };

    // Allocate memory on device for integrand values
    double *dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

    // Copy integration variables to device
    double *dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

    // Calculate values

    integrand_T4_testBispec_analytical<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                            container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                            container_->thetas_456.at(0), container_->thetas_456.at(1),
                                                            container_->thetas_456.at(2), dev_value);

    cudaFree(dev_vars); // Free variables

    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); // Free values

    return 0; // Success :)
}

double T4_testBispec_analytical(const double &theta1, const double &theta2, const double &theta3,
                                const double &theta4, const double &theta5, const double &theta6)
{
    // Set maximal l value such, that theta*l <= 10
    double thetaMin_123 = std::min({theta1, theta2, theta3});
    double thetaMin_456 = std::min({theta4, theta5, theta6});
    double thetaMin = std::min({thetaMin_123, thetaMin_456});
    double lMax = 10. / thetaMin;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));

    // Create container
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
    container.thetas_456 = std::vector<double>{theta4, theta5, theta6};

    // Do integration
    double result, error;

    double vals_min[3] = {0, 0, 0};
    double vals_max[3] = {lMax, lMax, lMax};

    hcubature_v(1, integrand_T4_testBispec_analytical, &container, 3, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
    result /= pow(2 * M_PI, 3);
    result *= pow(theta1 * theta2 * theta3 * theta4 * theta5 * theta6, 2) / 64;

    return result;
}

__global__ void integrand_T4_infinite_testBispec(const double *vars, unsigned ndim, int npts, double theta1, double theta2, double theta3,
                                                 double theta4, double theta5, double theta6, double *value)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
    {
        double l1 = vars[i * ndim];
        double l2 = vars[i * ndim + 1];
        double l5 = vars[i * ndim + 2];
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

            result *= testBispec(l1, l2, l3);
            result *= testBispec(l1, l5, l6);
            result *= l1 * l2 * l5;

            value[i] = result;
        }
    }
}

int integrand_T4_testBispec(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T1: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
        return 1;
    };

    // Allocate memory on device for integrand values
    double *dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

    // Copy integration variables to device
    double *dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

    // Calculate values

    integrand_T4_infinite_testBispec<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                          container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                          container_->thetas_456.at(0), container_->thetas_456.at(1),
                                                          container_->thetas_456.at(2), dev_value);

    cudaFree(dev_vars); // Free variables

    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); // Free values

    return 0; // Success :)
}

double T4_testBispec(const double &theta1, const double &theta2, const double &theta3,
                     const double &theta4, const double &theta5, const double &theta6)
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

    double vals_min[5] = {lMin, lMin, lMin, 0, 0};
    double vals_max[5] = {lMax, lMax, lMax, 2 * M_PI, 2 * M_PI};

    hcubature_v(1, integrand_T4_testBispec, &container, 5, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
    result = result / thetaMax / thetaMax * pow(2 * M_PI, 3);

    return result;
}

__device__ double testBispec(double &l1, double &l2, double &l3)
{
    double alpha = 1e6;
    double result = l1 * l1 * l2 * l2;
    result *= exp((-l1 * l1 - l2 * l2 - l3 * l3) / alpha);
    return result;
}
