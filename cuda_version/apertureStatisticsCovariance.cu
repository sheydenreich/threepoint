#include "apertureStatisticsCovariance.cuh"
#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"

#include "cubature.h"


#include <algorithm>
#include <iostream>

__constant__ double dev_thetaMax;
__constant__ double dev_lMin;
double lMin;
double thetaMax;
int type; //0: circle, 1: square, 2: infinite

__constant__ double devLogMmin, devLogMmax;
int __constant__ dev_n_mbins;

__constant__ double dev_sigmaW;
double sigmaW, VW, chiMax;

__constant__ double dev_sigma2_array[n_mbins]; 
__constant__ double dev_dSigma2dm_array[n_mbins]; 

void writeCov(const std::vector<double> &values, const int &N, const std::string &filename)
{
    if (values.size() != N * N)
    {
        throw std::out_of_range("writeCov: Values has wrong length");
    };

    std::ofstream out(filename);
    if (!out.is_open())
    {
        throw std::runtime_error("writeCov: Could not open " + filename);
    };

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            out << values.at(i * N + j) << " ";
        };
        out << std::endl;
    }
}


double Cov_Gaussian(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("Cov_Gaussian: Wrong number of aperture radii");
    };

    double result;
    if (type == 2)
    {
        result = T1_total(thetas_123, thetas_456);
    }
    else
    {
        result = T1_total(thetas_123, thetas_456) + T2_total(thetas_123, thetas_456);
    }
    return result;
}

double Cov_NonGaussian(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("Cov_NonGaussian: Wrong number of aperture radii");
    };

    if (type != 0 && type != 2)
    {
        throw std::logic_error("Cov_NonGaussian: Is only coded for circular and infinite surveys");
    };

    return T4_total(thetas_123, thetas_456);
}

double T1_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T1_total: Wrong number of aperture radii");
    };

    double result;
    if(thetas_456.at(0)==thetas_456.at(1))
    {
        if(thetas_456.at(0)==thetas_456.at(2))
        {
            result = 6*T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else
        {
            result = 2*T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2* T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(2), thetas_456.at(1));
            result += 2*T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(0), thetas_456.at(1));
        }
    }
    else if (thetas_456.at(0)==thetas_456.at(2))
    {
        result = 2*T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        result += 2*T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(2), thetas_456.at(1));
        result += 2*T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
    }
    else if (thetas_456.at(1)==thetas_456.at(2))
    {
        result = 2*T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        result += 2*T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        result += 2*T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(2), thetas_456.at(0));
    }
    else
    {
        result = T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        result += T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(2), thetas_456.at(1));
        result += T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        result += T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(2), thetas_456.at(0));
        result += T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(0), thetas_456.at(1));
        result += T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
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
    
    if(thetas_123.at(0)==thetas_123.at(1) && thetas_123.at(0)==thetas_123.at(2))
    {
        if(thetas_456.at(0)==thetas_456.at(1) && thetas_456.at(0)==thetas_456.at(2))
        {
            result = 9*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else if(thetas_456.at(0)==thetas_456.at(1))
        {
            result = 6*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 3*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
        }
        else if(thetas_456.at(0)==thetas_456.at(2))
        {
            result = 6*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 3*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        }
        else if(thetas_456.at(1)==thetas_456.at(2))
        {
            result = 6*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 3*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else
        {
            result = 3*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 3*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 3*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
        }
    }
    else if(thetas_123.at(0)==thetas_123.at(1))
    {
        if(thetas_456.at(0)==thetas_456.at(1) && thetas_456.at(0)==thetas_456.at(2))
        {
            result = 6*T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(0), thetas_456.at(2), thetas_456.at(1));
            result += 3*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else if(thetas_456.at(0)==thetas_456.at(1))
        {
            result = 2*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
            result += 4*T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2*T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
        }
        else if(thetas_456.at(0)==thetas_456.at(2))
        {
            result = 2*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 4*T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2*T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        }
        else if(thetas_456.at(1)==thetas_456.at(2))
        {
            result = 2*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 4*T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 2*T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else
        {
            result = T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
            result += 2*T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2*T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 2*T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
        }
    }
    else if(thetas_123.at(1)==thetas_123.at(2))
    {
        if(thetas_456.at(0)==thetas_456.at(1) && thetas_456.at(0)==thetas_456.at(2))
        {
            result = 6*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 3*T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else if(thetas_456.at(0)==thetas_456.at(1))
        {
            result = 2*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
            result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 4*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2*T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else if(thetas_456.at(0)==thetas_456.at(2))
        {
            result = 2*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 4*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2*T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else if(thetas_456.at(1)==thetas_456.at(2))
        {
            result = 2*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 4*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 2*T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        }
        else
        {
            result = 2*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 2*T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
            result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
        }
    }
    else
    {
        result = T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
        result += T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        result += T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        result += T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
        result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
    }

    result /= pow(2 * M_PI, 4);

    return result;
}


double T4_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T4_total: Wrong number of aperture radii");
    };

    double result = T4(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
    result += T4(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
    result += T4(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(1));
    result += T4(thetas_123.at(1), thetas_123.at(0), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
    result += T4(thetas_123.at(1), thetas_123.at(0), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
    result += T4(thetas_123.at(1), thetas_123.at(0), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
    result += T4(thetas_123.at(2), thetas_123.at(0), thetas_123.at(1), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
    result += T4(thetas_123.at(2), thetas_123.at(0), thetas_123.at(1), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
    result += T4(thetas_123.at(2), thetas_123.at(0), thetas_123.at(1), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));

    result /= pow(2 * M_PI, 8);
    return result;
}

double T1(const double &theta1, const double &theta2, const double &theta3,
    const double &theta4, const double &theta5, const double &theta6)
{
// Set maximal l value such, that theta*l <= 10
double thetaMin_123 = std::min({theta1, theta2, theta3});
double thetaMin_456 = std::min({theta4, theta5, theta6});
double thetaMin = std::max({thetaMin_123, thetaMin_456});
double lMax = 10. / thetaMin;
// Create container
ApertureStatisticsCovarianceContainer container;
container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
container.thetas_456 = std::vector<double>{theta4, theta5, theta6};

// Do integration
double result, error;
if (type == 2)
{

double vals_min[3] = {lMin, lMin , 0};
double vals_max[3] = {lMax, lMax, M_PI}; // use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

hcubature_v(1, integrand_T1, &container, 3, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

result *= 2 / thetaMax / thetaMax / 8 / M_PI / M_PI / M_PI;
}
else if (type == 0 || type == 1)
{
double vMax=lMax;
double vals_min[6]={-vMax,-vMax, -lMax, -lMax, -lMax, -lMax};
double vals_max[6]={1.02*vMax, 1.02*vMax, 1.02*lMax, 1.02*lMax, 1.02*lMax, 1.02*lMax};
hcubature_v(1, integrand_T1, &container, 6, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);

result /= pow(2 * M_PI, 6);
}
else
{
throw std::logic_error("T1: Wrong survey geometry");
};


return result;
}


double T2(const double &theta1, const double &theta2, const double &theta3,
    const double &theta4, const double &theta5, const double &theta6)
{
if (type == 2)
return 0;

// Integral over ell 1
std::vector<double> thetas1{theta1, theta2};
ApertureStatisticsCovarianceContainer container;
container.thetas_123 = thetas1;
double thetaMin=std::min({theta1, theta2});
double lMax = 10./thetaMin;

double result_A1, error_A1;

double vals_min1[1] = {lMin};
double vals_max1[1] = {lMax};
hcubature_v(1, integrand_T2_part1, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_A1, &error_A1);

// Integral over ell 3
std::vector<double> thetas2{theta5, theta6};
container.thetas_123 = thetas2;
thetaMin=std::min({theta5, theta6});
lMax = 10./thetaMin;

double result_A2, error_A2;

hcubature_v(1, integrand_T2_part1, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_A2, &error_A2);

// Integral over ell 2
std::vector<double> thetas3{theta3, theta4};
container.thetas_123 = thetas3;
thetaMin=std::min({theta3, theta4});
lMax = 10./thetaMin;
double result_B, error_B;
if (type == 0)
{
hcubature_v(1, integrand_T2_part2, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_B, &error_B);
}
else if (type == 1)
{
double vals_min2[2] = {-lMax, -lMax};
double vals_max2[2] = {lMax, lMax};
hcubature_v(1, integrand_T2_part2, &container, 2, vals_min2, vals_max2, 0, 0, 1e-4, ERROR_L1, &result_B, &error_B);
}
else
{
throw std::logic_error("T2: Wrong survey geometry");
};

return result_A1 * result_A2 * result_B;
}

double T4(const double &theta1, const double &theta2, const double &theta3,
    const double &theta4, const double &theta5, const double &theta6)
{
// Set maximal l value such, that theta*l <= 10
double thetaMin_123 = std::min({theta1, theta2, theta3});
double thetaMin_456 = std::min({theta4, theta5, theta6});
double thetaMin = std::max({thetaMin_123, thetaMin_456});
double lMax = 10. / thetaMin;

// Create container
ApertureStatisticsCovarianceContainer container;
container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
container.thetas_456 = std::vector<double>{theta4, theta5, theta6};
// Do integration
double result, error;
if (type == 2)
{
double vals_min[5] = {lMin, lMin, lMin, 0, 0};
double vals_max[5] = {lMax, lMax, lMax, M_PI, M_PI}; // use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

hcubature_v(1, integrand_T4, &container, 5, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
result = 4. * result / thetaMax / thetaMax / pow(2 * M_PI, 5);
}
else if (type == 0)
{
double vals_min[8] = {lMin, lMin, lMin, lMin, 0, 0, 0, 0};
double vals_max[8] = {lMax, lMax, lMax, lMax, 2 * M_PI, 2 * M_PI, 2 * M_PI, 2 * M_PI};
pcubature_v(1, integrand_T4, &container, 8, vals_min, vals_max, 0, 0, 0.2, ERROR_L1, &result, &error);
}
else
{
throw std::logic_error("T4: Wrong survey geometry");
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

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    };

    // Allocate memory on device for integrand values
    double* dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

    // Copy integration variables to device
    double* dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

    // Calculate values
    if(type==0)
    {
        integrand_T1_circle<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                container_->thetas_456.at(0), container_->thetas_456.at(1), 
                                                container_->thetas_456.at(2), dev_value);
    }
    else if(type==1)
    {
        integrand_T1_square<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                container_->thetas_456.at(0), container_->thetas_456.at(1), 
                                                container_->thetas_456.at(2), dev_value);
    }
    else if(type==2)
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

    cudaFree(dev_vars); //Free variables
  
    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); //Free values

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

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;

    };

    // Allocate memory on device for integrand values
    double* dev_value;

    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));


    // Copy integration variables to device
    double* dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

    // Calculate values
    if(type==0 || type==1)
    {
        integrand_T2_part1<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                container_->thetas_123.at(1), dev_value);
    }
    else // This should not happen
    {
        exit(-1);
    };

    cudaFree(dev_vars); //Free variables
  
    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); //Free values

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

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    };

    // Allocate memory on device for integrand values
    double* dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

    // Copy integration variables to device
    double* dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

    // Calculate values
    if(type==0)
    {
        integrand_T2_part2_circle<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                container_->thetas_123.at(1), dev_value);
    }
    else if(type==1)
    {
        integrand_T2_part2_square<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                container_->thetas_123.at(1), dev_value);
    }
    else // This should not happen
    {
        exit(-1);
    };

    cudaFree(dev_vars); //Free variables
  
    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); //Free values

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

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    };

    // Allocate memory on device for integrand values
    double* dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

    // Copy integration variables to device
    double* dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

    // Calculate values
    if(type==0)
    {
        integrand_T4_circle<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                container_->thetas_456.at(0), container_->thetas_456.at(1), 
                                                container_->thetas_456.at(2), dev_value);
    }
    else if(type==4)
    {
        integrand_T4_infinite<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
                                                    container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                    container_->thetas_456.at(0), container_->thetas_456.at(1), 
                                                    container_->thetas_456.at(2), dev_value);
    }
    else // This should not happen
    {
        exit(-1);
    };

    cudaFree(dev_vars); //Free variables
  
    // Copy results to host
    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); //Free values

    return 0; // Success :)
}

__global__ void integrand_T1_circle(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
    double theta4, double theta5, double theta6, double* value)
{
    int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double a=vars[i*ndim];
        double b=vars[i*ndim+1];
        double c=vars[i*ndim+2];
        double d=vars[i*ndim+3];
        double e=vars[i*ndim+4];
        double f=vars[i*ndim+5];


        double ell = sqrt(a * a + b * b);
        double Gfactor = G_circle(ell);
    
        double ell1 = sqrt((a - c - e) * (a - c - e) + (b - d - f) * (b - d - f));
        double ell2 = sqrt(c * c + d * d);
        double ell3 = sqrt(e * e + f * f);
    
        if (ell1 <= dev_lMin || ell2 <= dev_lMin || ell3 <= dev_lMin)
        {
            value[i]=0;
        }
        else
        {    
        double result = dev_Pell(ell1) * dev_Pell(ell2) * dev_Pell(ell3);
        result *= uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3 * theta3);
        result *= uHat(ell1 * theta4) * uHat(ell2 * theta5) * uHat(ell3 * theta6);
        result *= Gfactor;
    
        value[i]=result;
        };
    }
}


__global__ void integrand_T1_square(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
    double theta4, double theta5, double theta6, double* value)
{
    int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double a=vars[i*ndim];
        double b=vars[i*ndim+1];
        double c=vars[i*ndim+2];
        double d=vars[i*ndim+3];
        double e=vars[i*ndim+4];
        double f=vars[i*ndim+5];

        double Gfactor =G_square(a, b);
    
        double ell1 = sqrt((a - c - e) * (a - c - e) + (b - d - f) * (b - d - f));
        double ell2 = sqrt(c * c + d * d);
        double ell3 = sqrt(e * e + f * f);
    
        if (ell1 <= 0 || ell2 <= 0 || ell3 <= 0)
        {
            value[i]=0;
        }
        else
        {
        double result = dev_Pell(ell1) * dev_Pell(ell2) * dev_Pell(ell3);
        result *= uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3 * theta3);
        result *= uHat(ell1 * theta4) * uHat(ell2 * theta5) * uHat(ell3 * theta6);
        result *= Gfactor;
    
        value[i]=result;
        };
    }
}


__global__ void integrand_T1_infinite(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
    double theta4, double theta5, double theta6, double* value)
{
    int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double l1=vars[i*ndim];
        double l2=vars[i*ndim+1];
        double phi=vars[i*ndim+2];

        double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));

        if (l1 <= dev_lMin || l2 <= dev_lMin)
        {
            value[i]=0;
        }
        else
        {
            double result = dev_Pell(l1) *dev_Pell(l2) * dev_Pell(l3);
            result *= uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3);
            result *= uHat(l1 * theta4) * uHat(l2 * theta5) * uHat(l3 * theta6);
            result *= l1 * l2;
            value[i]=result;
        };
    }
}


__global__ void integrand_T2_part1(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double* value)
{
    int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double ell=vars[i*ndim];

        if(ell<dev_lMin)
        {
            value[i]=0;
        }
        else
        {
            double result = ell * dev_Pell(ell);
            result *= uHat(ell * theta1) * uHat(ell * theta2);
            value[i]=result;
        }
    }
}


__global__ void integrand_T2_part2_circle(const double* vars, unsigned ndim, int npts, double theta1, double theta2,
                                            double* value)
{
    int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double ell=vars[i*ndim];

        if(ell<dev_lMin)
        {
            value[i]=0;
        }
        else
        {
            double Gfactor = G_circle(ell);
            double result = dev_Pell(ell);
            result *= ell * uHat(ell * theta1) * uHat(ell * theta2);
            result *= Gfactor;
            value[i]=result;
        };
    }
}


__global__ void integrand_T2_part2_square(const double* vars, unsigned ndim, int npts, double theta1, double theta2, 
                                            double* value)
{
    int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double ellX=vars[i*ndim];
        double ellY=vars[i*ndim+1];

        double Gfactor = G_square(ellX, ellY);
        double ell = sqrt(ellX * ellX + ellY * ellY);

        if(ell<dev_lMin)
        {
            value[i]=0;
        }
        else
        {
            double result = dev_Pell(ell);
            result *= uHat(ell * theta1) * uHat(ell * theta2);
            result *= Gfactor;
            value[i]=result;
        };
    } 
}

__global__ void integrand_T4_circle(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
                                    double theta4, double theta5, double theta6, double* value)
{
    int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double v=vars[i*ndim];
        double ell2=vars[i*ndim+1];
        double ell4=vars[i*ndim+2];
        double ell5=vars[i*ndim+3];
        double alphaV=vars[i*ndim+4];
        double alpha2=vars[i*ndim+5];
        double alpha4=vars[i*ndim+6];
        double alpha5=vars[i*ndim*7];

        double ell1 = sqrt(v * v + ell4 * ell4 - 2 * v * ell4 * cos(alpha4 - alphaV));
        double ell3 = sqrt(ell1 * ell1 + ell2 * ell2 + 2 * ell1 * ell2 * cos(alpha2 - alpha4 + alphaV));
        double ell6 = sqrt(ell4 * ell4 + ell5 * ell5 + 2 * ell4 * ell5 * cos(alpha5 - alpha4));
    
        double Gfactor = G_circle(v);
    
        double result = bkappa(ell4, ell2, ell3);
        result *= bkappa(ell1, ell5, ell6);
        result *= uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3 * theta3);
        result *= uHat(ell4 * theta4) * uHat(ell5 * theta5) * uHat(ell6 * theta6);
        result *= v * ell2 * ell4 * ell5;
        result *= Gfactor;
        
        value[i]=result;
    } 
}

__global__ void integrand_T4_infinite(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
                                        double theta4, double theta5, double theta6, double* value)
{
    int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double l1=vars[i*ndim];
        double l2=vars[i*ndim+1];
        double l5=vars[i*ndim+2];
        double phi1=vars[i*ndim+3];
        double phi2=vars[i*ndim+4];

        double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi1));
        double l6 = sqrt(l1 * l1 + l5 * l5 + 2 * l1 * l5 * cos(phi2));
    
        double result = uHat(l1 * theta4) * uHat(l2 * theta2) * uHat(l3 * theta3) 
                        * uHat(l1 * theta1) * uHat(l5 * theta5) * uHat(l6 * theta6);
    
        result *= bkappa(l1, l2, l3);
        result *= bkappa(l1, l5, l6);
        result *= l1 * l2 * l5;

        value[i]=result;
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


/// THE FOLLOWING THINGS ARE NEEDED FOR THE SUPERSAMPLE COVARIANCE

double Cov_SSC(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    double cx = z_max / 2;
    double q=0;
    for(int i=0; i<48; i++)
    {
       std::cerr<<"Cov_SSC Redshift integral "<<i<<"/47"<<std::endl;
        q+=W96[i]*(integrand_Cov_SSC(cx-cx*A96[i], thetas_123, thetas_456)+integrand_Cov_SSC(cx+cx*A96[i], thetas_123, thetas_456));
    }
    std::cerr<<"Cov_SSC:"<<q*cx<<std::endl;
    return q*cx/thetaMax/thetaMax;
}

double integrand_Cov_SSC(const double& z, const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    double chi= c_over_H0*GQ96_of_Einv(0,z);
    double result = f(z, thetas_123, chi)*f(z, thetas_456, chi);
    result*=sigmaW*c_over_H0*E_inv(z);


    double didx = z / z_max * (n_redshift_bins - 1);
    int idx = didx;
    didx = didx - idx;
    if (idx == n_redshift_bins - 1)
    {
      idx = n_redshift_bins - 2;
      didx = 1.;
    }
    double g= g_array[idx] * (1 - didx) + g_array[idx + 1] * didx;
    result*=pow(g, 6);
    result/=pow(chi, 10);
    std::cerr<<"z:"<<z<<" integrand_Cov_SSC:"<<result<<std::endl;
    return result;
}

double f(const double& z, const std::vector<double> &thetas_123, const double& chi)
{
    ApertureStatisticsSSCContainer container;
    container.thetas_123=thetas_123;
    container.z=z;
    container.chi=chi;

    double thetaMin=std::min({thetas_123.at(0), thetas_123.at(1), thetas_123.at(2)});
    double lMax = 10./thetaMin;
    double result, error;

    double mmin=pow(10, logMmin);
    double mmax=pow(10, logMmax);


    double vals_min[4]={lMin, lMin, 0, mmin};
    double vals_max[4]={lMax, lMax, 2*M_PI, mmax};

    hcubature_v(1, integrand_f, &container, 4, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);

    return result;
}

int integrand_f(unsigned ndim, size_t npts, const double* vars, void* container, unsigned fdim, double* value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_f: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsSSCContainer *container_ = (ApertureStatisticsSSCContainer *) container;

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    };

   // Allocate memory on device for integrand values
   double* dev_value;
   CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

   // Copy integration variables to device
   double* dev_vars;
   CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
   CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

   integrand_f<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, dev_value, container_->thetas_123.at(0), container_->thetas_123.at(1), container_->thetas_123.at(2), container_->z, container_->chi);
    CUDA_SAFE_CALL(cudaFree(dev_vars));

    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(dev_value));
    return 0;
}

__global__ void integrand_f(const double* vars, unsigned ndim, int npts, double* value, double theta1, double theta2, double theta3, double z, double chi)
{
    int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double l1=vars[i*ndim];
        double l2=vars[i*ndim+1];
        double phi=vars[i*ndim+2];
        double m=vars[i*ndim+3];

        //printf("Integration vars: %f %f %f %f\n", l1, l2, phi, m);

        double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));
        if (l3==0)
        {
            value[i]=0;
        }
        else
        {
        double rho_m=dev_rhobar(z);

        double result = uHat(l1*theta1)*uHat(l2*theta2)*uHat(l3*theta3);
        result*=hmf(m, z)*halobias(m, z)*m*m*m/rho_m/rho_m/rho_m;
        result*=u_NFW(l1/chi, m, z)*u_NFW(l2/chi, m, z)*u_NFW(l3/chi, m, z);
        //printf("Integration vars: %f %f %f %f %f %E\n", l1, l2, phi, m, l3, result);
        value[i]=result;
        };
    };
}

__device__ double halobias(const double& m, const double& z)
{
    double q=0.707;
    double p=0.3;
    
    // Get sigma^2(m, z)
    double sigma2=get_sigma2(m, z);
  
    // Get critical density contrast
    double deltac=delta_c(z);
  
    double result=1+1./deltac*(q*deltac*deltac/sigma2 - 1 + 2*p/(1+pow(q*deltac*deltac/sigma2, p)));
    return result;
}

__device__ double dev_rhobar(const double& z)
{
    return dev_om*2.7754e11; //[Msun*h²/Mpc³]
}

double rhobar(const double& z)
{
    double result= cosmo.om*2.7754e11; //[Msun*h²/Mpc³]
    return result;
}


__device__ double hmf(const double& m, const double& z)
{

    double A=0.322;
    double q=0.707;
    double p=0.3;


    
    // Get sigma^2(m, z)
    double sigma2=get_sigma2(m, z);
    //printf("%f %f\n", m, sigma2);
  
    // Get dsigma^2/dm
    double dsigma2=get_dSigma2dm(m, z);
  
    // Get critical density contrast
    double deltac=delta_c(z);
  
    // Get mean density
    double rho_mean=dev_rhobar(z);
  
    double nu=deltac*deltac/sigma2;

    double result=-rho_mean/m/sigma2*dsigma2*A*(1+pow(q*nu, -p))*sqrt(q*nu/2/M_PI)*exp(-0.5*q*nu);


    /*
     * \f $n(m,z)=-\frac{\bar{\rho}}{m \sigma} \dv{\sigma}{m} A \sqrt{2q/pi} (1+(\frac{\sigma^2}{q\delta_c^2})^p) \frac{\delta_c}{\sigma} \exp(-\frac{q\delta_c^2}{2\sigma^2})$ \f
     */
    return result;
}

__host__ __device__ void SiCi(double x, double& si, double& ci)
  {
  double x2=x*x;
  double x4=x2*x2;
  double x6=x2*x4;
  double x8=x4*x4;
  double x10=x8*x2;
  double x12=x6*x6;
  double x14=x12*x2;

  if(x<4)
    { 
      
      double a=1-4.54393409816329991e-2*x2+1.15457225751016682e-3*x4
	-1.41018536821330254e-5*x6+9.43280809438713025e-8*x8
	-3.53201978997168357e-10*x10+7.08240282274875911e-13*x12
	-6.05338212010422477e-16*x14;

      double b=1+1.01162145739225565e-2*x2+4.99175116169755106e-5*x4
	+1.55654986308745614e-7*x6+3.28067571055789734e-10*x8
	+4.5049097575386581e-13*x10+3.21107051193712168e-16*x12;

      si=x*a/b;

      double gamma=0.5772156649;
      a=-0.25+7.51851524438898291e-3*x2-1.27528342240267686e-4*x4
	+1.05297363846239184e-6*x6-4.68889508144848019e-9*x8
	+1.06480802891189243e-11*x10-9.93728488857585407e-15*x12;
      
      b=1+1.1592605689110735e-2*x2+6.72126800814254432e-5*x4
	+2.55533277086129636e-7*x6+6.97071295760958946e-10*x8
	+1.38536352772778619e-12*x10+1.89106054713059759e-15*x12
	+1.39759616731376855e-18*x14;

      ci=gamma+std::log(x)+x2*a/b;
    }
  else
    {
      double x16=x8*x8;
      double x18=x16*x2;
      double x20=x10*x10;
      double cos_x=cos(x);
      double sin_x=sin(x);

      double f=(1+7.44437068161936700618e2/x2+1.96396372895146869801e5/x4
		+2.37750310125431834034e7/x6+1.43073403821274636888e9/x8
		+4.33736238870432522765e10/x10+6.40533830574022022911e11/x12
		+4.20968180571076940208e12/x14+1.00795182980368574617e13/x16
		+4.94816688199951963482e12/x18-4.94701168645415959931e11/x20)
	/(1+7.46437068161927678031e2/x2+1.97865247031583951450e5/x4
	  +2.41535670165126845144e7/x6+1.47478952192985464958e9/x8
	  +4.58595115847765779830e10/x10+7.08501308149515401563e11/x12
	  +5.06084464593475076774e12/x14+1.43468549171581016479e13/x16
	  +1.11535493509914254097e13/x18)/x;
      
      double g=(1+8.1359520115168615e2/x2+2.35239181626478200e5/x4
		+3.12557570795778731e7/x6+2.06297595146763354e9/x8
		+6.83052205423625007e10/x10+1.09049528450362786e12/x12
		+7.57664583257834349e12/x14+1.81004487464664575e13/x16
		+6.43291613143049485e12/x18-1.36517137670871689e12/x20)/
	(1+8.19595201151451564e2/x2+2.40036752835578777e5/x4
	 +3.26026661647090822e7/x6+2.23355543278099360e9/x8
	 +7.87465017341829930e10/x10+1.39866710696414565e12/x12
	 +1.17164723371736605e13/x14+4.01839087307656620e13/x16
	 +3.99653257887490811e13/x18)/x2;

      si=0.5*M_PI-f*cos_x-g*sin_x;
      ci=f*sin_x-g*cos_x;
    };
  return;
}


__device__  double r_200(const double& m, const double& z)
{
return pow(0.75/M_PI*m/dev_rhobar(z)/200, 1./3.);
}

double r_200_host(const double& m, const double& z)
{
double result=pow(0.75/M_PI*m/rhobar(z)/200, 1./3.);
std::cerr<<result<<std::endl;
return result;
}


__device__ double u_NFW(const double& k, const double& m, const double& z)
{
// Get concentration
double c=concentration(m, z);

double arg1=k*r_200(m,z)/c;
double arg2=arg1*(1+c);

double si1, ci1, si2, ci2;
SiCi(arg1, si1, ci1);
SiCi(arg2, si2, ci2);

double term1=sin(arg1)*(si2-si1);
double term2=cos(arg1)*(ci2-ci1);
double term3=-sin(arg1*c)/arg2;
double F=std::log(1.+c)-c/(1.+c);


double result=(term1+term2+term3)/F;

return result;
}

__host__ double u_NFW_host(const double& k, const double& m, const double& z)
{
// Get concentration
double c=concentration(m, z);

double arg1=k*r_200_host(m,z)/c;
double arg2=arg1*(1+c);

double si1, ci1, si2, ci2;
SiCi(arg1, si1, ci1);
SiCi(arg2, si2, ci2);

double term1=sin(arg1)*(si2-si1);
double term2=cos(arg1)*(ci2-ci1);
double term3=-sin(arg1*c)/arg2;
double F=std::log(1.+c)-c/(1.+c);


double result=(term1+term2+term3)/F;
return result;
}

__host__ __device__ double concentration(const double& m, const double& z)
{
    //Using Duffy+ 08 (second mass definition, all halos)
    //To Do: Bullock01 might be better!
    double A= 10.14;
    double Mpiv=2e12; //Msun h⁻1
    double B= -0.081;
    double C= -1.01;
    double result=A*pow(m/Mpiv, B)*pow(1+z, C);
   // printf("conc:%f\n", result);
    return result;
}

__device__ double delta_c(const double& z)
{
    return 1.686*(1+0.0123*log10(dev_om_m_of_z(z)));
}

double sigma2(const double& m)
{
  double R=pow(0.75/M_PI*m/rhobar(0.0), 1./3.); //[Mpc/h]

  SigmaContainer container;
  container.R=R;
  
  double k_min[1]={0};
  double k_max[1]={1e12};
  double result, error;
  
  hcubature_v(1, integrand_sigma2, &container, 1, k_min, k_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
  
  return result/2/M_PI/M_PI;
}

int integrand_sigma2(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim!=1)
    {
      std::cerr<<"integrand_sigma2: Wrong fdim"<<std::endl;
      exit(1);
    };
  SigmaContainer* container = (SigmaContainer*) thisPtr;

 
  double R=container->R;

  
  #pragma omp parallel for
  for(unsigned int i=0; i<npts; i++)
    {
      double k_=k[i*ndim];
      double W=window(k_*R, 0);
      
      value[i]=k_*k_*W*W*linear_pk(k_);
    };
  
  return 0;
}


void setSigma2()
{
    double sigma2_[n_mbins];

    double deltaM=(logMmax-logMmin)/n_mbins;

    for(int i=0; i<n_mbins; i++)
    {


            double m=pow(10, logMmin+i*deltaM);
            double R=pow(0.75/M_PI*m/rhobar(0.0), 1./3.);
            double sigma = sigma2(m);
            sigma2_[i]=sigma;
    }
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma2_array, &sigma2_, n_mbins*sizeof(double)));
    std::cerr<<"Finished precalculating sigma2(m)"<<std::endl;
}




double dSigma2dm(const double& m)
{
  double rho_m=rhobar(0.0);

  double R=pow(0.75/M_PI*m/rhobar(0.0), 1./3.); //[Mpc/h]
  double dR=pow(0.75/M_PI/rhobar(0.0), 1./3.)*pow(m, -2./3.)/3.; //[Mpc/h]


  SigmaContainer container;
  container.R=R;
  container.dR=dR;
  
  
  double k_min[1]={0};
  double k_max[1]={1e12};
  double result, error;
  
  hcubature_v(1, integrand_dSigma2dm, &container, 1, k_min, k_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

  return result/M_PI/M_PI;
}


int integrand_dSigma2dm(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim!=1)
    {
      std::cerr<<"integrand_dSigma2dm: Wrong fdim"<<std::endl;
      exit(1);
    };
  SigmaContainer* container = (SigmaContainer*) thisPtr;

 
  double R=container->R;
  double dR=container->dR;
 
  
  #pragma omp parallel for
  for(unsigned int i=0; i<npts; i++)
    {
      double k_=k[i*ndim];
      double W=window(k_*R,0);
      double Wprime=window(k_*R, 4);
      
      value[i]=k_*k_*W*Wprime*dR*linear_pk(k_)*k_;
    };
  
  return 0;
}

void setdSigma2dm()
{
    double dSigma2dm_[n_mbins];

    double deltaM=(logMmax-logMmin)/n_mbins;

    for(int i=0; i<n_mbins; i++)
    {

            double m=pow(10, logMmin+i*deltaM);
        
            dSigma2dm_[i] = dSigma2dm(m);
            //std::cerr<<i<<" "<<dSigma2dm_[i]<<std::endl;

    }
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_dSigma2dm_array, &dSigma2dm_, n_mbins*sizeof(double)));
    std::cerr<<"Finished precalculating dsigma2/dm"<<std::endl;
}

__device__ double get_sigma2(const double& m, const double& z)
{
    double logM=log10(m);
    if(logM>devLogMmax)
    {
        printf("get_sigma2: ran into Mmax");
        return 0;
    };

    double didx = (logM-devLogMmin) / (devLogMmax-devLogMmin) * (dev_n_mbins - 1);
    int idx = didx;
    didx = didx - idx;
    if (idx == dev_n_mbins - 1)
    {
      idx = dev_n_mbins - 2;
      didx = 1.;
    }
    double sigma=dev_sigma2_array[idx]*(1-didx)+dev_sigma2_array[idx+1]*didx;


    didx = z / dev_z_max * (dev_n_redshift_bins - 1);
    idx = didx;
    didx = didx - idx;
    if (idx == dev_n_redshift_bins - 1)
    {
      idx = dev_n_redshift_bins - 2;
      didx = 1.;
    }

    double D1=dev_D1_array[idx] * (1 - didx) + dev_D1_array[idx + 1] * didx;

    sigma*=D1*D1;

    return sigma;
}


__device__ double get_dSigma2dm(const double& m, const double& z)
{
    double logM=log10(m);
    if(logM>devLogMmax)
    {
        printf("get_dSigma2dm: ran into Mmax");
        return 0;
    };

    double didx = (logM-devLogMmin) / (devLogMmax-devLogMmin) * (dev_n_mbins - 1);
    int idx = didx;
    didx = didx - idx;
    if (idx == dev_n_mbins - 1)
    {
      idx = dev_n_mbins - 2;
      didx = 1.;
    }
    double dsigma=dev_dSigma2dm_array[idx]*(1-didx)+dev_dSigma2dm_array[idx+1]*didx;

    
    didx = z / dev_z_max * (dev_n_redshift_bins - 1);
    idx = didx;
    didx = didx - idx;
    if (idx == dev_n_redshift_bins - 1)
    {
      idx = dev_n_redshift_bins - 2;
      didx = 1.;
    }

    double D1=dev_D1_array[idx] * (1 - didx) + dev_D1_array[idx + 1] * didx;

    dsigma*=D1*D1;
    return dsigma;
}


void setSurveyVolume()
{
    chiMax=GQ96_of_Einv(0, z_max);
    chiMax*=c_over_H0;
    VW = thetaMax*thetaMax*chiMax*chiMax*chiMax/3;
}

void setSigmaW()
{
    sigmaW=1;
    // SigmaContainer container;

    
    // double k_min[3]={-1e10, -1e10, -1e10};
    // double k_max[3]={1e10, 1e10, 1e10};
    // double result, error;
    
    // hcubature_v(1, integrand_SigmaW, &container, 3, k_min, k_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
  
    // result/=(8*M_PI*M_PI*M_PI*VW*VW);

    // sigmaW=result;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigmaW, &sigmaW, sizeof(double)));
}

int integrand_SigmaW(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value)
{
    if(fdim!=1)
    {
      std::cerr<<"integrand_SigmaW: Wrong fdim"<<std::endl;
      exit(1);
    };
  SigmaContainer* container = (SigmaContainer*) thisPtr;

  if(npts>1e8)
  {
      std::cerr<<"WARNING: large number of points:"<<npts<<std::endl;
  };
 
  
 #pragma omp parallel for
  for(unsigned int i=0; i<npts; i++)
    {
      double k1=k[i*ndim];
      double k2=k[i*ndim+1];
      double k3=k[i*ndim+2];
    
      double k=sqrt(k1*k1+k2*k2+k3*k3);
        if(k==0)
        {
            value[i]=0;
        }
        else
        {
        double W=WindowSurvey(k1, k2, k3);
        if(isnan(W))
        {
        printf("k1: %f k2: %f k3: %f Window: %E\n", k1, k2, k3, W);
        };
         value[i]=W*linear_pk(k);
        };
    };
  
  return 0;
}

double WindowSurvey(const double& k1, const double& k2, const double& k3)
{
    double a=k3;
    double b=k1*thetaMax/2;
    double c=k2*thetaMax/2;
    double d=chiMax;
    double result;
    if(a==0)
    {
        if(b*b==c*c) result=0;
        else
        {
            result=c*sin(b*d)*cos(c*d)-b*sin(c*d)*cos(b*d);
            result*=result;
            result/=(b*b+c*c)*(b*b+c*c);
        };
    }
    else
    {
        result=(pow(a*((a - b - c)*(a + b + c)*cos((b - c)*d) - 
        (a + b - c)*(a - b + c)*cos((b + c)*d))*sin(a*d) - 
     cos(a*d)*((a - b - c)*(b - c)*(a + b + c)*
         sin((b - c)*d) - 
        (a + b - c)*(a - b + c)*(b + c)*sin((b + c)*d)),2) + 
   pow(a*(a + b - c)*(a - b + c) - 
     a*(a - b - c)*(a + b + c) + 
     a*cos(a*d)*((a - b - c)*(a + b + c)*cos((b - c)*d) - 
        (a + b - c)*(a - b + c)*cos((b + c)*d)) + 
     sin(a*d)*((a - b - c)*(b - c)*(a + b + c)*
         sin((b - c)*d) - 
        (a + b - c)*(a - b + c)*(b + c)*sin((b + c)*d)),2))/
 (4.*pow(a + b - c,2)*pow(a - b + c,2)*pow(-a + b + c,2)*
   pow(a + b + c,2));
    };

    return result;
}


void initSSC()
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(devLogMmin, &logMmin, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(devLogMmax, &logMmax, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_mbins, &n_mbins, sizeof(int)));
    initCovariance();
    std::cerr<<"Finished copying bispectrum basics"<<std::endl;
    setSurveyVolume();
    std::cerr<<"Finished setting survey volumne"<<std::endl;
    std::cerr<<"VW:"<<VW<<std::endl;
    setSigmaW();
    std::cerr<<"Finished calculating sigma_W"<<std::endl;
    std::cerr<<"sigma_W:"<<sigmaW<<std::endl;
    setSigma2();
    std::cerr<<"Finished calculating sigma²(m)"<<std::endl;
    setdSigma2dm();
    std::cerr<<"Finished calculating dSigma²dm"<<std::endl;
    std::cerr<<"Finished all initializations"<<std::endl;
}


double Cov_Bispec_SSC(const double& ell)
{
    double cx = z_max / 2;
    double q=0;
    for(int i=0; i<48; i++)
    {
        q+=W96[i]*(integrand_Cov_Bispec_SSC(cx-cx*A96[i], ell)+integrand_Cov_Bispec_SSC(cx+cx*A96[i], ell));
    }
    return q*cx/thetaMax/thetaMax;
}

double integrand_Cov_Bispec_SSC(const double& z, const double& ell)
{
    double chi= c_over_H0*GQ96_of_Einv(0,z);
    double result = I3(ell/chi, ell/chi, ell/chi, z, chi)*I3(ell/chi, ell/chi, ell/chi, z, chi);
    result*=sigmaW*c_over_H0*E_inv(z);


    double didx = z / z_max * (n_redshift_bins - 1);
    int idx = didx;
    didx = didx - idx;
    if (idx == n_redshift_bins - 1)
    {
      idx = n_redshift_bins - 2;
      didx = 1.;
    }
    double g= g_array[idx] * (1 - didx) + g_array[idx + 1] * didx;
    result*=pow(g, 6);
    result/=pow(chi, 10);
    return result;
}

double I3(const double& k1, const double& k2, const double& k3, const double& z, const double& chi)
{
    ApertureStatisticsSSCContainer container;
    container.z=z;
    container.chi=chi;
    container.ell1=k1;
    container.ell2=k2;
    container.ell3=k3;

    double mmin=pow(10, logMmin);
    double mmax=pow(10, logMmax);
    double vals_min[1]={mmin};
    double vals_max[1]={mmax};
    double result, error;
    hcubature_v(1, integrand_I3, &container, 1, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

    return result;
}

int integrand_I3(unsigned ndim, size_t npts, const double* vars, void* container, unsigned fdim, double* value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_f: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsSSCContainer *container_ = (ApertureStatisticsSSCContainer *) container;

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    };


   // Allocate memory on device for integrand values
   double* dev_value;
   CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

   // Copy integration variables to device
   double* dev_vars;
   CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
   CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

   integrand_I3<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, dev_value, container_->ell1, container_->ell2, container_->ell3, container_->z, container_->chi);
    CUDA_SAFE_CALL(cudaFree(dev_vars));

    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(dev_value));
    return 0;
}

__global__ void integrand_I3(const double* vars, unsigned ndim, int npts, double* value, double k1, double k2, double k3, double z, double chi)
 {
    int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double m=vars[i*ndim];

        //printf("Integration vars: %f %f %f %f\n", l1, l2, phi, m);


        double rho_m=dev_rhobar(z);

        double result=hmf(m, z)*halobias(m, z)*m*m*m/rho_m/rho_m/rho_m;
        result*=u_NFW(k1, m, z)*u_NFW(k2, m, z)*u_NFW(k3, m, z);
       //printf("%E %E \n", m, hmf(m, z));
        //printf("Integration vars: %f %f %f %f %f %E\n", l1, l2, phi, m, l3, result);
        value[i]=result;
        
    };  
 }