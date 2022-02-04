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
hcubature_v(1, integrand_T2_part1, &container, 1, vals_min1, vals_max1, 0, 0, 1e-3, ERROR_L1, &result_A1, &error_A1);

// Integral over ell 3
std::vector<double> thetas2{theta5, theta6};
container.thetas_123 = thetas2;
thetaMin=std::min({theta5, theta6});
lMax = 10./thetaMin;

double result_A2, error_A2;

hcubature_v(1, integrand_T2_part1, &container, 1, vals_min1, vals_max1, 0, 0, 1e-3, ERROR_L1, &result_A2, &error_A2);

// Integral over ell 2
std::vector<double> thetas3{theta3, theta4};
container.thetas_123 = thetas3;
thetaMin=std::min({theta3, theta4});
lMax = 10./thetaMin;
double result_B, error_B;
if (type == 0)
{
hcubature_v(1, integrand_T2_part2, &container, 1, vals_min1, vals_max1, 0, 0, 1e-3, ERROR_L1, &result_B, &error_B);
}
else if (type == 1)
{
double vals_min2[2] = {-lMax, -lMax};
double vals_max2[2] = {lMax, lMax};
hcubature_v(1, integrand_T2_part2, &container, 2, vals_min2, vals_max2, 0, 0, 1e-3, ERROR_L1, &result_B, &error_B);
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
        double result = Pell(ell1) * Pell(ell2) * Pell(ell3);
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
        double result = Pell(ell1) * Pell(ell2) * Pell(ell3);
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

        if (l1 <= dev_lMin || l2 <= dev_lMin || l3 <= dev_lMin)
        {
            value[i]=0;
        }
        else
        {
            double result = Pell(l1) * Pell(l2) * Pell(l3);
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
            double result = ell * Pell(ell);
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
            double result = Pell(ell);
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
            double result = Pell(ell);
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