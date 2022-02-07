#include "apertureStatisticsCovariance.hpp"

#include "cubature.h"
#include "gsl/gsl_sf.h"

#include <stdexcept>
#include <fstream>
#include <algorithm>

ApertureStatisticsCovariance::ApertureStatisticsCovariance(const std::string &type_, const double &thetaMax_, ApertureStatistics *apertureStatistics_) : type(type_), thetaMax(thetaMax_), apertureStatistics(apertureStatistics_)
{
    if (type != "circle" && type != "square" && type != "infinite")
    {
        throw std::invalid_argument("ApertureStatisticsCovariance: Unrecognized survey geometry type");
    };
    lMin = 2 * M_PI / thetaMax;
}

void ApertureStatisticsCovariance::writeCov(const std::vector<double> &values, const int &N, const std::string &filename)
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

double ApertureStatisticsCovariance::Cov_Gaussian(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("Cov_Gaussian: Wrong number of aperture radii");
    };

    double result;
    if (type == "infinite")
    {
        result = T1_total(thetas_123, thetas_456);
    }
    else
    {
        result = T1_total(thetas_123, thetas_456) + T2_total(thetas_123, thetas_456);
    }
    return result;
}

double ApertureStatisticsCovariance::Cov_NonGaussian(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("Cov_NonGaussian: Wrong number of aperture radii");
    };

    if (type != "circle" && type != "infinite")
    {
        throw std::logic_error("Cov_NonGaussian: Is only coded for circular and infinite surveys");
    };

    return T4_total(thetas_123, thetas_456);
}

double ApertureStatisticsCovariance::T1_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T1_total: Wrong number of aperture radii");
    };

    double result;
    if (thetas_456.at(0) == thetas_456.at(1))
    {
        if (thetas_456.at(0) == thetas_456.at(2))
        {
            result = 6 * T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else
        {
            result = 2 * T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2 * T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(2), thetas_456.at(1));
            result += 2 * T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(0), thetas_456.at(1));
        }
    }
    else if (thetas_456.at(0) == thetas_456.at(2))
    {
        result = 2 * T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        result += 2 * T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(2), thetas_456.at(1));
        result += 2 * T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
    }
    else if (thetas_456.at(1) == thetas_456.at(2))
    {
        result = 2 * T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        result += 2 * T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        result += 2 * T1(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(2), thetas_456.at(0));
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

double ApertureStatisticsCovariance::T2_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T2_total: Wrong number of aperture radii");
    };

    double result;

    if (thetas_123.at(0) == thetas_123.at(1) && thetas_123.at(0) == thetas_123.at(2))
    {
        if (thetas_456.at(0) == thetas_456.at(1) && thetas_456.at(0) == thetas_456.at(2))
        {
            result = 9 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else if (thetas_456.at(0) == thetas_456.at(1))
        {
            result = 6 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 3 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
        }
        else if (thetas_456.at(0) == thetas_456.at(2))
        {
            result = 6 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 3 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        }
        else if (thetas_456.at(1) == thetas_456.at(2))
        {
            result = 6 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 3 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else
        {
            result = 3 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 3 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 3 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
        }
    }
    else if (thetas_123.at(0) == thetas_123.at(1))
    {
        if (thetas_456.at(0) == thetas_456.at(1) && thetas_456.at(0) == thetas_456.at(2))
        {
            result = 6 * T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(0), thetas_456.at(2), thetas_456.at(1));
            result += 3 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else if (thetas_456.at(0) == thetas_456.at(1))
        {
            result = 2 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
            result += 4 * T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2 * T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
        }
        else if (thetas_456.at(0) == thetas_456.at(2))
        {
            result = 2 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 4 * T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2 * T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        }
        else if (thetas_456.at(1) == thetas_456.at(2))
        {
            result = 2 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 4 * T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 2 * T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else
        {
            result = T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
            result += 2 * T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2 * T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 2 * T2(thetas_123.at(0), thetas_123.at(2), thetas_123.at(1), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
        }
    }
    else if (thetas_123.at(1) == thetas_123.at(2))
    {
        if (thetas_456.at(0) == thetas_456.at(1) && thetas_456.at(0) == thetas_456.at(2))
        {
            result = 6 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 3 * T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else if (thetas_456.at(0) == thetas_456.at(1))
        {
            result = 2 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
            result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 4 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2 * T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else if (thetas_456.at(0) == thetas_456.at(2))
        {
            result = 2 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 4 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2 * T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
        }
        else if (thetas_456.at(1) == thetas_456.at(2))
        {
            result = 2 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 4 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 2 * T2(thetas_123.at(1), thetas_123.at(2), thetas_123.at(0), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
        }
        else
        {
            result = 2 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(0), thetas_456.at(1), thetas_456.at(2));
            result += 2 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(1), thetas_456.at(0), thetas_456.at(2));
            result += 2 * T2(thetas_123.at(0), thetas_123.at(1), thetas_123.at(2), thetas_456.at(2), thetas_456.at(1), thetas_456.at(0));
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

double ApertureStatisticsCovariance::T4_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
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

double ApertureStatisticsCovariance::T1(const double &theta1, const double &theta2, const double &theta3,
                                        const double &theta4, const double &theta5, const double &theta6)
{
    // Set maximal l value such, that theta*l <= 10
    double thetaMin_123 = std::min({theta1, theta2, theta3});
    double thetaMin_456 = std::min({theta4, theta5, theta6});
    double thetaMin = std::max({thetaMin_123, thetaMin_456});
    double lMax = 10. / thetaMin;
    // Create container
    ApertureStatisticsCovarianceContainer container;
    container.apertureStatisticsCovariance = this;
    container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
    container.thetas_456 = std::vector<double>{theta4, theta5, theta6};

    // Do integration
    double result, error;
    if (type == "infinite")
    {
        double vals_min[3] = {lMin, lMin, 0};
        double vals_max[3] = {lMax, lMax, M_PI}; // use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

        hcubature_v(1, integrand_T1, &container, 3, vals_min, vals_max, 0, 0, 1e-6, ERROR_L1, &result, &error);

        result *= 2 / thetaMax / thetaMax / 8 / M_PI / M_PI / M_PI;
    }
    else if (type == "circle" || type == "square")
    {
        double vMax = lMax;
        double vals_min[6] = {-vMax, -vMax, -lMax, -lMax, -lMax, -lMax};
        double vals_max[6] = {1.01 * vMax, 1.01 * vMax, 1.01 * lMax, 1.01 * lMax, 1.01 * lMax, 1.01 * lMax};
        hcubature_v(1, integrand_T1, &container, 6, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);

        result /= pow(2 * M_PI, 6);
    }
    else
    {
        throw std::logic_error("T1: Wrong survey geometry");
    };

    return result;
}

double ApertureStatisticsCovariance::T2(const double &theta1, const double &theta2, const double &theta3,
                                        const double &theta4, const double &theta5, const double &theta6)
{
    if (type == "infinite")
        return 0;

    // Integral over ell 1
    std::vector<double> thetas1{theta1, theta2};
    ApertureStatisticsCovarianceContainer container;
    container.apertureStatisticsCovariance = this;
    container.thetas_123 = thetas1;
    double thetaMin = std::min({theta1, theta2});
    double lMax = 10. / thetaMin;

    double result_A1, error_A1;

    double vals_min1[1] = {lMin};
    double vals_max1[1] = {lMax};
    hcubature_v(1, integrand_T2_part1, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_A1, &error_A1);

    // Integral over ell 3
    std::vector<double> thetas2{theta5, theta6};
    container.thetas_123 = thetas2;
    thetaMin = std::min({theta5, theta6});
    lMax = 10. / thetaMin;
    double result_A2, error_A2;

    hcubature_v(1, integrand_T2_part1, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_A2, &error_A2);

    // Integral over ell 2
    std::vector<double> thetas3{theta3, theta4};
    container.thetas_123 = thetas3;
    thetaMin = std::min({theta3, theta4});
    lMax = 10. / thetaMin;
    double result_B, error_B;
    if (type == "circle")
    {
        hcubature_v(1, integrand_T2_part2, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_B, &error_B);
    }
    else if (type == "square")
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

double ApertureStatisticsCovariance::T4(const double &theta1, const double &theta2, const double &theta3,
                                        const double &theta4, const double &theta5, const double &theta6)
{
    // Set maximal l value such, that theta*l <= 10
    double thetaMin_123 = std::min({theta1, theta2, theta3});
    double thetaMin_456 = std::min({theta4, theta5, theta6});
    double thetaMin = std::max({thetaMin_123, thetaMin_456});
    double lMax = 10. / thetaMin;

    // Create container
    ApertureStatisticsCovarianceContainer container;
    container.apertureStatisticsCovariance = this;
    container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
    container.thetas_456 = std::vector<double>{theta4, theta5, theta6};
    // Do integration
    double result, error;
    if (type == "infinite")
    {
        double vals_min[5] = {lMin, lMin, lMin, 0, 0};
        double vals_max[5] = {lMax, lMax, lMax, M_PI, M_PI}; // use symmetry, integrate only from 0 to pi and multiply result by 2 in the end

        hcubature_v(1, integrand_T4, &container, 5, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
        result = 4. * result / thetaMax / thetaMax / pow(2 * M_PI, 5);
    }
    else if (type == "circle")
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

int ApertureStatisticsCovariance::integrand_T1(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T1: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;
    ApertureStatisticsCovariance *apertureStatisticsCovariance = container_->apertureStatisticsCovariance;

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    };

#pragma omp parallel for
    for (unsigned int i = 0; i < npts; i++)
    {
        if (apertureStatisticsCovariance->type == "infinite")
        {
            double ell1 = vars[i * ndim];
            double ell2 = vars[i * ndim + 1];
            double phi = vars[i * ndim + 2];

            value[i] = apertureStatisticsCovariance->integrand_T1_infinite(ell1, ell2, phi, container_->thetas_123.at(0),
                                                                           container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                                           container_->thetas_456.at(0), container_->thetas_456.at(1), container_->thetas_456.at(2));
        }
        else if (apertureStatisticsCovariance->type == "circle")
        {
            double ell1X = vars[i * ndim];
            double ell1Y = vars[i * ndim + 1];
            double ell2X = vars[i * ndim + 2];
            double ell2Y = vars[i * ndim + 3];
            double ell3X = vars[i * ndim + 4];
            double ell3Y = vars[i * ndim + 5];
            value[i] = apertureStatisticsCovariance->integrand_T1_circle(ell1X, ell1Y, ell2X, ell2Y, ell3X, ell3Y, container_->thetas_123.at(0),
                                                                         container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                                         container_->thetas_456.at(0), container_->thetas_456.at(1), container_->thetas_456.at(2));
        }
        else if (apertureStatisticsCovariance->type == "square")
        {
            double ell1X = vars[i * ndim];
            double ell1Y = vars[i * ndim + 1];
            double ell2X = vars[i * ndim + 2];
            double ell2Y = vars[i * ndim + 3];
            double ell3X = vars[i * ndim + 4];
            double ell3Y = vars[i * ndim + 5];
            value[i] = apertureStatisticsCovariance->integrand_T1_square(ell1X, ell1Y, ell2X, ell2Y, ell3X, ell3Y, container_->thetas_123.at(0),
                                                                         container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                                         container_->thetas_456.at(0), container_->thetas_456.at(1), container_->thetas_456.at(2));
        }
        else // This should not happen
        {
            exit(-1);
        }
    }

    return 0; // Success :)
}

int ApertureStatisticsCovariance::integrand_T2_part1(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T2_part1: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;
    ApertureStatisticsCovariance *apertureStatisticsCovariance = container_->apertureStatisticsCovariance;

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    };

#pragma omp parallel for
    for (unsigned int i = 0; i < npts; i++)
    {
        if (apertureStatisticsCovariance->type == "circle" || apertureStatisticsCovariance->type == "square")
        {
            double ell = vars[i * ndim];
            value[i] = apertureStatisticsCovariance->integrand_T2_part1(ell, container_->thetas_123.at(0), container_->thetas_123.at(1));
        }
        else // This should not happen
        {
            exit(-1);
        }
    }

    return 0; // Success :)
}

int ApertureStatisticsCovariance::integrand_T2_part2(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T2_part2: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;
    ApertureStatisticsCovariance *apertureStatisticsCovariance = container_->apertureStatisticsCovariance;

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    };

#pragma omp parallel for
    for (unsigned int i = 0; i < npts; i++)
    {
        if (apertureStatisticsCovariance->type == "circle")
        {
            double ell = vars[i * ndim];
            value[i] = 2 * M_PI * apertureStatisticsCovariance->integrand_T2_part2_circle(ell, container_->thetas_123.at(0), container_->thetas_123.at(1));
        }
        else if (apertureStatisticsCovariance->type == "square")
        {
            double ellX = vars[i * ndim];
            double ellY = vars[i * ndim + 1];
            value[i] = apertureStatisticsCovariance->integrand_T2_part2_square(ellX, ellY, container_->thetas_123.at(0), container_->thetas_123.at(1));
        }
        else // This should not happen
        {
            exit(-1);
        }
    }

    return 0; // Success :)
}

int ApertureStatisticsCovariance::integrand_T4(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T4: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;
    ApertureStatisticsCovariance *apertureStatisticsCovariance = container_->apertureStatisticsCovariance;

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
    };

#pragma omp parallel for
    for (unsigned int i = 0; i < npts; i++)
    {
        if (apertureStatisticsCovariance->type == "circle")
        {
            double v = vars[i * ndim];
            double ell2 = vars[i * ndim + 1];
            double ell4 = vars[i * ndim + 2];
            double ell5 = vars[i * ndim + 3];
            double alphaV = vars[i * ndim + 4];
            double alpha2 = vars[i * ndim + 5];
            double alpha4 = vars[i * ndim + 6];
            double alpha5 = vars[i * ndim + 7];
            value[i] = apertureStatisticsCovariance->integrand_T4_circle(v, ell2, ell4, ell5, alphaV, alpha2, alpha4, alpha5,
                                                                         container_->thetas_123.at(0), container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                                         container_->thetas_456.at(0), container_->thetas_456.at(1), container_->thetas_456.at(2));
        }
        else if (apertureStatisticsCovariance->type == "infinite")
        {
            double ell1 = vars[i * ndim];
            double ell2 = vars[i * ndim + 1];
            double ell5 = vars[i * ndim + 2];
            double phi1 = vars[i * ndim + 3];
            double phi2 = vars[i * ndim + 4];
            value[i] = apertureStatisticsCovariance->integrand_T4_infinite(ell1, ell2, ell5, phi1, phi2,
                                                                           container_->thetas_123.at(0), container_->thetas_123.at(1), container_->thetas_123.at(2),
                                                                           container_->thetas_456.at(0), container_->thetas_456.at(1), container_->thetas_456.at(2));
        }
        else // THis should not happen
        {
            exit(-1);
        }
    }

    return 0; // Success :)
}

double ApertureStatisticsCovariance::integrand_T1_circle(const double &a, const double &b, const double &c, const double &d, const double &e, const double &f,
                                                         const double &theta1, const double &theta2, const double &theta3, const double &theta4,
                                                         const double &theta5, const double &theta6)
{
    double ell = sqrt(a * a + b * b);
    double Gfactor = G_circle(ell);

    double ell1 = sqrt((a - c - e) * (a - c - e) + (b - d - f) * (b - d - f));
    double ell2 = sqrt(c * c + d * d);
    double ell3 = sqrt(e * e + f * f);

    if (ell1 <= lMin || ell2 <= lMin || ell3 <= lMin)
        return 0;

    double result = apertureStatistics->Bispectrum_->Pell(ell1) * apertureStatistics->Bispectrum_->Pell(ell2) * apertureStatistics->Bispectrum_->Pell(ell3);
    result *= apertureStatistics->uHat(ell1 * theta1) * apertureStatistics->uHat(ell2 * theta2) * apertureStatistics->uHat(ell3 * theta3);
    result *= apertureStatistics->uHat(ell1 * theta4) * apertureStatistics->uHat(ell2 * theta5) * apertureStatistics->uHat(ell3 * theta6);
    result *= Gfactor;

    return result;
}

double ApertureStatisticsCovariance::integrand_T1_square(const double &a, const double &b, const double &c, const double &d, const double &e, const double &f,
                                                         const double &theta1, const double &theta2, const double &theta3, const double &theta4,
                                                         const double &theta5, const double &theta6)
{
    double Gfactor = G_square(a, b);

    double ell1 = sqrt((a - c - e) * (a - c - e) + (b - d - f) * (b - d - f));
    double ell2 = sqrt(c * c + d * d);
    double ell3 = sqrt(e * e + f * f);

    if (ell1 <= lMin || ell2 <= lMin || ell3 <= lMin)
        return 0;

    double result = apertureStatistics->Bispectrum_->Pell(ell1) * apertureStatistics->Bispectrum_->Pell(ell2) * apertureStatistics->Bispectrum_->Pell(ell3);
    result *= apertureStatistics->uHat(ell1 * theta1) * apertureStatistics->uHat(ell2 * theta2) * apertureStatistics->uHat(ell3 * theta3);
    result *= apertureStatistics->uHat(ell1 * theta4) * apertureStatistics->uHat(ell2 * theta5) * apertureStatistics->uHat(ell3 * theta6);
    result *= Gfactor;

    return result;
}

double ApertureStatisticsCovariance::integrand_T1_infinite(const double &l1, const double &l2, const double &phi,
                                                           const double &theta1, const double &theta2, const double &theta3, const double &theta4,
                                                           const double &theta5, const double &theta6)
{
    double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));

    if (l1 <= lMin || l2 <= lMin || l3 <= lMin)
        return 0;
    double result = apertureStatistics->Bispectrum_->Pell(l1) * apertureStatistics->Bispectrum_->Pell(l2) * apertureStatistics->Bispectrum_->Pell(l3);
    result *= apertureStatistics->uHat(l1 * theta1) * apertureStatistics->uHat(l2 * theta2) * apertureStatistics->uHat(l3 * theta3);
    result *= apertureStatistics->uHat(l1 * theta4) * apertureStatistics->uHat(l2 * theta5) * apertureStatistics->uHat(l3 * theta6);
    result *= l1 * l2;

    return result;
}

double ApertureStatisticsCovariance::integrand_T2_part1(const double &ell, const double &theta1, const double &theta2)
{
    if (ell <= lMin)
        return 0;
    double result = ell * apertureStatistics->Bispectrum_->Pell(ell);
    result *= apertureStatistics->uHat(ell * theta1) * apertureStatistics->uHat(ell * theta2);
    return result;
}

double ApertureStatisticsCovariance::integrand_T2_part2_circle(const double &ell, const double &theta1, const double &theta2)
{
    if (ell <= lMin)
        return 0;
    double Gfactor = G_circle(ell);
    double result = apertureStatistics->Bispectrum_->Pell(ell);
    result *= ell * apertureStatistics->uHat(ell * theta1) * apertureStatistics->uHat(ell * theta2);
    result *= Gfactor;

    return result;
}

double ApertureStatisticsCovariance::integrand_T2_part2_square(const double &ellX, const double &ellY, const double &theta1, const double &theta2)
{
    double Gfactor = G_square(ellX, ellY);
    double ell = sqrt(ellX * ellX + ellY * ellY);
    if (ell <= lMin)
        return 0;
    double result = apertureStatistics->Bispectrum_->Pell(ell);
    result *= apertureStatistics->uHat(ell * theta1) * apertureStatistics->uHat(ell * theta2);
    result *= Gfactor;

    return result;
}

double ApertureStatisticsCovariance::integrand_T4_circle(const double &v, const double &ell2, const double &ell4, const double &ell5,
                                                         const double &alphaV, const double &alpha2, const double &alpha4, const double &alpha5,
                                                         const double &theta1, const double &theta2, const double &theta3,
                                                         const double &theta4, const double &theta5, const double &theta6)

{
    double ell1 = sqrt(v * v + ell4 * ell4 - 2 * v * ell4 * cos(alpha4 - alphaV));
    double ell3 = sqrt(ell1 * ell1 + ell2 * ell2 + 2 * ell1 * ell2 * cos(alpha2 - alpha4 + alphaV));
    double ell6 = sqrt(ell4 * ell4 + ell5 * ell5 + 2 * ell4 * ell5 * cos(alpha5 - alpha4));

    double Gfactor = G_circle(v);

    double result = apertureStatistics->Bispectrum_->bkappa(ell4, ell2, ell3);
    result *= apertureStatistics->Bispectrum_->bkappa(ell1, ell5, ell6);
    result *= apertureStatistics->uHat(ell1 * theta1) * apertureStatistics->uHat(ell2 * theta2) * apertureStatistics->uHat(ell3 * theta3) * apertureStatistics->uHat(ell4 * theta4) * apertureStatistics->uHat(ell5 * theta5) * apertureStatistics->uHat(ell6 * theta6);
    result *= v * ell2 * ell4 * ell5;
    result *= Gfactor;

    return result;
}

double ApertureStatisticsCovariance::integrand_T4_infinite(const double &l1, const double &l2, const double &l5, const double &phi1, const double &phi2,
                                                           const double &theta1, const double &theta2, const double &theta3,
                                                           const double &theta4, const double &theta5, const double &theta6)
{
    double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi1));
    double l6 = sqrt(l1 * l1 + l5 * l5 + 2 * l1 * l5 * cos(phi2));

    double result = apertureStatistics->uHat(l1 * theta4) * apertureStatistics->uHat(l2 * theta2) * apertureStatistics->uHat(l3 * theta3) * apertureStatistics->uHat(l1 * theta1) * apertureStatistics->uHat(l5 * theta5) * apertureStatistics->uHat(l6 * theta6);

    result *= apertureStatistics->Bispectrum_->bkappa(l1, l2, l3);
    result *= apertureStatistics->Bispectrum_->bkappa(l1, l5, l6);
    result *= l1 * l2 * l5;
    return result;
}

double ApertureStatisticsCovariance::G_circle(const double &ell)
{
    double tmp = thetaMax * ell;
    double result = gsl_sf_bessel_J1(tmp);
    result *= result;
    result *= 4 / tmp / tmp;
    return result;
}

double ApertureStatisticsCovariance::G_square(const double &ellX, const double &ellY)
{
    double tmp1 = 0.5 * ellX * thetaMax;
    double tmp2 = 0.5 * ellY * thetaMax;

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