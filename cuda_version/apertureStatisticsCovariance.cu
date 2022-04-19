#include "apertureStatisticsCovariance.cuh"
#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"
#include "halomodel.cuh"


#include "cubature.h"


#include <algorithm>
#include <iostream>

__constant__ double dev_thetaMax;
__constant__ double dev_lMin;
__constant__ double dev_lMax;
double lMin;
double thetaMax;
int type; //0: circle, 1: square, 2: infinite

__constant__ double dev_sigmaW;
double sigmaW, VW, chiMax;

void writeCov(const std::vector<double> &values, const int &N, const std::string &filename)
{
    if (values.size() != N * (N+1)/2)
    {
        throw std::out_of_range("writeCov: Values has wrong length");
    };

    std::ofstream out(filename);
    if (!out.is_open())
    {
        throw std::runtime_error("writeCov: Could not open " + filename);
    };

    std::vector<double> cov_tmp(N*N);
    int ix=0;
    for (int i=0; i<N; i++)
    {
        for(int j=i; j<N; j++)
        {
            cov_tmp.at(i*N+j)=values.at(ix);
            ix+=1;
        }
    }

    for(int i=0; i<N; i++)
    {
        for(int j=0; j<i; j++)
        {
            cov_tmp.at(i*N+j)=cov_tmp.at(j*N+i);
        }
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            out << cov_tmp.at(i * N + j) << " ";
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

    return T4_total(thetas_123, thetas_456)+T5_total(thetas_123, thetas_456);
}

double T1_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T1_total: Wrong number of aperture radii");
    };

    double th0=thetas_123.at(0);
    double th1=thetas_123.at(1);
    double th2=thetas_123.at(2);
    double th3=thetas_456.at(0);
    double th4=thetas_456.at(1);
    double th5=thetas_456.at(2);

    double result;


    if(th3 == th4)
    {
        if(th4 == th5) // All thetas_456 are equal
        {
            result = 6* T1(th0, th1, th2, th3, th3, th3);
        }
        else
        {
            result = 2 * T1(th0, th1, th2, th3, th3, th5);
            result += 2 * T1(th0, th1, th2, th3, th5, th3);
            result += 2 * T1(th0, th1, th2, th5, th3, th3);
        }
    }
    else if(th3 == th5)
    {
        result = 2 * T1(th0, th1, th2, th3, th4, th3);
        result += 2 * T1(th0, th1, th2, th3, th3, th4);
        result += 2 * T1(th0, th1, th2, th4, th3, th3);
    }
    else if(th4 == th5)
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

    double th0=thetas_123.at(0);
    double th1=thetas_123.at(1);
    double th2=thetas_123.at(2);
    double th3=thetas_456.at(0);
    double th4=thetas_456.at(1);
    double th5=thetas_456.at(2);
    
    if(th0==th1 && th1 == th2) // All thetas_123 are the same
    {
        if(th3 == th4 && th4 == th5) // All thetas_456 are the same
        {
            result = 9*T2(th0, th0, th0, th3, th3, th3);
        }
        else if(th3 == th4) // th3=th4 and th4 \neq th5
        {
            result = 6*T2(th0, th0, th0, th3, th3, th5);
            result += 3*T2(th0, th0, th0, th5, th3, th3);
        }
        else if(th3 == th5) // th3=th5 and th4 \neq th5
        {
            result = 6*T2(th0, th0, th0, th3, th4, th3);
            result += 3*T2(th0, th0, th0, th4, th3, th3);
        }
        else if(th4 == th5) // th4 = th5 and th3 \neq th4
        {
            result = 6*T2(th0, th0, th0, th4, th3, th4);
            result += 3*T2(th0, th0, th0, th3, th4, th4);
        }
        else // All thetas_456 are different
        {
            result = 3*T2(th0, th0, th0, th3, th4, th5);
            result += 3*T2(th0, th0, th0, th4, th3, th5);
            result += 3*T2(th0, th0, th0, th5, th3, th4);
        }
    }
    else if(th0==th1) // th0=th1 and th0 \neq th2
    {
        if(th3==th4 && th4==th5)
        {
            result = 6*T2(th0, th2, th0, th3, th3, th3);
            result += 3*T2(th0, th0, th2, th3, th3, th3);
        }
        else if(th3==th4) /// th3=th4 and th4 \neq th5
        {
            result = 2*T2(th0, th0, th2, th3, th3, th5);
            result += T2(th0, th0, th2, th5, th3, th3);
            result += 4*T2(th0, th2, th0, th3, th3, th5);
            result += 2*T2(th0, th2, th0, th5, th3, th3);
        }
        else if(th3==th5)
        {
            result = 2*T2(th0, th0, th2, th3, th4, th3);
            result += T2(th0, th0, th2, th4, th3, th3);
            result += 4*T2(th0, th2, th0, th3, th4, th3);
            result += 2*T2(th0, th2, th0, th4, th3, th3);
        }
        else if(th4==th5)
        {
            result = 2*T2(th0, th0, th2, th4, th3, th4);
            result += T2(th0, th0, th2, th3, th4, th4);
            result += 4*T2(th0, th2, th0, th4, th3, th4);
            result += 2*T2(th0, th2, th0, th3, th4, th4);
        }
        else
        {
            result = T2(th0, th0, th2, th3, th4, th5);
            result += T2(th0, th0, th2, th4, th3, th5);
            result += T2(th0, th0, th2, th5, th3, th4);
            result += 2*T2(th0, th2, th0, th3, th4, th5);
            result += 2*T2(th0, th2, th0, th4, th3, th5);
            result += 2*T2(th0, th2, th0, th5, th3, th4);
        }
    }
    else if(th1==th2) // th1=th2 and th0 \neq th1
    {
        if(th3==th4 && th4==th5)
        {
            result = 6*T2(th0, th1, th1, th3, th3, th3);
            result += 3*T2(th1, th1, th0, th3, th3, th3);
        }
        else if(th3==th4)
        {
            result = 2*T2(th0, th1, th1, th5, th3, th3);
            result += T2(th1, th1, th0, th5, th3, th3);
            result += 4*T2(th0, th1, th1, th3, th3, th5);
            result += 2*T2(th1, th1, th0, th3, th3, th5);
        }
        else if(th3==th5)
        {
            result = 4*T2(th0, th1, th1, th3, th4, th3);
            result += 2*T2(th0, th1, th1, th4, th3, th3);
            result += 2*T2(th1, th1, th0, th3, th4, th3);
            result += T2(th1, th1, th0, th4, th3, th3);
        }
        else if(th4==th5)
        {
            result = 2*T2(th0, th1, th1, th3, th4, th4);
            result += T2(th1, th1, th0, th3, th4, th4);
            result += 4*T2(th0, th1, th1, th4, th3, th4);
            result += 2*T2(th1, th1, th0, th4, th3, th4);
        }
        else
        {
            result = 2*T2(th0, th1, th1, th3, th4, th5);
            result += 2*T2(th0, th1, th1, th4, th3, th5);
            result += 2*T2(th0, th1, th1, th5, th3, th4);
            result += T2(th1, th1, th0, th3, th4, th5);
            result += T2(th1, th1, th0, th4, th3, th5);
            result += T2(th1, th1, th0, th5, th3, th4);
        }
    }
    else if(th0==th2) // th0=th2 and th0 \neq th1
    {
        if(th3==th4 && th4==th5)
        {
            result = 6*T2(th0, th1, th0, th3, th3, th3);
            result += 3*T2(th1, th0, th0, th3, th3, th3);
        }
        else if(th3==th4)
        {
            result = 2*T2(th0, th1, th0, th5, th3, th3);
            result += T2(th1, th0, th0, th5, th3, th3);
            result += 4*T2(th0, th1, th0, th3, th3, th5);
            result += 2*T2(th1, th0, th0, th3, th3, th5);
        }
        else if(th3==th5)
        {
            result = 4*T2(th0, th1, th0, th3, th4, th3);
            result += 2*T2(th0, th1, th0, th4, th3, th3);
            result += 2*T2(th1, th0, th0, th3, th4, th3);
            result += T2(th1, th0, th0, th4, th3, th3);
        }
        else if(th4==th5)
        {
            result = 2*T2(th0, th1, th0, th3, th4, th4);
            result += T2(th1, th0, th0, th3, th4, th4);
            result += 4*T2(th0, th1, th0, th4, th3, th4);
            result += 2*T2(th1, th0, th0, th4, th3, th4);
        }
        else
        {
            result = 2*T2(th0, th1, th0, th3, th4, th5);
            result += 2*T2(th0, th1, th0, th4, th3, th5);
            result += 2*T2(th0, th1, th0, th5, th3, th4);
            result += T2(th1, th0, th0, th3, th4, th5);
            result += T2(th1, th0, th0, th4, th3, th5);
            result += T2(th1, th0, th0, th5, th3, th4);
        }
    }
    else // All thetas_123 are different
    {
        if(th3 == th4 && th4 == th5)
        {
            result = 3*T2(th0, th1, th2, th3, th3, th3);
            result += 3*T2(th0, th2, th1, th3, th3, th3);
            result += 3*T2(th1, th2, th0, th3, th3, th3);
        }
        else if(th3 == th4)
        {
            result = 2* T2(th0, th1, th2, th3, th3, th5);
            result += T2(th0, th1, th2, th5, th3, th3);
            result += 2*T2(th0, th2, th1, th3, th3, th5);
            result += T2(th0, th2, th1, th5, th3, th3);
            result += 2*T2(th1, th2, th0, th3, th3, th5);
            result += T2(th1, th2, th0, th5, th3, th3);
        }
        else if(th3 == th5)
        {
            result = 2* T2(th0, th1, th2, th3, th4, th3);
            result += T2(th0, th1, th2, th4, th3, th3);
            result += 2*T2(th0, th2, th1, th3, th4, th3);
            result += T2(th0, th2, th1, th4, th3, th3);
            result += 2*T2(th1, th2, th0, th3, th4, th3);
            result += T2(th1, th2, th0, th4, th3, th3);
        }
        else if(th4 == th5)
        {
            result = 2* T2(th0, th1, th2, th4, th3, th4);
            result += T2(th0, th1, th2, th3, th4, th4);
            result += 2*T2(th0, th2, th1, th4, th3, th4);
            result += T2(th0, th2, th1, th3, th4, th4);
            result += 2*T2(th1, th2, th0, th4, th3, th4);
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

    result /= pow(2 * M_PI, 4);

    return result;
}


double T4_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T4_total: Wrong number of aperture radii");
    };

    double th0=thetas_123.at(0);
    double th1=thetas_123.at(1);
    double th2=thetas_123.at(2);
    double th3=thetas_456.at(0);
    double th4=thetas_456.at(1);
    double th5=thetas_456.at(2);

    double result;

    if(th0==th1 && th0 == th2)
    {

        if(th3==th4 && th3==th5)
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th0, th3, th3);
            result = 9*T4(th3, th0, th0, th0, th3, th3);
        }
        else if(th3 == th4)
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th0, th3, th5);
            result = 6*T4(th3, th0, th0, th0, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th0, th0, th3, th3);
            result+=3*T4(th5, th0, th0, th0, th3, th3);
        }
        else
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th0, th4, th5);
            result = 3*T4(th3, th0, th0, th0, th4, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th0, th0, th0, th3, th5);
            result+=3*T4(th4, th0, th0, th0, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th0, th0, th3, th4);
            result+=3*T4(th5, th0, th0, th0, th3, th4);
        }
    }
    else if(th0==th1)
    {
        if(th3==th4 && th3==th5)
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th0, th3, th3);
            result = 6*T4(th3, th0, th2, th0, th3, th3);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th2, th3, th3);
            result+=3*T4(th3, th0, th0, th2, th3, th3);
        }
        else if (th3==th4)
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th2, th3, th5);
            result = 2*T4(th3, th0, th0, th2, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th0, th3, th5);
            result+=4*T4(th3, th0, th2, th0, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th0, th2, th3, th3);
            result+=T4(th5, th0, th0, th2, th3, th3);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th2, th0, th3, th3);
            result+=2*T4(th5, th0, th2, th0, th3, th3);
        }
        else
        {
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th0, th2, th4, th5);
            result = T4(th3, th0, th0, th2, th4, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th0, th4, th5);
            result+=2*T4(th3, th0, th2, th0, th4, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th0, th0, th2, th3, th5);
            result+=T4(th4, th0, th0, th2, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th0, th2, th0, th3, th5);
            result+=2*T4(th4, th0, th2, th0, th3, th5);
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th0, th2, th3, th4);
            result+=T4(th5, th0, th0, th2, th3, th4);;
            printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th2, th0, th3, th4);
            result+=2*T4(th5, th0, th2, th0, th3, th4);
        }
        
    }
    else if(th3==th4)
    {
        if(th3==th5)
        {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th1, th2, th3, th3);
        result = 3*T4(th3, th0, th1, th2, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th1, th3, th3);
        result+=3*T4(th3, th0, th2, th1, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th1, th2, th0, th3, th3);
        result+=3*T4(th3, th1, th2, th0, th3, th3);
        }
        else
        {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th1, th2, th3, th5);
        result = 2*T4(th3, th0, th1, th2, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th1, th3, th5);
        result+=2*T4(th3, th0, th2, th1, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th1, th2, th0, th3, th5);
        result+=2*T4(th3, th1, th2, th0, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th1, th2, th3, th4);
        result += T4(th5, th0, th1, th2, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th1, th2, th0, th3, th3);
        result+=T4(th5, th1, th2, th0, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th2, th1, th3, th3);
        result+=T4(th5, th0, th2, th1, th3, th3);
        }
    }
    else if(th4==th5)
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th1, th2, th3, th5);
        result = T4(th3, th0, th1, th2, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th1, th3, th5);
        result+=T4(th3, th0, th2, th1, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th1, th2, th0, th3, th5);
        result+=T4(th3, th1, th2, th0, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th1, th2, th3, th4);
        result += 2*T4(th4, th0, th1, th2, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th1, th2, th0, th3, th3);
        result+=2*T4(th4, th1, th2, th0, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th2, th1, th3, th3);
        result+=2*T4(th4, th0, th2, th1, th3, th4);
        }
    else
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th1, th2, th4, th5);
        result = T4(th3, th0, th1, th2, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th0, th2, th1, th4, th5);
        result+=T4(th3, th0, th2, th1, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th1, th2, th0, th4, th5);
        result+=T4(th3, th1, th2, th0, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th0, th1, th2, th3, th5);
        result+=T4(th4, th0, th1, th2, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th0, th2, th1, th3, th5);
        result+=T4(th4, th0, th2, th1, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th1, th2, th0, th3, th5);
        result+=T4(th4, th1, th2, th0, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th1, th2, th3, th4);
        result+=T4(th5, th0, th1, th2, th3, th4);;
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th0, th2, th1, th3, th4);
        result+=T4(th5, th0, th2, th1, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th5, th1, th2, th0, th3, th4);
        result+=T4(th5, th1, th2, th0, th3, th4);
    }

    result /= pow(2 * M_PI, 8);
    return result;
}

double T5_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T4_total: Wrong number of aperture radii");
    };

    double th0=thetas_123.at(0);
    double th1=thetas_123.at(1);
    double th2=thetas_123.at(2);
    double th3=thetas_456.at(0);
    double th4=thetas_456.at(1);
    double th5=thetas_456.at(2);

    double result;

    if(th0==th1)
    {
        if(th0==th1)
        {
            if(th3==th4 && th3==th5)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th3, th3);
                result = 9*T5(th0, th0, th0, th3, th3, th3);
            }
            else if(th3==th4)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th3, th5);
                result = 6*T5(th0, th0, th0, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th5, th3, th3);
                result+=3*T5(th0, th0, th0, th5, th3, th3);
            }
            else if(th4==th5)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th4, th4);
                result = 3*T5(th0, th0, th0, th3, th4, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th4, th3, th4);
                result+=6*T5(th0, th0, th0, th4, th3, th4);
            }
            else
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th4, th5);
                result = 3*T5(th0, th0, th0, th3, th4, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th4, th3, th5);
                result+=3*T5(th0, th0, th0, th4, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th5, th3, th4);
                result+=3*T5(th0, th0, th0, th5, th3, th4);
            }
        }
        else
        {
            if(th3==th4 && th3==th5)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th3, th3);
                result = 6*T5(th0, th0, th2, th3, th3, th3);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th3, th3, th3);
                result+=3*T5(th2, th0, th0, th3, th3, th3);
            }
            else if(th3==th4)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th3, th5);
                result = 4*T5(th0, th0, th2, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th5, th3, th3);
                result+=2*T5(th0, th0, th2, th5, th3, th3);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th3, th3, th5);
                result+=2*T5(th2, th0, th0, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th5, th3, th3);
                result+=T5(th2, th0, th0, th5, th3, th3);
            }
            else if(th4==th5)
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th4, th3, th4);
                result = 4*T5(th0, th0, th2, th4, th3, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th4, th4);
                result+=2*T5(th0, th0, th2, th3, th4, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th4, th3, th4);
                result+=2*T5(th2, th0, th0, th4, th3, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th3, th4, th4);
                result+=T5(th2, th0, th0, th3, th4, th4);

            }
            else
            {
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th4, th5);
                result = 2*T5(th0, th0, th2, th3, th4, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th4, th3, th5);
                result+=2*T5(th0, th0, th2, th4, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th5, th3, th4);
                result+=2*T5(th0, th0, th2, th5, th3, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th3, th4, th5);
                result+=T5(th2, th0, th0, th3, th4, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th4, th3, th5);
                result+=T5(th2, th0, th0, th4, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th0, th5, th3, th4);
                result+=T5(th2, th0, th0, th5, th3, th4);
            }
        }

    }
    else if(th3==th4 && th3==th5)
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th3, th3);
        result = 3*T5(th0, th1, th2, th3, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th3, th3, th3);
        result+=3*T5(th1, th0, th2, th3, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th3, th3, th3);
        result+=3*T5(th2, th0, th1, th3, th3, th3);
    }
    else if(th3==th4)
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th3, th5);
        result = 2*T5(th0, th1, th2, th3, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th5, th3, th3);
        result+=T5(th0, th1, th2, th5, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th3, th4, th5);
        result+=2*T5(th1, th0, th2, th3, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th5, th3, th3);
        result+=T5(th1, th0, th2, th5, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th3, th3, th5);
        result+=2*T5(th2, th0, th1, th3, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th5, th3, th3);
        result+=T5(th2, th0, th1, th5, th3, th3);
    }
    else if(th4==th5)
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th4, th4);
        result = T5(th0, th1, th2, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th4, th3, th4);
        result+=2*T5(th0, th1, th2, th4, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th3, th4, th4);
        result+=T5(th1, th0, th2, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th4, th3, th4);
        result+=2*T5(th1, th0, th2, th4, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th3, th4, th4);
        result+=T5(th2, th0, th1, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th4, th3, th4);
        result+=2*T5(th2, th0, th1, th4, th3, th4);
    }
    else
    {
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th4, th5);
        result = T5(th0, th1, th2, th3, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th4, th3, th5);
        result+=T5(th0, th1, th2, th4, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th5, th3, th4);
        result+=T5(th0, th1, th2, th5, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th3, th4, th5);
        result+=T5(th1, th0, th2, th3, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th4, th3, th5);
        result+=T5(th1, th0, th2, th4, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th0, th2, th5, th3, th4);
        result+=T5(th1, th0, th2, th5, th3, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th3, th4, th5);
        result+=T5(th2, th0, th1, th3, th4, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th4, th3, th5);
        result+=T5(th2, th0, th1, th4, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th2, th0, th1, th5, th3, th4);
        result+=T5(th2, th0, th1, th5, th3, th4);
        
    }

    result /= pow(2 * M_PI, 8);
    return result;
}

double T6_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T4_total: Wrong number of aperture radii");
    };

    double th0=thetas_123.at(0);
    double th1=thetas_123.at(1);
    double th2=thetas_123.at(2);
    double th3=thetas_456.at(0);
    double th4=thetas_456.at(1);
    double th5=thetas_456.at(2);

    double result;

    if(th0==th1)
    {
        if(th0==th2)
        {
            if(th3==th4 && th3==th5)
            {
                //A
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th3, th3);
                result = 3*T6(th0, th0, th0, th3, th3, th3);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th0, th0, th3);
                result += 3*T6(th3, th3, th0, th0, th0, th3);
            }
            else if(th3==th4)
            {
                //B
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th3, th5);
                result = 3*T6(th0, th0, th0, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th0, th0, th5);
                result += T6(th3, th3, th0, th0, th0, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th5, th0, th0, th0, th3);
                result += 2*T6(th3, th5, th0, th0, th0, th3);
            }
            else if(th4==th5)
            {
                //C
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th4, th4);
                result = 3*T6(th0, th0, th0, th3, th4, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th4, th0, th0, th0, th4);
                result += 2*T6(th3, th4, th0, th0, th0, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th4, th0, th0, th0, th3);
                result += T6(th4, th4, th0, th0, th0, th3);
            }
            else
            {
                //D
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th0, th3, th4, th5);
                result = 3*T6(th0, th0, th0, th3, th4, th5);
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
            if(th3==th4 && th3==th5)
            {
                //E
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th3, th3);
                result = T6(th0, th0, th2, th3, th3, th3);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th0, th3, th3, th3);
                result += 2*T6(th0, th2, th0, th3, th3, th3);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th0, th2, th3);
                result += 3*T6(th3, th3, th0, th0, th2, th3);
            }
            else if(th3==th4)
            {
                //F
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th3, th5);
                result = T6(th0, th0, th2, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th0, th3, th3, th5);
                result += 2*T6(th0, th2, th0, th3, th3, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th0, th2, th5);
                result += T6(th3, th3, th0, th0, th2, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th5, th0, th0, th2, th3);
                result += 2*T6(th3, th5, th0, th0, th2, th3);
            }
            else if(th4==th5)
            {
                //G
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th4, th4);
                result = T6(th0, th0, th2, th3, th4, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th0, th3, th4, th4);
                result += 2*T6(th0, th2, th0, th3, th4, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th4, th0, th0, th2, th4);
                result += 2*T6(th3, th4, th0, th0, th2, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th4, th0, th0, th2, th3);
                result += T6(th4, th4, th0, th0, th2, th3);
            }
            else
            {
                //H
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th0, th2, th3, th4, th5);
                result = T6(th0, th0, th2, th3, th4, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th0, th3, th4, th5);
                result += 2*T6(th0, th2, th0, th3, th4, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th4, th0, th0, th2, th5);
                result += T6(th3, th4, th0, th0, th2, th5);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th5, th0, th0, th2, th4);
                result += T6(th3, th5, th0, th0, th2, th4);
                printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th5, th0, th0, th2, th3);
                result += T6(th4, th5, th0, th0, th2, th3);
            }
        }
    }
    else if(th3==th4 && th3==th5)
    {
        //I
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th3, th3);
        result = T6(th0, th1, th2, th3, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th1, th3, th3, th3);
        result += T6(th0, th2, th1, th3, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th2, th0, th3, th3, th3);
        result += T6(th1, th2, th0, th3, th3, th3);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th1, th2, th3);
        result += 3*T6(th3, th3, th0, th1, th2, th3);
    }
    else if(th3==th4)
    {
        //J
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th3, th5);
        result = T6(th0, th1, th2, th3, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th1, th3, th3, th5);
        result += T6(th0, th2, th1, th3, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th2, th0, th3, th3, th5);
        result += T6(th1, th2, th0, th3, th3, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th3, th0, th1, th2, th5);
        result += T6(th3, th3, th0, th1, th2, th5);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th5, th0, th1, th2, th3);
        result += 2*T6(th3, th5, th0, th1, th2, th3);
    }
    else if(th4==th5)
    {
        //K
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th1, th2, th3, th4, th4);
        result = T6(th0, th1, th2, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th0, th2, th1, th3, th4, th4);
        result += T6(th0, th2, th1, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th1, th2, th0, th3, th4, th4);
        result += T6(th1, th2, th0, th3, th4, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th3, th4, th0, th1, th2, th4);
        result += 2*T6(th3, th4, th0, th1, th2, th4);
        printf("Doing %f, %f, %f, %f, %f, %f \n", th4, th4, th0, th1, th2, th3);
        result += T6(th4, th4, th0, th1, th2, th3);
    }
    else
    {
        //L
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





    if(th0==th1 && th1==th2) // All thetas_123 are equal
    {
        if(th3==th4 && th4==th5) // All thetas_456 are equal
        {
 
        }
        else
        {
 
        }
    }
    else if(th3==th4 && th4==th5) //All thetas_456 are equal
    {

    }
    else
    {

    }

    result /= pow(2 * M_PI, 6);
    return result;
}

double T7_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
{
    if (thetas_123.size() != 3 || thetas_456.size() != 3)
    {
        throw std::invalid_argument("T7_total: Wrong number of aperture radii");
    };

    double th0=thetas_123.at(0);
    double th1=thetas_123.at(1);
    double th2=thetas_123.at(2);
    double th3=thetas_456.at(0);
    double th4=thetas_456.at(1);
    double th5=thetas_456.at(2);

    double result;
    result=T7(th0, th1, th2, th3, th4,th5);


    result /= pow(2 * M_PI, 6);
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

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));

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
    hcubature_v(1, integrand_T1, &container, 6, vals_min, vals_max, 0, 0, 1e-3, ERROR_L1, &result, &error);

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
    double thetaMin=std::min({theta1, theta2, theta3, theta4, theta5});
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

double T4(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
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
    double vals_min[5] = {lMin, lMin, lMin, 0, 0};
    double vals_max[5] = {lMax, lMax, lMax, 2*M_PI, 2*M_PI}; 

    hcubature_v(1, integrand_T4, &container, 5, vals_min, vals_max, 0, 0, 1e-1, ERROR_L1, &result, &error);
    result = result / thetaMax / thetaMax * pow(2 * M_PI, 2);
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


double T5(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
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
        double mmin=pow(10, logMmin);
        double mmax=pow(10, logMmax);
    double vals_min[7] = {lMin, lMin, lMin, 0, 0, mmin, 0};
    double vals_max[7] = {lMax, lMax, lMax, 2*M_PI, 2*M_PI, mmax, z_max}; 

    hcubature_v(1, integrand_T5, &container, 7, vals_min, vals_max, 0, 0, 1e-1, ERROR_L1, &result, &error);
    result = result / thetaMax / thetaMax * pow(2 * M_PI, 2);
    }
    else
    {
    throw std::logic_error("T5: Wrong survey geometry, only coded for infinite survey");
    };

    return result;
}

double T6(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
{
    if(type!=1)
    {
        throw std::logic_error("T6: Wrong survey geometry, only coded for square survey");
        };
    // Set maximal l value such, that theta*l <= 10
    double thetaMin_123 = std::min({theta1, theta2, theta3});
    double thetaMin_456 = std::min({theta4, theta5, theta6});
    double thetaMin = std::max({thetaMin_123, thetaMin_456});
    double lMax = 10. / thetaMin;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
    // Integral over ell 1
    std::vector<double> thetas1{theta1, theta2};
    ApertureStatisticsCovarianceContainer container;
    container.thetas_123 = thetas1;
    
    double result_A1, error_A1;
    
    double vals_min1[1] = {lMin};
    double vals_max1[1] = {lMax};
    hcubature_v(1, integrand_T2_part1, &container, 1, vals_min1, vals_max1, 0, 0, 1e-4, ERROR_L1, &result_A1, &error_A1);
    

    // Integral over ell3 to ell5


    // Create container

    container.thetas_123 = std::vector<double>{theta1, theta2, theta3};
    container.thetas_456 = std::vector<double>{theta4, theta5, theta6};

    // Do integration
    double result, error;

    double mmin=pow(10, logMmin);
    double mmax=pow(10, logMmax);
    // double vals_min[8] = {lMin, lMin, lMin, 0, 0, 0, mmin, 0};
    // double vals_max[8] = {lMax, lMax, lMax, 2*M_PI, 2*M_PI, 2*M_PI, mmax, z_max}; 

    // hcubature_v(1, integrand_T6, &container, 8, vals_min, vals_max, 0, 0, 1e-1, ERROR_L1, &result, &error);

    double vals_min[7] = {lMin, lMin, lMin, 0, 0, 0, mmin};
    double vals_max[7] = {lMax, lMax, lMax, 2*M_PI, 2*M_PI, 2*M_PI, mmax}; 

    hcubature_v(1, integrand_T6, &container, 7, vals_min, vals_max, 1e8, 0, 1e-1, ERROR_L1, &result, &error);


    std::cerr<<result_A1<<" "<<result<<std::endl;
    return result_A1* result;
}


double T7(const double &theta1, const double &theta2, const double &theta3, const double &theta4, const double &theta5, const double &theta6)
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
        double mmin=pow(10, logMmin);
        double mmax=pow(10, logMmax);
    double vals_min[7] = {lMin, lMin, lMin, lMin, 0, 0, mmin};
    double vals_max[7] = {lMax, lMax, lMax, lMax, 2*M_PI, 2*M_PI, mmax}; 

    hcubature_v(1, integrand_T7, &container, 7, vals_min, vals_max, 0, 0, 1e-1, ERROR_L1, &result, &error);
    result = result / thetaMax / thetaMax;
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
        return 1;
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
    else if(type==2)
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

int integrand_T5(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T5: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
        return 1;
    };

    // Allocate memory on device for integrand values
    double* dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

    // Copy integration variables to device
    double* dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

    // Calculate values
    if(type==2)
    {
        integrand_T5_infinite<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
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

int integrand_T6(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T6: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;
    //std::cerr<<npts<<std::endl;
    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
        return 1;
    };

    // Allocate memory on device for integrand values
    double* dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

    // Copy integration variables to device
    double* dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

    // Calculate values
    if(type==1)
    {
        integrand_T6_square<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
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

int integrand_T7(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value)
{
    if (fdim != 1)
    {
        std::cerr << "integrand_T7: wrong function dimension" << std::endl;
        return -1;
    };

    ApertureStatisticsCovarianceContainer *container_ = (ApertureStatisticsCovarianceContainer *)container;

    if (npts > 1e8)
    {
        std::cerr << "WARNING: Large number of points: " << npts << std::endl;
        return 1;
    };

    // Allocate memory on device for integrand values
    double* dev_value;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

    // Copy integration variables to device
    double* dev_vars;
    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

    // Calculate values
    if(type==2)
    {
        integrand_T7_infinite<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, container_->thetas_123.at(0),
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
    
        if (ell1 <= 0 || ell2 <= 0 || ell3 <= 0 || ell3>dev_lMax)
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

        if (l1 <= dev_lMin || l2 <= dev_lMin || l3 <= dev_lMin || l3>dev_lMax)
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

        if(l3 > dev_lMax || l6 > dev_lMax || l3 <= 0 || l6 <= 0)
        {
            value[i]=0;
        }
        else
        {    
        double result = uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) 
                        * uHat(l1 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);
    
        result *= bkappa(l1, l2, l3);
        result *= bkappa(l1, l5, l6);
        result *= l1 * l2 * l5;

        value[i]=result;
        }
    }

}

__global__ void integrand_T5_infinite(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
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
        double m=vars[i*ndim+5];
        double z=vars[i*ndim+6];

        double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi1));
        double l6 = sqrt(l1 * l1 + l5 * l5 + 2 * l1 * l5 * cos(phi2));
        double l4=l1;

        if(l1<=dev_lMin || l2<=dev_lMin || l3<=dev_lMin || l4<=dev_lMin || l5<=dev_lMin || l6<=dev_lMin || l3>dev_lMax || l6 >dev_lMax)
        {
            value[i]=0;
        }
        else
        {

            double result = uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) 
            * uHat(l4 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);

            result *= dev_Pell(l1);
            result *= trispectrum_integrand(m, z, l2, l3, l5, l6);
            result *= l1 * l2 * l5;

            //printf("%e %f %e %e %e %e %.2e \n", m, z, l2, l3, l5, l6, trispectrum_integrand(m, z, l2, l3, l5, l6));
            value[i]=result;
        }
    }

}


__global__ void integrand_T6_square(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
    double theta4, double theta5, double theta6, double* value)
{
int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double l3=vars[i*ndim];
        double l4=vars[i*ndim+1];
        double l5=vars[i*ndim+2];
        double phi3=vars[i*ndim+3];
        double phi4=vars[i*ndim+4];
        double phi5=vars[i*ndim+5];
        double m=vars[i*ndim+6];
      //  double z=vars[i*ndim+7];

        double l3x = l3*cos(phi3);
        double l3y = l3*sin(phi3);
        double l6 = l3*l3+l4*l4+l5*l5+2*(l3*l4*cos(phi3-phi4)-l3*l5*cos(phi3-phi5)-l4*l5*cos(phi4-phi5));
        if(l6>0)
        {
        l6=sqrt(l6);
        }
        else
        {
            //printf("l6 is negative\n");
            l6=0;
        }

        if(l3<=dev_lMin || l4<=dev_lMin || l5<=dev_lMin || l6<=dev_lMin || l6 > dev_lMax)
        {
            value[i]=0;
        }
        else
        {
            double Gfactor = G_square(l3x, l3y);
            double result = uHat(l3 * theta3) 
            * uHat(l4 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);

            double trispec=trispectrum_limber_integrated(0, dev_z_max, m, l3, l4, l5, l6);//trispectrum_integrand(m, z, l3, l4, l5, l6);
            result*=trispec;
            result *= l3 * l4 * l5;
            result*=Gfactor;


            value[i]=result;
        }
    }

}

__global__ void integrand_T7_infinite(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
    double theta4, double theta5, double theta6, double* value)
{
int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

    for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
        double l1=vars[i*ndim];
        double l2=vars[i*ndim+1];
        double l4=vars[i*ndim+2];
        double l5=vars[i*ndim+3];
        double phi1=vars[i*ndim+4];
        double phi2=vars[i*ndim+5];
        double m=vars[i*ndim+6];

        double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi1));
        double l6 = sqrt(l1 * l1 + l5 * l5 + 2 * l1 * l5 * cos(phi2));

        if(l1<=dev_lMin || l2<=dev_lMin || l3<=dev_lMin || l4<=dev_lMin || l5<=dev_lMin || l6<=dev_lMin || l3>dev_lMax || l6 >dev_lMax)
        {
            value[i]=0;
        }
        else
        {

            double result = uHat(l1 * theta1) * uHat(l2 * theta2) * uHat(l3 * theta3) 
            * uHat(l4 * theta4) * uHat(l5 * theta5) * uHat(l6 * theta6);

            double pentaspec=pentaspectrum_limber_integrated(0, dev_z_max, m, l1, l2, l3, l4, l5, l6);
            result *= pentaspec;
            result *= l1 * l2 * l4 * l5;

            //printf("%e %f %e %e %e %e %.2e \n", m, z, l2, l3, l5, l6, trispectrum_integrand(m, z, l2, l3, l5, l6));
            value[i]=result;
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


// /// THE FOLLOWING THINGS ARE NEEDED FOR THE SUPERSAMPLE COVARIANCE

// double Cov_SSC(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
// {
//     double cx = z_max / 2;
//     double q=0;
//     for(int i=0; i<48; i++)
//     {
//        std::cerr<<"Cov_SSC Redshift integral "<<i<<"/47"<<std::endl;
//         q+=W96[i]*(integrand_Cov_SSC(cx-cx*A96[i], thetas_123, thetas_456)+integrand_Cov_SSC(cx+cx*A96[i], thetas_123, thetas_456));
//     }
//     std::cerr<<"Cov_SSC:"<<q*cx<<std::endl;
//     return q*cx/thetaMax/thetaMax;
// }

// double integrand_Cov_SSC(const double& z, const std::vector<double> &thetas_123, const std::vector<double> &thetas_456)
// {
//     double chi= c_over_H0*GQ96_of_Einv(0,z);
//     double result = f(z, thetas_123, chi)*f(z, thetas_456, chi);
//     result*=sigmaW*c_over_H0*E_inv(z);


//     double didx = z / z_max * (n_redshift_bins - 1);
//     int idx = didx;
//     didx = didx - idx;
//     if (idx == n_redshift_bins - 1)
//     {
//       idx = n_redshift_bins - 2;
//       didx = 1.;
//     }
//     double g= g_array[idx] * (1 - didx) + g_array[idx + 1] * didx;
//     result*=pow(g, 6);
//     result/=pow(chi, 10);
//     std::cerr<<"z:"<<z<<" integrand_Cov_SSC:"<<result<<std::endl;
//     return result;
// }

// double f(const double& z, const std::vector<double> &thetas_123, const double& chi)
// {
//     ApertureStatisticsSSCContainer container;
//     container.thetas_123=thetas_123;
//     container.z=z;
//     container.chi=chi;

//     double thetaMin=std::min({thetas_123.at(0), thetas_123.at(1), thetas_123.at(2)});
//     double lMax = 10./thetaMin;
//     double result, error;

//     double mmin=pow(10, logMmin);
//     double mmax=pow(10, logMmax);


//     double vals_min[4]={lMin, lMin, 0, mmin};
//     double vals_max[4]={lMax, lMax, 2*M_PI, mmax};

//     hcubature_v(1, integrand_f, &container, 4, vals_min, vals_max, 0, 0, 1e-2, ERROR_L1, &result, &error);

//     return result;
// }

// int integrand_f(unsigned ndim, size_t npts, const double* vars, void* container, unsigned fdim, double* value)
// {
//     if (fdim != 1)
//     {
//         std::cerr << "integrand_f: wrong function dimension" << std::endl;
//         return -1;
//     };

//     ApertureStatisticsSSCContainer *container_ = (ApertureStatisticsSSCContainer *) container;

//     if (npts > 1e8)
//     {
//         std::cerr << "WARNING: Large number of points: " << npts << std::endl;
//     };

//    // Allocate memory on device for integrand values
//    double* dev_value;
//    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

//    // Copy integration variables to device
//    double* dev_vars;
//    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
//    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

//    integrand_f<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, dev_value, container_->thetas_123.at(0), container_->thetas_123.at(1), container_->thetas_123.at(2), container_->z, container_->chi);
//     CUDA_SAFE_CALL(cudaFree(dev_vars));

//     CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

//     CUDA_SAFE_CALL(cudaFree(dev_value));
//     return 0;
// }

// __global__ void integrand_f(const double* vars, unsigned ndim, int npts, double* value, double theta1, double theta2, double theta3, double z, double chi)
// {
//     int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

//     for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
//     {
//         double l1=vars[i*ndim];
//         double l2=vars[i*ndim+1];
//         double phi=vars[i*ndim+2];
//         double m=vars[i*ndim+3];

//         //printf("Integration vars: %f %f %f %f\n", l1, l2, phi, m);

//         double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));
//         if (l3==0)
//         {
//             value[i]=0;
//         }
//         else
//         {
//         double rho_m=dev_rhobar(z);

//         double result = uHat(l1*theta1)*uHat(l2*theta2)*uHat(l3*theta3);
//         result*=hmf(m, z)*halobias(m, z)*m*m*m/rho_m/rho_m/rho_m;
//         result*=u_NFW(l1/chi, m, z)*u_NFW(l2/chi, m, z)*u_NFW(l3/chi, m, z);
//         //printf("Integration vars: %f %f %f %f %f %E\n", l1, l2, phi, m, l3, result);
//         value[i]=result;
//         };
//     };
// }

// __device__ double halobias(const double& m, const double& z)
// {
//     double q=0.707;
//     double p=0.3;
    
//     // Get sigma^2(m, z)
//     double sigma2=get_sigma2(m, z);
  
//     // Get critical density contrast
//     double deltac=delta_c(z);
  
//     double result=1+1./deltac*(q*deltac*deltac/sigma2 - 1 + 2*p/(1+pow(q*deltac*deltac/sigma2, p)));
//     return result;
// }





// __device__ double delta_c(const double& z)
// {
//     return 1.686*(1+0.0123*log10(dev_om_m_of_z(z)));
// }



// void setSurveyVolume()
// {
//     chiMax=GQ96_of_Einv(0, z_max);
//     chiMax*=c_over_H0;
//     VW = thetaMax*thetaMax*chiMax*chiMax*chiMax/3;
// }

// void setSigmaW()
// {
//     sigmaW=1;
//     // SigmaContainer container;

    
//     // double k_min[3]={-1e10, -1e10, -1e10};
//     // double k_max[3]={1e10, 1e10, 1e10};
//     // double result, error;
    
//     // hcubature_v(1, integrand_SigmaW, &container, 3, k_min, k_max, 0, 0, 1e-2, ERROR_L1, &result, &error);
  
//     // result/=(8*M_PI*M_PI*M_PI*VW*VW);

//     // sigmaW=result;
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigmaW, &sigmaW, sizeof(double)));
// }

// int integrand_SigmaW(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value)
// {
//     if(fdim!=1)
//     {
//       std::cerr<<"integrand_SigmaW: Wrong fdim"<<std::endl;
//       exit(1);
//     };
//   SigmaContainer* container = (SigmaContainer*) thisPtr;

//   if(npts>1e8)
//   {
//       std::cerr<<"WARNING: large number of points:"<<npts<<std::endl;
//   };
 
  
//  #pragma omp parallel for
//   for(unsigned int i=0; i<npts; i++)
//     {
//       double k1=k[i*ndim];
//       double k2=k[i*ndim+1];
//       double k3=k[i*ndim+2];
    
//       double k=sqrt(k1*k1+k2*k2+k3*k3);
//         if(k==0)
//         {
//             value[i]=0;
//         }
//         else
//         {
//         double W=WindowSurvey(k1, k2, k3);
//         if(isnan(W))
//         {
//         printf("k1: %f k2: %f k3: %f Window: %E\n", k1, k2, k3, W);
//         };
//          value[i]=W*linear_pk(k);
//         };
//     };
  
//   return 0;
// }

// double WindowSurvey(const double& k1, const double& k2, const double& k3)
// {
//     double a=k3;
//     double b=k1*thetaMax/2;
//     double c=k2*thetaMax/2;
//     double d=chiMax;
//     double result;
//     if(a==0)
//     {
//         if(b*b==c*c) result=0;
//         else
//         {
//             result=c*sin(b*d)*cos(c*d)-b*sin(c*d)*cos(b*d);
//             result*=result;
//             result/=(b*b+c*c)*(b*b+c*c);
//         };
//     }
//     else
//     {
//         result=(pow(a*((a - b - c)*(a + b + c)*cos((b - c)*d) - 
//         (a + b - c)*(a - b + c)*cos((b + c)*d))*sin(a*d) - 
//      cos(a*d)*((a - b - c)*(b - c)*(a + b + c)*
//          sin((b - c)*d) - 
//         (a + b - c)*(a - b + c)*(b + c)*sin((b + c)*d)),2) + 
//    pow(a*(a + b - c)*(a - b + c) - 
//      a*(a - b - c)*(a + b + c) + 
//      a*cos(a*d)*((a - b - c)*(a + b + c)*cos((b - c)*d) - 
//         (a + b - c)*(a - b + c)*cos((b + c)*d)) + 
//      sin(a*d)*((a - b - c)*(b - c)*(a + b + c)*
//          sin((b - c)*d) - 
//         (a + b - c)*(a - b + c)*(b + c)*sin((b + c)*d)),2))/
//  (4.*pow(a + b - c,2)*pow(a - b + c,2)*pow(-a + b + c,2)*
//    pow(a + b + c,2));
//     };

//     return result;
// }


// void initSSC()
// {
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(devLogMmin, &logMmin, sizeof(double)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(devLogMmax, &logMmax, sizeof(double)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_mbins, &n_mbins, sizeof(int)));
//     initCovariance();
//     std::cerr<<"Finished copying bispectrum basics"<<std::endl;
//     setSurveyVolume();
//     std::cerr<<"Finished setting survey volumne"<<std::endl;
//     std::cerr<<"VW:"<<VW<<std::endl;
//     setSigmaW();
//     std::cerr<<"Finished calculating sigma_W"<<std::endl;
//     std::cerr<<"sigma_W:"<<sigmaW<<std::endl;
//     setSigma2();
//     std::cerr<<"Finished calculating sigma²(m)"<<std::endl;
//     setdSigma2dm();
//     std::cerr<<"Finished calculating dSigma²dm"<<std::endl;
//     std::cerr<<"Finished all initializations"<<std::endl;
// }


// double Cov_Bispec_SSC(const double& ell)
// {
//     double cx = z_max / 2;
//     double q=0;
//     for(int i=0; i<48; i++)
//     {
//         q+=W96[i]*(integrand_Cov_Bispec_SSC(cx-cx*A96[i], ell)+integrand_Cov_Bispec_SSC(cx+cx*A96[i], ell));
//     }
//     return q*cx/thetaMax/thetaMax;
// }

// double integrand_Cov_Bispec_SSC(const double& z, const double& ell)
// {
//     double chi= c_over_H0*GQ96_of_Einv(0,z);
//     double result = I3(ell/chi, ell/chi, ell/chi, z, chi)*I3(ell/chi, ell/chi, ell/chi, z, chi);
//     result*=sigmaW*c_over_H0*E_inv(z);


//     double didx = z / z_max * (n_redshift_bins - 1);
//     int idx = didx;
//     didx = didx - idx;
//     if (idx == n_redshift_bins - 1)
//     {
//       idx = n_redshift_bins - 2;
//       didx = 1.;
//     }
//     double g= g_array[idx] * (1 - didx) + g_array[idx + 1] * didx;
//     result*=pow(g, 6);
//     result/=pow(chi, 10);
//     return result;
// }

// double I3(const double& k1, const double& k2, const double& k3, const double& z, const double& chi)
// {
//     ApertureStatisticsSSCContainer container;
//     container.z=z;
//     container.chi=chi;
//     container.ell1=k1;
//     container.ell2=k2;
//     container.ell3=k3;

//     double mmin=pow(10, logMmin);
//     double mmax=pow(10, logMmax);
//     double vals_min[1]={mmin};
//     double vals_max[1]={mmax};
//     double result, error;
//     hcubature_v(1, integrand_I3, &container, 1, vals_min, vals_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

//     return result;
// }

// int integrand_I3(unsigned ndim, size_t npts, const double* vars, void* container, unsigned fdim, double* value)
// {
//     if (fdim != 1)
//     {
//         std::cerr << "integrand_f: wrong function dimension" << std::endl;
//         return -1;
//     };

//     ApertureStatisticsSSCContainer *container_ = (ApertureStatisticsSSCContainer *) container;

//     if (npts > 1e8)
//     {
//         std::cerr << "WARNING: Large number of points: " << npts << std::endl;
//     };


//    // Allocate memory on device for integrand values
//    double* dev_value;
//    CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));

//    // Copy integration variables to device
//    double* dev_vars;
//    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim*npts*sizeof(double))); //alocate memory
//    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim*npts*sizeof(double), cudaMemcpyHostToDevice)); //copying

//    integrand_I3<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, dev_value, container_->ell1, container_->ell2, container_->ell3, container_->z, container_->chi);
//     CUDA_SAFE_CALL(cudaFree(dev_vars));

//     CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

//     CUDA_SAFE_CALL(cudaFree(dev_value));
//     return 0;
// }

// __global__ void integrand_I3(const double* vars, unsigned ndim, int npts, double* value, double k1, double k2, double k3, double z, double chi)
//  {
//     int thread_index=blockIdx.x*blockDim.x+threadIdx.x;

//     for (int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
//     {
//         double m=vars[i*ndim];

//         //printf("Integration vars: %f %f %f %f\n", l1, l2, phi, m);


//         double rho_m=dev_rhobar(z);

//         double result=hmf(m, z)*halobias(m, z)*m*m*m/rho_m/rho_m/rho_m;
//         result*=u_NFW(k1, m, z)*u_NFW(k2, m, z)*u_NFW(k3, m, z);
//        //printf("%E %E \n", m, hmf(m, z));
//         //printf("Integration vars: %f %f %f %f %f %E\n", l1, l2, phi, m, l3, result);
//         value[i]=result;
        
//     };  
//  }
