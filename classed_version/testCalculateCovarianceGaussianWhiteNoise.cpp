#include "helper.hpp"
#include <iostream>
#include <vector>
#include <complex>

double U(double vartheta, double theta)
{
    double x = vartheta / theta;
    x = x * x / 2;
    return 1 / (2 * M_PI * theta * theta) * exp(-x) * (1.0 - x);
}

void getVarthetas(int Npix, double deltaTheta, std::vector<std::complex<double>> &varthetas)
{

#pragma omp parallel for
    for (int i = 0; i < Npix; i++)
    {
        double x = i * deltaTheta;

        for (int j = 0; j < Npix; j++)
        {
            double y = j * deltaTheta;
            std::complex<double> coord(x, y);
            varthetas.at(i * Npix + j) = coord;
        }
    }
}

void getAs(double theta1, double theta2, double theta3,
           double theta4, double theta5, double theta6,
           int Npix, int NpixInner, const std::vector<std::complex<double>> &varthetas, const std::vector<std::complex<double>> &varthetasInner,
           std::vector<double> &A12, std::vector<double> &A13, std::vector<double> &A14, std::vector<double> &A15,
           std::vector<double> &A16, std::vector<double> &A23, std::vector<double> &A24, std::vector<double> &A25,
           std::vector<double> &A26, std::vector<double> &A34, std::vector<double> &A35, std::vector<double> &A36,
           std::vector<double> &A45, std::vector<double> &A46, std::vector<double> &A56)
{
    int Npix2d = Npix * Npix;
    int Npix2dInner = NpixInner * NpixInner;
    std::cerr << "Setting As" << std::endl;

#pragma omp parallel for default(shared) schedule(static)
    for (int i = 0; i < Npix2dInner; i++)
    {
        std::complex<double> vartheta_i = varthetasInner.at(i);
        for (int j = 0; j < Npix2dInner; j++)
        {
            double a12, a13, a14, a15, a16, a23, a24, a25, a26, a34, a35, a36, a45, a46, a56;
            a12 = a13 = a14 = a15 = a16 = a23 = a24 = a25 = a26 = a34 = a35 = a36 = a45 = a46 = a56 = 0;
            std::complex<double> vartheta_j = varthetasInner.at(j);
            for (int k = 0; k < Npix2d; k++)
            {
                std::complex<double> vartheta_k = varthetas.at(k);
                double distance_1 = std::abs(vartheta_i - vartheta_k);
                double distance_2 = std::abs(vartheta_j - vartheta_k);
                a12 += U(distance_1, theta1) * U(distance_2, theta2);
                a13 += U(distance_1, theta1) * U(distance_2, theta3);
                a14 += U(distance_1, theta1) * U(distance_2, theta4);
                a15 += U(distance_1, theta1) * U(distance_2, theta5);
                a16 += U(distance_1, theta1) * U(distance_2, theta6);
                a23 += U(distance_1, theta2) * U(distance_2, theta3);
                a24 += U(distance_1, theta2) * U(distance_2, theta4);
                a25 += U(distance_1, theta2) * U(distance_2, theta5);
                a26 += U(distance_1, theta2) * U(distance_2, theta6);
                a34 += U(distance_1, theta3) * U(distance_2, theta4);
                a35 += U(distance_1, theta3) * U(distance_2, theta5);
                a36 += U(distance_1, theta3) * U(distance_2, theta6);
                a45 += U(distance_1, theta4) * U(distance_2, theta5);
                a46 += U(distance_1, theta4) * U(distance_2, theta6);
            }
            A13.at(i * Npix2dInner + j) = a13;
            A12.at(i * Npix2dInner + j) = a12;
            A14.at(i * Npix2dInner + j) = a14;
            A15.at(i * Npix2dInner + j) = a15;
            A16.at(i * Npix2dInner + j) = a16;
            A23.at(i * Npix2dInner + j) = a23;
            A24.at(i * Npix2dInner + j) = a24;
            A25.at(i * Npix2dInner + j) = a25;
            A26.at(i * Npix2dInner + j) = a26;
            A34.at(i * Npix2dInner + j) = a34;
            A35.at(i * Npix2dInner + j) = a35;
            A36.at(i * Npix2dInner + j) = a36;
            A45.at(i * Npix2dInner + j) = a45;
            A46.at(i * Npix2dInner + j) = a46;
            A56.at(i * Npix2dInner + j) = a56;
        }
    }
}

double term1(int Npix2dInner, std::vector<double> &A14, std::vector<double> &A15,
             std::vector<double> &A16, std::vector<double> &A24, std::vector<double> &A25,
             std::vector<double> &A26, std::vector<double> &A34, std::vector<double> &A35, std::vector<double> &A36)
{
    double t1 = 0;
    std::cerr << "Calculating Term 1" << std::endl;
#pragma omp parallel for
    for (int i = 0; i < Npix2dInner; i++)
    {
        for (int j = 0; j < Npix2dInner; j++)
        {
            t1 += A14[i * Npix2dInner + j] * A25[i * Npix2dInner + j] * A36[i * Npix2dInner + j];
            t1 += A14[i * Npix2dInner + j] * A26[i * Npix2dInner + j] * A35[i * Npix2dInner + j];
            t1 += A15[i * Npix2dInner + j] * A26[i * Npix2dInner + j] * A34[i * Npix2dInner + j];
            t1 += A15[i * Npix2dInner + j] * A24[i * Npix2dInner + j] * A36[i * Npix2dInner + j];
            t1 += A16[i * Npix2dInner + j] * A24[i * Npix2dInner + j] * A35[i * Npix2dInner + j];
            t1 += A16[i * Npix2dInner + j] * A25[i * Npix2dInner + j] * A34[i * Npix2dInner + j];
        }
    }
    return t1;
}

double term2(int Npix2dInner, std::vector<double> &A12, std::vector<double> &A13, std::vector<double> &A14, std::vector<double> &A15,
             std::vector<double> &A16, std::vector<double> &A23, std::vector<double> &A24, std::vector<double> &A25,
             std::vector<double> &A26, std::vector<double> &A34, std::vector<double> &A35, std::vector<double> &A36,
             std::vector<double> &A45, std::vector<double> &A46, std::vector<double> &A56)
{
    double t2 = 0;
    std::cerr << "Calculating Term 2" << std::endl;
#pragma omp parallel for
    for (int i = 0; i < Npix2dInner; i++)
    {
        for (int j = 0; j < Npix2dInner; j++)
        {
            t2 += A12[i * Npix2dInner + i] * A34[i * Npix2dInner + j] * A56[j * Npix2dInner + j];
            t2 += A12[i * Npix2dInner + i] * A35[i * Npix2dInner + j] * A46[j * Npix2dInner + j];
            t2 += A12[i * Npix2dInner + i] * A36[i * Npix2dInner + j] * A45[j * Npix2dInner + j];
            t2 += A13[i * Npix2dInner + i] * A24[i * Npix2dInner + j] * A56[j * Npix2dInner + j];
            t2 += A13[i * Npix2dInner + i] * A25[i * Npix2dInner + j] * A46[j * Npix2dInner + j];
            t2 += A13[i * Npix2dInner + i] * A26[i * Npix2dInner + j] * A45[j * Npix2dInner + j];
            t2 += A23[i * Npix2dInner + i] * A14[i * Npix2dInner + j] * A56[j * Npix2dInner + j];
            t2 += A23[i * Npix2dInner + i] * A15[i * Npix2dInner + j] * A46[j * Npix2dInner + j];
            t2 += A23[i * Npix2dInner + i] * A16[i * Npix2dInner + j] * A45[j * Npix2dInner + j];
        }
    }
    return t2;
}

void totalCov(double theta1, double theta2, double theta3, double theta4, double theta5, double theta6,
              int Npix, int NpixInner, double Apix, double sigma, const std::vector<std::complex<double>> &varthetas, const std::vector<std::complex<double>> &varthetasInner,
              double &t1, double &t2, double &total)
{
    int Npix2dInner = NpixInner * NpixInner;
    double norm = pow(Apix * sigma, 6) / Npix2dInner / Npix2dInner;
    std::vector<double> A12(Npix2dInner * Npix2dInner), A13(Npix2dInner * Npix2dInner), A14(Npix2dInner * Npix2dInner), A15(Npix2dInner * Npix2dInner), A16(Npix2dInner * Npix2dInner), A23(Npix2dInner * Npix2dInner), A24(Npix2dInner * Npix2dInner),
        A25(Npix2dInner * Npix2dInner), A26(Npix2dInner * Npix2dInner), A34(Npix2dInner * Npix2dInner), A35(Npix2dInner * Npix2dInner), A36(Npix2dInner * Npix2dInner), A45(Npix2dInner * Npix2dInner), A46(Npix2dInner * Npix2dInner), A56(Npix2dInner * Npix2dInner);
    getAs(theta1, theta2, theta3, theta4, theta5, theta6, Npix, NpixInner, varthetas, varthetasInner, A12, A13, A14, A15,
          A16, A23, A24, A25, A26, A34, A35, A36, A45, A46, A56);
    std::cerr << "Finished calculating As" << std::endl;
    t1 = term1(Npix2dInner, A14, A15, A16, A24, A25, A26, A34, A35, A36);
    t1 *= norm;
    std::cerr << "Term1:" << t1 << std::endl;
    t2 = term2(Npix2dInner, A12, A13, A14, A15,
               A16, A23, A24, A25, A26, A34, A35, A36, A45, A46, A56);
    t2 *= norm;
    std::cerr << "Term2:" << t2 << std::endl;
    std::cerr << "Term1:" << t1 << " "
              << "Term2:" << t2 << std::endl;
    total = t1 + t2;
    std::cerr << "Total:" << total << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Need to specify Npix, sidelength and NpixInner" << std::endl;
        exit(1);
    };
    int Npix = std::stoi(argv[1]);
    double side = convert_angle_to_rad(std::stod(argv[2]), "deg");
    int NpixInner = std::stoi(argv[3]);

    double A = side * side;
    double Apix = A / Npix / Npix;
    double deltaTheta = side / Npix;
    double sigma = 0.3;
    std::cerr << "Finished setting basics" << std::endl;
    std::vector<double> thetas{2, 8}; //,4,8,16};
    int N = thetas.size();

    std::vector<std::complex<double>> varthetas(Npix * Npix);
    getVarthetas(Npix, deltaTheta, varthetas);
    std::vector<std::complex<double>> varthetasInner(NpixInner * NpixInner);
    getVarthetas(NpixInner, deltaTheta, varthetasInner);
    std::cerr << "Finished creating varthetas " << varthetas.size() << std::endl;

    for (int i = 0; i < N; i++)
    {
        double theta1 = convert_angle_to_rad(thetas.at(i));
        for (int j = i; j < N; j++)
        {
            double theta2 = convert_angle_to_rad(thetas.at(j));
            for (int k = j; k < N; k++)
            {
                double theta3 = convert_angle_to_rad(thetas.at(k));
                for (int l = i; l < N; l++)
                {
                    double theta4 = convert_angle_to_rad(thetas.at(l));
                    for (int m = l; m < N; m++)
                    {
                        double theta5 = convert_angle_to_rad(thetas.at(m));
                        for (int n = m; n < N; n++)
                        {
                            if (l + m + n >= i + j + k)
                            {
                                double theta6 = convert_angle_to_rad(thetas.at(n));
                                double t1, t2, total;
                                totalCov(theta1, theta2, theta3, theta4, theta5, theta6, Npix, NpixInner, Apix, sigma, varthetas, varthetasInner, t1, t2, total);
                                std::cout << theta1 << " "
                                          << theta2 << " "
                                          << theta3 << " "
                                          << theta4 << " "
                                          << theta5 << " "
                                          << theta6 << " "
                                          << t1 << " " << t2 << " " << total << std::endl;
                            };
                        }
                    }
                }
            }
        }
    }

    return 0;
}