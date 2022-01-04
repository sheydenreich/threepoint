#include "apertureStatistics.hpp"
#include "helper.hpp"
#include <iostream>
#include <chrono>

int main()
{
    // Set up Cosmology
    struct cosmology cosmo;

    std::cerr << "using Millennium cosmology" << std::endl;
    cosmo.h = 0.73;
    cosmo.sigma8 = 0.9;
    cosmo.omb = 0.045;
    cosmo.omc = 0.25 - cosmo.omb;
    cosmo.ns = 1.;
    cosmo.w = -1.0;
    cosmo.om = cosmo.omc + cosmo.omb;
    cosmo.ow = 1. - cosmo.om;

    double z_max = 1.1;
    int n_z = 100;

    double thetaMax = 10;
    std::cerr << "Fieldsize:" << thetaMax << "x" << thetaMax << "deg^2" << std::endl;

    thetaMax = convert_angle_to_rad(thetaMax, "deg"); // Convert to radians

    // Initialize Bispectrum
    BispectrumCalculator bispectrum(&cosmo, n_z, z_max, false);

    std::cerr<<"Finished setting bispectrum"<<std::endl;

    // Initialize Aperture Statistics
    ApertureStatistics apertureStatistics(&bispectrum);

    std::cerr<<"Finished setting aperture statistics"<<std::endl;

    // Set up thetas for which ApertureStatistics are calculated
    std::vector<double> thetas{2, 4, 8, 16};
    int N = thetas.size();

    int N_ind = N * (N + 1) * (N + 2) / 6; // Number of independent theta-combinations
    int N_total = N_ind * N_ind;

    std::vector<double> Cov_MapMapMaps;
    int completed_steps = 0;

    auto begin = std::chrono::high_resolution_clock::now(); // Begin time measurement
    for (int i = 0; i < N; i++)
    {
        double theta1 = convert_angle_to_rad(thetas.at(i), "arcmin"); // Conversion to rad
        for (int j = i; j < N; j++)
        {
            double theta2 = convert_angle_to_rad(thetas.at(j), "arcmin");
            for (int k = j; k < N; k++)
            {
                double theta3 = convert_angle_to_rad(thetas.at(k), "arcmin");
                std::vector<double> thetas_123 = {theta1, theta2, theta3};
                for (int l = 0; l < N; l++)
                {
                    double theta4 = convert_angle_to_rad(thetas.at(l), "arcmin"); // Conversion to rad
                    for (int m = l; m < N; m++)
                    {
                        double theta5 = convert_angle_to_rad(thetas.at(m), "arcmin");
                        for (int n = m; n < N; n++)
                        {
                            double theta6 = convert_angle_to_rad(thetas.at(n), "arcmin");
                                                        std::cerr<<theta1<<" "<<theta2<<" "<<theta3<<" "<<theta4<<" "<<theta5<<" "<<theta6<<std::endl;
                            std::vector<double> thetas_456 = {theta4, theta5, theta6};
                            double Cov_MapMapMap = apertureStatistics.Cov(thetas_123, thetas_456, thetaMax);

                            Cov_MapMapMaps.push_back(Cov_MapMapMap);

                            // Progress for the impatient user
                            auto end = std::chrono::high_resolution_clock::now();
                            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                            completed_steps++;
                            double progress = (completed_steps * 1.) / (N_total);


                            // printf("\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step. Current thetas: (%.1f, %.1f, %.1f, %.1f, %.1f, %.1f)",
                            //        static_cast<int>(progress * 100),
                            //        elapsed.count() * 1e-9 / 3600,
                            //        (N_total - completed_steps) * elapsed.count() * 1e-9 / 3600 / completed_steps,
                            //        elapsed.count() * 1e-9 / completed_steps,
                            //        convert_rad_to_angle(theta1), convert_rad_to_angle(theta2), convert_rad_to_angle(theta3),
                            //        convert_rad_to_angle(theta4), convert_rad_to_angle(theta5), convert_rad_to_angle(theta6));
                        }
                    }
                }
            }
        }
    }

    // Output

    for (int i = 0; i < N_ind; i++)
    {
        for (int j = 0; j < N_ind; j++)
        {
            std::cout << Cov_MapMapMaps.at(i * N_ind + j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}