#include "apertureStatistics.hpp"
#include "helper.hpp"
#include <iostream>
#include <chrono>

#define CALCULATE_TERM1 false

int main()
{
    // Set up Cosmology
    struct cosmology cosmo;
    double thetaMax, z_max;
    if (slics)
    {
        std::cerr << "using SLICS cosmology..." << std::endl;
        cosmo.h = 0.6898;               // Hubble parameter
        cosmo.sigma8 = 0.826;           // sigma 8
        cosmo.omb = 0.0473;             // Omega baryon
        cosmo.omc = 0.2905 - cosmo.omb; // Omega CDM
        cosmo.ns = 0.969;               // spectral index of linear P(k)
        cosmo.w = -1.0;
        cosmo.om = cosmo.omb + cosmo.omc;
        cosmo.ow = 1 - cosmo.om;

        thetaMax = 8.93;
        z_max = 3;
    }
    else
    {
        std::cerr << "using Millennium cosmology..." << std::endl;
        cosmo.h = 0.73;
        cosmo.sigma8 = 0.9;
        cosmo.omb = 0.045;
        cosmo.omc = 0.25 - cosmo.omb;
        cosmo.ns = 1.;
        cosmo.w = -1.0;
        cosmo.om = cosmo.omc + cosmo.omb;
        cosmo.ow = 1. - cosmo.om;

        thetaMax = 4;
        z_max = 1.1;
    }

    int n_z = 100;

    double sigma = 0.3;
    double n = 4096 * 4096 / 10.0 / 10.0;
    std::string folder, type;
#if CONSTANT_POWERSPECTRUM
    folder = "/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields/";
    type = "shapenoise";
#else
    folder = "/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS/";
    type = "slics";
#endif

    std::cerr << "Fieldsize:" << thetaMax << "x" << thetaMax << " deg^2" << std::endl;
    std::cerr << "Shape noise:" << sigma << std::endl;
    std::cerr << "n:" << n << " [1/deg^2]" << std::endl;
    std::cerr << "Writing results to folder " << folder << std::endl;
#if CALCULATE_TERM1
    std::cerr<< "Calculates Term1 and Term2"<<std::endl;
#else
    std::cerr<<"Calculates only Term2"<<std::endl;
#endif

#if CONSTANT_POWERSPECTRUM
    std::cerr << "Warning: Using constant powerspectrum" << std::endl;
#endif

    double thetaMaxRad = convert_angle_to_rad(thetaMax, "deg"); // Convert to radians

    // Initialize Bispectrum
    BispectrumCalculator bispectrum(&cosmo, n_z, z_max, false);
    bispectrum.sigma = sigma;
    bispectrum.n = n * 180 * 180 / M_PI / M_PI;

    std::cerr << "Finished setting bispectrum" << std::endl;

    // Initialize Aperture Statistics
    ApertureStatistics apertureStatistics(&bispectrum);

    std::cerr << "Finished setting aperture statistics" << std::endl;

    // Set up thetas for which ApertureStatistics are calculated
    std::vector<double> thetas{2, 4, 8, 16};
    int N = thetas.size();

    int N_ind = N * (N + 1) * (N + 2) / 6; // Number of independent theta-combinations
    int N_total = N_ind * N_ind;

    std::vector<double> Cov_term1s, Cov_term2s;
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
                            std::vector<double> thetas_456 = {theta4, theta5, theta6};

#if CALCULATE_TERM1
                            double term1 = apertureStatistics.L1_total(thetas_123, thetas_456, thetaMaxRad);
                            Cov_term1s.push_back(term1);
#endif

                            double term2 = apertureStatistics.L2_total(thetas_123, thetas_456, thetaMaxRad);
                            Cov_term2s.push_back(term2);

                            // Progress for the impatient user
                            auto end = std::chrono::high_resolution_clock::now();
                            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                            completed_steps++;
                            double progress = (completed_steps * 1.) / (N_total);

                            fprintf(stderr, "\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step. Last thetas: (%.1f, %.1f, %.1f, %.1f, %.1f, %.1f)",
                                    static_cast<int>(progress * 100),
                                    elapsed.count() * 1e-9 / 3600,
                                    (N_total - completed_steps) * elapsed.count() * 1e-9 / 3600 / completed_steps,
                                    elapsed.count() * 1e-9 / completed_steps,
                                    convert_rad_to_angle(theta1), convert_rad_to_angle(theta2), convert_rad_to_angle(theta3),
                                    convert_rad_to_angle(theta4), convert_rad_to_angle(theta5), convert_rad_to_angle(theta6));
                        }
                    }
                }
            }
        }
    }

    // Output
#if CALCULATE_TERM1
    char filename1[255];
    sprintf(filename1, "cov_%s_term1Numerical_sigma_%.1f_n_%.2f_thetaMax_%.2f.dat", 
                    type.c_str(), sigma, n, thetaMax);
    std::string fn_term1 = folder + std::string(filename1);
    std::cerr << "Writing Term 1 to " << fn_term1 << std::endl;
    
    std::ofstream out(fn_term1);
    if (!out.is_open())
    {
        std::cerr << "Couldn't write to " << fn_term1 << std::endl;
        std::cerr << "Writing instead to Term1.dat" << std::endl;
        out.clear();
        out.open("Term1.dat");
    };

    for (int i = 0; i < N_ind; i++)
    {
        for (int j = 0; j < N_ind; j++)
        {
            out << Cov_term1s.at(i * N_ind + j) << " ";
        }
        out << std::endl;
    }

    out.close();
    out.clear();

#endif
    char filename2[255];
    sprintf(filename2, "cov_%s_term2Numerical_sigma_%.1f_n_%.2f_thetaMax_%.2f.dat", 
                    type.c_str(), sigma, n, thetaMax);
    std::string fn_term2 = folder + std::string(filename2);
    std::cerr << "Writing Term 2 to " << fn_term2 << std::endl;

    std::ofstream out(fn_term2);
    if (!out.is_open())
    {
        std::cerr << "Couldn't write to " << fn_term2 << std::endl;
        std::cerr << "Writing instead to Term2.dat" << std::endl;
        out.clear();
        out.open("Term2.dat");
    };

    for (int i = 0; i < N_ind; i++)
    {
        for (int j = 0; j < N_ind; j++)
        {
            out << Cov_term2s.at(i * N_ind + j) << " ";
        }
        out << std::endl;
    }

    return 0;
}