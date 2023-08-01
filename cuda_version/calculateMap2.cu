#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cosmology.cuh"
#include "cuda_helpers.cuh"
#include "helpers.cuh"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
/**
 * @file calculateApertureStatistics.cu
 * This executable calculates <MapMapMap> from the
 * Takahashi+ Bispectrum
 * Aperture radii are read from file and <MapMapMap> is only calculated for
 * independent combis of thetas Code uses CUDA and cubature library  (See
 * https://github.com/stevengj/cubature for documentation)
 * @author Pierre Burger
 */
int main(int argc, char *argv[])
{
    // Read in command line

    const char *message = R"( 
calculateApertureStatistics.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas [arcmin]
Argument 3: Outputfilename, directory needs to exist 
Argument 4: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateApertureStatistics.x ../necessary_files/MR_cosmo.dat ../necessary_files/HOWLS_thetas.dat ../../results_MR/MapMapMap_bispec_gpu_nz.dat ../necessary_files/nz_MR.dat
)";

    if (argc < 4) // Give out error message if too few CLI arguments
    {
        std::cerr << message << std::endl;
        exit(1);
    };

    std::string cosmo_paramfile = argv[1];
    std::string z_combi_file = argv[2];
    std::string theta_combi_file = argv[3];
    std::string output_file = argv[4];

    int Ntomo = std::stoi(argv[5]);

    std::vector<std::string> nzfns;
    for (int i = 0; i < Ntomo; i++)
    {
        std::string nzfn = argv[6 + i];
        nzfns.push_back(nzfn);
    }
    std::string shape_noise_file = argv[6 + Ntomo];

    std::string outputmode = "full";
    if (argc == 8 + Ntomo)
        outputmode = argv[7 + Ntomo]; // Either "full" or "mcmc"

    // Check if output file can be opened
    std::ofstream out;
    out.open(output_file.c_str());
    if (!out.is_open())
    {
        std::cerr << "Couldn't open " << output_file << std::endl;
        exit(1);
    };

    std::vector<std::vector<double>> theta_combis;
    std::vector<std::vector<int>> z_combis;
    int n_combis;
    read_combis(z_combi_file, theta_combi_file, z_combis, theta_combis, n_combis);

    std::vector<double> sigma_epsilon_per_bin;
    std::vector<double> ngal_per_bin;
    read_shapenoise(shape_noise_file, sigma_epsilon_per_bin, ngal_per_bin);

    // Read in cosmology
    cosmology cosmo(cosmo_paramfile);

    // Read in n_z
    std::vector<std::vector<double>> nzs;
    for (int i = 0; i < Ntomo; i++)
    {
        std::vector<double> nz;
        read_n_of_z(nzfns.at(i), n_redshift_bins, cosmo.zmax, nz);
        nzs.push_back(nz);
    }

    copyConstants();
    double *dev_g_array, *dev_p_array;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_g_array, Ntomo * n_redshift_bins * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_p_array, Ntomo * n_redshift_bins * sizeof(double)));

    double *dev_sigma_epsilon, *dev_ngal;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_sigma_epsilon, Ntomo * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_ngal, Ntomo * sizeof(double)));

    set_cosmology(cosmo, dev_g_array, dev_p_array, &nzs, &sigma_epsilon_per_bin, &ngal_per_bin, dev_sigma_epsilon, dev_ngal);

    for (int i = 0; i < n_combis; i++)
    {
        double theta = convert_angle_to_rad(theta_combis[0][i]); // Conversion to rad
        std::vector<int> zbins_calc = {z_combis[0][i], z_combis[1][i]};

        if (z_combis[0][i] >= Ntomo || z_combis[1][i] >= Ntomo)
        {
            std::cerr << "Issue with tomo bins! You want to access a zbin which is outside the number of tomographic bins!" << std::endl;
            std::cerr << "Note that tomo bins must be numbered as 0,1,2,...!" << std::endl;
            std::cerr << "Exiting." << std::endl;
            exit(1);
        }

        double Map2_value = Map2(theta, zbins_calc, dev_g_array, dev_p_array, Ntomo, dev_sigma_epsilon, dev_ngal); // Do calculation

        std::cerr << i << "/" << n_combis << ": Theta= " << theta_combis[0][i] << " "
                  << ": zbin1= " << z_combis[0][i] << " "
                  << ": zbin2= " << z_combis[1][i] << " "
                  << ": Map2= " << Map2_value << " \r";
        std::cerr.flush();

        if (outputmode == "full")
        {
            out << theta_combis[0][i] << " "
                << z_combis[0][i] << " "
                << z_combis[1][i] << " "
                << Map2_value << std::endl;
        }
        else if (outputmode == "mcmc")
        {
            out << Map2_value << " " << std::endl;
        }
        else
        {
            std::cerr << "Outputmode not specified. Exiting" << std::endl;
            exit(1);
        };
    }

    cudaFree(dev_g_array);
    cudaFree(dev_p_array);
    cudaFree(dev_sigma_epsilon);
    cudaFree(dev_ngal);

    return 0;
}