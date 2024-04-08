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
 * Takahashi+ Bispectrum for different tomographic bins
 * Aperture radii are read from file
 * Tomo bins are read from file
 * Code uses CUDA and cubature library  (See
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

    if (argc < 6) // Give out error message if too few CLI arguments
    {
        std::cerr << message << std::endl;
        exit(1);
    };

    std::string cosmo_paramfile = argv[1];
    std::string R_file = argv[2];
    std::string output_file = argv[3];
    std::string nz_lenses = argv[4];
    std::string nz_sources = argv[5];
    double b = std::stod(argv[6]);

    std::vector<std::string> nzfns;
    nzfns.push_back(nz_lenses);
    nzfns.push_back(nz_sources);

    // Check if output file can be opened
    std::ofstream out;
    out.open(output_file.c_str());
    if (!out.is_open())
    {
        std::cerr << "Couldn't open " << output_file << std::endl;
        exit(1);
    };

    std::vector<double> Rs;
    read_thetas(R_file, Rs);

    // Read in cosmology
    cosmology cosmo(cosmo_paramfile);

    // Read in n_z
    std::vector<std::vector<double>> nzs;
    for (int i = 0; i < 2; i++)
    {
        std::vector<double> nz;
        read_n_of_z(nzfns.at(i), n_redshift_bins, cosmo.zmax, nz);
        nzs.push_back(nz);
    }

    copyConstants();
    double *dev_g_array, *dev_p_array;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_g_array, 2 * n_redshift_bins * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_p_array, 2 * n_redshift_bins * sizeof(double)));

    set_cosmology(cosmo, dev_g_array, dev_p_array, &nzs);

    for (int i = 0; i < Rs.size(); i++)
    {
        double NNM = NNM_IA(Rs.at(i), b, dev_p_array);
        std::cerr << i << "/" << Rs.size() << ": R=" << Rs.at(i) << " "
                  << "NNM=" << NNM << std::endl; //" \r";
        //std::cerr.flush();

        out << Rs.at(i) << " " << NNM << std::endl;
    };
    cudaFree(dev_g_array);
    cudaFree(dev_p_array);

    return 0;
}