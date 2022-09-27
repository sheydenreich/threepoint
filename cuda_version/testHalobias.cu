#include "apertureStatisticsCovariance.cuh"
#include "cosmology.cuh"
#include "bispectrum.cuh"
#include "helpers.cuh"
#include "cuda_helpers.cuh"
#include "halomodel.cuh"

#include <iostream>
#include <chrono>

/**
 * @file calculateApertureStatisticsCovariance.cpp
 * This executable calculates the covariance of <MapMapMap> as given by the real-space estimator
 * Calculates Terms T1-T7 individually
 * Aperture radii, cosmology, n(z), and survey properties are read from file
 * Model uses Revised Halofit Powerspectrum, BiHalofit Bispectrum, and 1-halo terms for Tri- and Pentaspectrum
 * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 */
int main(int argc, char *argv[])
{
    // Read in CLI
    const char *message = R"( 
calculateApertureStatisticsCovariance.x : Wrong number of command line parameters (Needed: 10)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas [arcmin]
Argument 3: Filename with n(z)
Argument 4: Outputfolder (needs to exist)
Argument 5: Filename for covariance parameters (ASCII, see necessary_files/cov_param for an example)
Argument 6: Survey geometry, either circle, square, infinite, or rectangular
)";

    if (argc != 3)
    {
        std::cerr << message << std::endl;
        exit(-1);
    };

    std::string cosmo_paramfile = argv[1]; // Parameter file
    std::string nzfn = argv[2];

    // Set Cosmology
    cosmology cosmo(cosmo_paramfile);

    // Read in n(z)
    std::vector<double> nz;
    try
    {
        read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }

    set_cosmology(cosmo, &nz);

    std::vector<double> ms{1.00000000e+12, 1.27427499e+12, 1.62377674e+12, 2.06913808e+12,
                           2.63665090e+12, 3.35981829e+12, 4.28133240e+12, 5.45559478e+12,
                           6.95192796e+12, 8.85866790e+12, 1.12883789e+13, 1.43844989e+13,
                           1.83298071e+13, 2.33572147e+13, 2.97635144e+13, 3.79269019e+13,
                           4.83293024e+13, 6.15848211e+13, 7.84759970e+13, 1.00000000e+14};

    initHalomodel();

    for (int i = 0; i < ms.size(); i++)
    {
        std::cout << ms[i] << " " << halo_bias(ms[i], 0) << std::endl;
    };

    return 0;
}
