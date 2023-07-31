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
    initHalomodel();


    double mmin=1e9;
    double mmax=1e17;
    //double z=0.1;
    double zmin=0.001;
    double zmax=3;

    std::vector<double> ks{  1.00000000e+00, 1.46779927e+00, 2.15443469e+00, 3.16227766e+00,
       4.64158883e+00, 6.81292069e+00, 1.00000000e+01, 1.46779927e+01,
       2.15443469e+01, 3.16227766e+01, 4.64158883e+01, 6.81292069e+01,
       1.00000000e+02, 1.46779927e+02, 2.15443469e+02, 3.16227766e+02,
       4.64158883e+02, 6.81292069e+02, 1.00000000e+03, 1.46779927e+03,
       2.15443469e+03, 3.16227766e+03, 4.64158883e+03, 6.81292069e+03,
       1.00000000e+04   };


    for (int i=0; i<ks.size(); i++)
    {
        double k=ks[i];
        std::cout<<k<<" "
        <<trispectrum_1halo(zmin, zmax, mmin, mmax, k, k, k, k)<<" "
        <<trispectrum_2halo(zmin, zmax, mmin, mmax, k, k, k, k, k, k, k, k)<<" "
        <<std::endl;
    }


    return 0;
}
