/**
 * @file testHalomodel.cu
 * @author Laila Linke
 * @brief Files for testing halomodel implementation, in particular u_NFW and hmf
 * 
 */

 #include "halomodel.cuh"
 #include "cosmology.cuh"
 #include "bispectrum.cuh"
 #include "helpers.cuh"
 #include <iostream>
 #include <vector>

 int main(int argc, char* argv[])
 {
  // Read in CLI
  const char *message = R"( 
    testHalomodel.x : Wrong number of command line parameters (Needed: 3)
    Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
    Argument 2: Filename with n(z)
    Argument 3: Outputfolder (needs to exist)
    )";

    if(argc != 4)
    {
        std::cerr<<message<<std::endl;
        exit(-1);
    };

    std::string cosmo_paramfile = argv[1]; // Parameter file
    std::string nzfn = argv[2];
    std::string out_folder = argv[3];


    std::cerr << "Using cosmology from " << cosmo_paramfile << std::endl;
    std::cerr << "Using n(z) from " << nzfn << std::endl;
    std::cerr << "Results are written to " << out_folder << std::endl;

    // Initializations

    cosmology cosmo(cosmo_paramfile);

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

    // Calculate u_NFW for a certain halo mass and redshift
    {
        double m=1e14; //[h^-1 Msun]
        double z=0.2;

        double kmin=1e-2;
        double kmax=1e2;
        int Nbins=128;
        double deltaK=log(kmax/kmin)/Nbins;

        std::ofstream out;
        out.open(out_folder+"/u_nfw_threepointCode.dat");

        for(int i=0; i<Nbins; i++)
        {
            double k=kmin*exp(i*deltaK);
            out<<k<<" "<<u_NFW(k, m, z)<<std::endl;
        };
    }

    // Calculate HMF for a certain redshift

    {
        double z=0.2;

        std::ofstream out;
        out.open(out_folder+"/hmf_threepointCode.dat");

        double mmin=1e10; //[h^-1 Msun]
        double mmax=1e16; //[h^-1 Msun]
        int Nbins=128;
        double deltaM=log(mmax/mmin)/Nbins;

        for(int i=0; i<Nbins; i++)
        {
            double m=mmin*exp(i*deltaM);
            out<<m<<" "<<hmf(m, z)<<std::endl;
        };
    }
 }