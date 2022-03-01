// #include "apertureStatisticsCovariance.cuh"
// #include "cosmology.cuh"
// #include "bispectrum.cuh"
// #include "helpers.cuh"
// #include "cuda_helpers.cuh"

// #include <iostream>
// #include <chrono>

// int main(int argc, char *argv[])
// {
//   // Read in CLI
//   const char *message = R"( 
// calculateApertureStatisticsSSC.x : Wrong number of command line parameters (Needed: 12)
// Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
// Argument 2: Filename with thetas [arcmin]
// Argument 3: Filename with n(z)
// Argument 4: Outputfolder (needs to exist)
// Argument 5: Filename for covariance parameters (ASCII, see necessary_files/cov_param for an example)
// )";

//   if (argc != 6)
//   {
//     std::cerr << message << std::endl;
//     exit(-1);
//   };


//   std::string cosmo_paramfile = argv[1]; // Parameter file
//   std::string thetasfn = argv[2];
//   std::string nzfn = argv[3];
//   std::string out_folder = argv[4];
//   std::string covariance_paramfile = argv[5];


//   std::cerr << "Using cosmology from " << cosmo_paramfile << std::endl;
//   std::cerr << "Using thetas from " << thetasfn << std::endl;
//   std::cerr << "Using n(z) from " << nzfn << std::endl;
//   std::cerr << "Results are written to " << out_folder << std::endl;
//   std::cerr << "Using covariance parameters from" <<covariance_paramfile<<std::endl;


//   // Initializations
//   covarianceParameters covPar(covariance_paramfile);
//   constant_powerspectrum=covPar.shapenoiseOnly;

//   if(constant_powerspectrum)
//   {
//   std::cerr << "WARNING: Uses constant powerspectrum" << std::endl;
//   };  

//   thetaMax = covPar.thetaMax;
//   sigma = covPar.shapenoise_sigma;
//   n = covPar.galaxy_density;
//   lMin = 2*M_PI/thetaMax;
//   type=1;

//   cosmology cosmo(cosmo_paramfile);


//   std::vector<double> nz;
//   try
//   {
//     read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);
//   }
//   catch (const std::exception &e)
//   {
//     std::cerr << e.what() << '\n';
//     return -1;
//   }

//   set_cosmology(cosmo, &nz);

//   std::vector<double> thetas;

//   try
//   {
//     read_thetas(thetasfn, thetas);
//   }
//   catch (const std::exception &e)
//   {
//     std::cerr << e.what() << '\n';
//     return -1;
//   }

//   // Initialize Covariance
//   initSSC();

//   // Set up output
//   std::string filename="bispec_cov_ssc.dat";
//   std::ofstream out(out_folder+filename);
   
 
//   // Set up ells
//   double ellMin=1e2;
//   double ellMax=1e4;
//   int nBins=32;
//   double deltaEll=(log(ellMax)-log(ellMin))/nBins;
//   // Do calcs
//   for (int i=0; i<nBins; i++)
//   {
//     double ell=exp(log(ellMin)+i*deltaEll);
//     double cov=Cov_Bispec_SSC(ell);
//     out<<ell<<" "<<cov<<std::endl;
//   };

//   return 0;
//  }

 