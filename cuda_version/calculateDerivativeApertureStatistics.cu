#include "apertureStatistics.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono> //For time measurements

/**
 * @file calculateDerivativeApertureStatistics.cpp
 * This executable calculates the derivative of <MapMapMap> wrt to
 * the cosmological parameters \f$h$\f, \f$\sigma_8$\f, \f$\Omega_b$\f, 
 * \f$n_s$\f, \f$w$\f, \f$\Omega_m$\f, and \f$\Omega_\Lambda$\f 
 * from the Takahashi+ Bispectrum
 * Uses either 3 or 5-point stencil with stepsize h
 * Aperture radii and cosmology are read from file
 * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
 * @author Laila Linke
 */
int main(int argc, char *argv[])
{
  // Read in command line

  const char *message = R"( 
calculateDerivativeApertureStatistics.x : Wrong number of command line parameters (Needed: 7)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with thetas [arcmin]
Argument 3: Outputfilename, directory needs to exist 
Argument 4: 0: use three-point stencil, 1: use five-point stencil
Argument 5: Stencil stepsize
Argument 6: 0: use analytic n(z) (only works for MR and SLICS), or 1: use n(z) from file                  
Argument 7 (optional): Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)

Example:
./calculateDerivativeApertureStatistics.x ../necessary_files/MR_cosmo.dat ../necessary_files/HOWLS_thetas.dat ../../results_MR/MapMapMap_derivatives.dat 1 0.01 1 ../necessary_files/nz_MR.dat
)";

  if (argc < 7)
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile, thetasfn, outfn, nzfn;
  bool nz_from_file = false;
  bool five_point = false;
  double h;

  cosmo_paramfile = argv[1];
  thetasfn = argv[2];
  outfn = argv[3];
  five_point = std::stoi(argv[4]);
  h = std::stod(argv[5]);
  nz_from_file = std::stoi(argv[6]);
  if (nz_from_file)
  {
    nzfn = argv[7];
  };

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);                   ///<cosmology at which derivative is calculated
  double dz = cosmo.zmax / ((double)n_redshift_bins); //redshift binsize

  // Read in n_z
  std::vector<double> nz;
  if (nz_from_file)
  {
    read_n_of_z(nzfn, dz, n_redshift_bins, nz);
  };

  // Check if output file can be opened
  std::ofstream out;
  out.open(outfn.c_str());
  if (!out.is_open())
  {
    std::cerr << "Couldn't open " << outfn << std::endl;
    exit(1);
  };

  // Read in thetas
  std::vector<double> thetas;
  read_thetas(thetasfn, thetas);
  int N = thetas.size();

  // User output
  std::cerr << "Using cosmology from " << cosmo_paramfile << ":" << std::endl;
  std::cerr << cosmo;
  std::cerr << "Using thetas in " << thetasfn << std::endl;
  std::cerr << "Writing to:" << outfn << std::endl;
  if (five_point)
  {
    std::cerr << "Using five-point stencil" << std::endl;
  }
  else
  {
    std::cerr << "Using three-point stencil" << std::endl;
  };

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A96, &A96, 48 * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_W96, &W96, 48 * sizeof(double)));

  // Borders of integral
  double phiMin = 0.0;
  double phiMax = 6.28319;
  double lMin = 1;

  // Set up cosmologies at which Map^3 is calculated
  // This can probably be done smarter

  std::vector<cosmology> cosmos;             ///<container for all cosmologies
  std::vector<double> derivative_parameters; //parameters the derivatives are taken in (Why do we need this?)

  cosmology newCosmo = cosmo;

  // Set h
  if (five_point)
  {
    newCosmo.h = cosmo.h * (1. - 2 * h);
    cosmos.push_back(newCosmo);
  };
  newCosmo.h = cosmo.h * (1. - h);
  cosmos.push_back(newCosmo);
  newCosmo.h = cosmo.h * (1. + h);
  cosmos.push_back(newCosmo);
  if (five_point)
  {
    newCosmo.h = cosmo.h * (1. + 2 * h);
    cosmos.push_back(newCosmo);
  };
  derivative_parameters.push_back(cosmo.h);

  // Set sigma_8
  newCosmo = cosmo;
  if (five_point)
  {
    newCosmo.sigma8 = cosmo.sigma8 * (1. - 2 * h);
    cosmos.push_back(newCosmo);
  };
  newCosmo.sigma8 = cosmo.sigma8 * (1. - h);
  cosmos.push_back(newCosmo);
  newCosmo.sigma8 = cosmo.sigma8 * (1. + h);
  cosmos.push_back(newCosmo);
  if (five_point)
  {
    newCosmo.sigma8 = cosmo.sigma8 * (1. + 2 * h);
    cosmos.push_back(newCosmo);
  };
  derivative_parameters.push_back(cosmo.sigma8);

  // Set Omega_b (leaves Omega_m constant, changes Omega_cdm appropriately)
  newCosmo = cosmo;
  if (five_point)
  {
    newCosmo.omb = cosmo.omb * (1 - 2 * h);
    newCosmo.omc = newCosmo.om - newCosmo.omb;
    cosmos.push_back(newCosmo);
  };
  newCosmo.omb = cosmo.omb * (1 - h);
  newCosmo.omc = newCosmo.om - newCosmo.omb;
  cosmos.push_back(newCosmo);
  newCosmo.omb = cosmo.omb * (1 + h);
  newCosmo.omc = newCosmo.om - newCosmo.omb;
  cosmos.push_back(newCosmo);
  if (five_point)
  {
    newCosmo.omb = cosmo.omb * (1 + 2 * h);
    newCosmo.omc = newCosmo.om - newCosmo.omb;
    cosmos.push_back(newCosmo);
  };
  derivative_parameters.push_back(cosmo.omb);

  // Set n_s
  newCosmo = cosmo;
  if (five_point)
  {
    newCosmo.ns = cosmo.ns * (1 - 2 * h);
    cosmos.push_back(newCosmo);
  };
  newCosmo.ns = cosmo.ns * (1 - h);
  cosmos.push_back(newCosmo);
  newCosmo.ns = cosmo.ns * (1 + h);
  cosmos.push_back(newCosmo);
  if (five_point)
  {
    newCosmo.ns = cosmo.ns * (1 + 2 * h);
    cosmos.push_back(newCosmo);
  };
  derivative_parameters.push_back(cosmo.ns);

  // Set w (here h--> -h because w<0)
  newCosmo = cosmo;
  if (five_point)
  {
    newCosmo.w = cosmo.w * (1 + 2 * h);
    cosmos.push_back(newCosmo);
  };
  newCosmo.w = cosmo.w * (1 + h);
  cosmos.push_back(newCosmo);
  newCosmo.w = cosmo.w * (1 - h);
  cosmos.push_back(newCosmo);
  if (five_point)
  {
    newCosmo.w = cosmo.w * (1 - 2 * h);
    cosmos.push_back(newCosmo);
  };
  derivative_parameters.push_back(cosmo.w);

  // Set Omega_m (changes Omega_Lambda to preserve flat Universe, leaves Omega_b constant, changes Omega_cdm appropriately)
  newCosmo = cosmo;
  if (five_point)
  {
    newCosmo.om = cosmo.om * (1 - 2 * h);
    newCosmo.omc = newCosmo.om - newCosmo.omb;
    newCosmo.ow = 1 - newCosmo.om;
    cosmos.push_back(newCosmo);
  };
  newCosmo.om = cosmo.om * (1 - h);
  newCosmo.omc = newCosmo.om - newCosmo.omb;
  newCosmo.ow = 1 - newCosmo.om;
  cosmos.push_back(newCosmo);
  newCosmo.om = cosmo.om * (1 + h);
  newCosmo.omc = newCosmo.om - newCosmo.omb;
  newCosmo.ow = 1 - newCosmo.om;
  cosmos.push_back(newCosmo);
  if (five_point)
  {
    newCosmo.om = cosmo.om * (1 + 2 * h);
    newCosmo.omc = newCosmo.om - newCosmo.omb;
    newCosmo.ow = 1 - newCosmo.om;
    cosmos.push_back(newCosmo);
  };
  derivative_parameters.push_back(cosmo.om);

  // Set Omega_Lambda (changes Omega_m to preserve flat Universe, leaves Omega_b constant, changes Omega_cdm appropriately )
  newCosmo = cosmo;
  if (five_point)
  {
    newCosmo.ow = cosmo.ow * (1 - 2 * h);
    newCosmo.om = 1 - newCosmo.ow;
    newCosmo.omc = newCosmo.om-newCosmo.omb;
    cosmos.push_back(newCosmo);
  };

  newCosmo.ow = cosmo.ow * (1 - h);
  newCosmo.om = 1 - newCosmo.ow;
  newCosmo.omc = newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  newCosmo.ow = cosmo.ow * (1 + h);
  newCosmo.om = 1 - newCosmo.ow;
  newCosmo.omc = newCosmo.om-newCosmo.omb;
  cosmos.push_back(newCosmo);
  if (five_point)
  {
    newCosmo.ow = cosmo.ow * (1 + 2 * h);
    newCosmo.om = 1 - newCosmo.ow;
    newCosmo.omc = newCosmo.om-newCosmo.omb;
    cosmos.push_back(newCosmo);
  };
  derivative_parameters.push_back(cosmo.ow);

  int Ncosmos = cosmos.size(); ///<Number of cosmologies

  // Calculation of Map^3
  int Ntotal = N * (N + 1) * (N + 2) / 6.;     //Total number of bins that need to be calculated, = (N+3-1) ncr 3
  std::vector<std::vector<double>> MapMapMaps; //<Array which will contain MapMapMap calculated
  for (int i = 0; i < Ncosmos; i++)
  {
    std::cout << "Doing calculations for cosmology " << i << " of " << Ncosmos << std::endl;
    auto begin = std::chrono::high_resolution_clock::now(); //Begin time measurement
    // Initialize Bispectrum
    if (nz_from_file)
    {
      set_cosmology(cosmos[i], dz, nz_from_file, &nz);
    }
    else
    {
      set_cosmology(cosmos[i], dz);
    };

    //Needed for monitoring
    int step = 0;

    std::vector<double> MapMapMap_thiscosmo;

    //Calculate <MapMapMap>(theta1, theta2, theta3) in three loops
    // Calculation only for theta1<=theta2<=theta3, other combinations are assigned
    for (int j = 0; j < N; j++)
    {
      double theta1 = thetas.at(j) * 3.1416 / 180. / 60; //Conversion to rad
      for (int k = j; k < N; k++)
      {
        double theta2 = thetas.at(k) * 3.1416 / 180. / 60.;
        for (int l = k; l < N; l++)
        {
          double theta3 = thetas.at(l) * 3.1416 / 180. / 60.;
          double thetas_calc[3] = {theta1, theta2, theta3};
          //Progress for the impatient user (Thetas in arcmin)
          step += 1;
          std::cout << step << "/" << Ntotal << ": Thetas:" << thetas.at(j) << " " << thetas.at(k) << " " << thetas.at(l) << " \r"; //\r is so that only one line is shown
          std::cout.flush();

          double Map3 = MapMapMap(thetas_calc, phiMin, phiMax, lMin); //Do calculation

          MapMapMap_thiscosmo.push_back(Map3);
        };
      };
    };

    MapMapMaps.push_back(MapMapMap_thiscosmo);

    // Stop measuring time and calculate the elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    std::cout << "Time needed for last cosmology:" << elapsed.count() * 1e-9 << std::endl;
  };

  // Calculation of Derivatives
  int Nderivs = int(Ncosmos / 2);
  if (five_point)
    Nderivs = int(Ncosmos / 4);                 ///<Number of derivatives
  double derivs_MapMapMaps[Nderivs][Ntotal]; ///<Array which will contain MapMapMap calculated

#pragma omp parallel for collapse(2)
  for (int i = 0; i < Nderivs; i++)
  {
    for (int j = 0; j < Ntotal; j++)
    {
      if (five_point)
      {
        // Stencil calculation: df/dx = [f(x-2h)-8f(x-h)+8f(x+h)-f(x+2h)]/(12h)
        derivs_MapMapMaps[i][j] = (MapMapMaps[4 * i][j] - 8 * MapMapMaps[4 * i + 1][j] + 8 * MapMapMaps[4 * i + 2][j] - MapMapMaps[4 * i + 3][j]) / (12. * h * derivative_parameters.at(i));
      }
      else
      {
        derivs_MapMapMaps[i][j] = (MapMapMaps[2 * i + 1][j] - MapMapMaps[2 * i][j]) / 2 / h / derivative_parameters.at(i);
      }
    }
  }

  // Output (Cannot be parallelized!!)
  std::cout << "Writing results to " << outfn << std::endl;
  for (int i = 0; i < Nderivs; i++)
  {
    for (int j = 0; j < Ntotal; j++)
    {
      out << derivs_MapMapMaps[i][j] << " ";
    }
    out << std::endl;
  }
}
