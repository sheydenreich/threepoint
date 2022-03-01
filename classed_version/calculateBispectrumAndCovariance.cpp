#include "apertureStatistics.hpp"
#include "bispectrum.hpp"
#include <fstream>
#include "helper.hpp"

int main(int argc, char * argv[])
{

  // Read in CLI
  const char *message = R"( 
calculateBispectrumAndCovariance.x : Wrong number of command line parameters (Needed: 12)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename with n(z)
Argument 3: Theta_Max [unit], this is the sidelength for a square survey
Argument 4: Outputfolder (needs to exist)
Argument 5: sigma, shapenoise (for both components)
Argument 6: n [unit^-2] Galaxy numberdensity
Argument 7: unit, either arcmin, deg, or rad
)";

  if (argc != 8)
  {
    std::cerr << message << std::endl;
    exit(-1);
  };
  std::string cosmo_paramfile = argv[1]; // Parameter file
  std::string nzfn = argv[2];
  std::string out_folder = argv[3];
  double thetaMax = std::stod(argv[4]);
  double sigma = std::stod(argv[5]);
  double n = std::stod(argv[6]);
  std::string unit = argv[7];

  std::cerr << "Using cosmology from " << cosmo_paramfile << std::endl;
  std::cerr << "Using n(z) from " << nzfn << std::endl;
  std::cerr << "Results are written to " << out_folder << std::endl;

    // Initializations

  double thetaMaxRad = convert_angle_to_rad(thetaMax, unit);
  double nRad = n / convert_angle_to_rad(1, unit) / convert_angle_to_rad(1, unit);
  double survey_area=thetaMaxRad*thetaMaxRad;

  cosmology cosmo(cosmo_paramfile);

  std::vector<double> nz;
  try
  {
    read_n_of_z(nzfn, 100, cosmo.zmax, nz);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return -1;
  }


  BispectrumCalculator bispectrum(&cosmo, nz, 100, cosmo.zmax);
  bispectrum.sigma = sigma;
  bispectrum.n = nRad;

  // double ell_min = 100;
  // double ell_max = 100000;

  // double lell_min = log(ell_min);
  // double lell_max = log(ell_max);

  int n_ell = 30;

  std::vector<double> bispectrum_covariance_array(n_ell * n_ell * n_ell);
  std::vector<double> bispectrum_array(n_ell * n_ell * n_ell);
  std::vector<double> powerspectrum_array(n_ell);
  std::vector<double> bispectrum_array_diag(n_ell);
  std::vector<double> bispectrum_covariance_array_diag(n_ell);

  // double ell_array[30] = {100.0, 126.89610031679221, 161.02620275609394, 204.33597178569417, 259.2943797404667, 329.03445623126674, 
  // 417.53189365604004, 529.8316906283708, 672.3357536499335, 853.1678524172805, 1082.636733874054, 1373.8237958832638, 1743.3288221999874, 2212.21629107045, 2807.2162039411755, 
  // 3562.2478902624443, 4520.35365636024, 5736.152510448682, 7278.953843983146, 9236.708571873865, 11721.022975334794, 14873.521072935118, 18873.918221350996, 23950.26619987486, 
  // 30391.95382313195, 38566.20421163472, 48939.00918477499, 62101.694189156166, 78804.62815669904, 100000.0};

    double ell_array[30] = {  31.6227766,     38.56620421 ,   47.03420342  ,  57.3615251,
    69.95642157  ,  85.31678524  , 104.04983104 ,  126.89610032,
   154.75873546 ,  188.73918221  , 230.1807313  ,  280.72162039,
   342.35979576  , 417.53189366  , 509.20956368 ,  621.01694189,
   757.37391759 ,  923.67085719 , 1126.48169234 , 1373.82379588,
  1675.47491892 , 2043.35971786 , 2492.02115138 , 3039.19538231,
  3706.51291092 , 4520.35365636 , 5512.88978877 , 6723.3575365,
  8199.6082446 , 10000.         };

  for (int i = 0; i < n_ell; i++)
  {
    powerspectrum_array[i] = bispectrum.Pell(ell_array[i]);
  }
  for (int i = 0; i < n_ell; i++)
  {
    double ell1 = ell_array[i];
    for (int j = i; j < n_ell; j++)
    {
      double ell2 = ell_array[j];
      for (int k = j; k < n_ell; k++)
      {
        double ell3 = ell_array[k];
        double temp, temp2;
        if (is_triangle(ell1, ell2, ell3))
        {
          temp = bispectrum.bispectrumCovariance(ell1, ell2, ell3, ell1, ell2, ell3,
                                                 0.13 * ell1, 0.13 * ell2, 0.13 * ell3, 0.13 * ell1, 0.13 * ell2, 0.13 * ell3, survey_area);

          temp2 = bispectrum.bkappa(ell1, ell2, ell3);
        }
        else
        {
          temp = 0;
          temp2 = 0;
        }
        if(i==j && i==k)
        {
          bispectrum_array_diag[i]=temp2;
          bispectrum_covariance_array_diag[i]=temp;
        };
        bispectrum_covariance_array[i * n_ell * n_ell + j * n_ell + k] = temp;
        bispectrum_covariance_array[i * n_ell * n_ell + k * n_ell + j] = temp;
        bispectrum_covariance_array[j * n_ell * n_ell + i * n_ell + k] = temp;
        bispectrum_covariance_array[j * n_ell * n_ell + k * n_ell + i] = temp;
        bispectrum_covariance_array[k * n_ell * n_ell + i * n_ell + j] = temp;
        bispectrum_covariance_array[k * n_ell * n_ell + j * n_ell + i] = temp;

        bispectrum_array[i * n_ell * n_ell + j * n_ell + k] = temp2;
        bispectrum_array[i * n_ell * n_ell + k * n_ell + j] = temp2;
        bispectrum_array[j * n_ell * n_ell + i * n_ell + k] = temp2;
        bispectrum_array[j * n_ell * n_ell + k * n_ell + i] = temp2;
        bispectrum_array[k * n_ell * n_ell + i * n_ell + j] = temp2;
        bispectrum_array[k * n_ell * n_ell + j * n_ell + i] = temp2;
      }
    }
  }


  std::string outfn =out_folder+"/model_bispectrum";
// "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_SLICS/model_bispectrum_2";
  std::ofstream out;
  std::string outfn2 = out_folder+"/model_cov_bispectrum";
  //"/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_SLICS/model_bispectrum_gaussian_covariance_2";
  std::ofstream out2;
  std::string outfn3 = out_folder+"/model_powerspectrum";

  //"/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_SLICS/model_powerspectrum_2";
  std::ofstream out3;

  std::string outfn4 = out_folder+"/model_bispectrum_diag";
  std::string outfn5 = out_folder+"/model_cov_bispectrum_diag";

  std::ofstream out4, out5;

#if TREAT_DEGENERATE_TRIANGLES
  outfn2.append("_approx_degenerate_triangles");
  outfn5.append("_approx_degenerate_triangles");
#endif
  outfn.append(".dat");
  outfn2.append(".dat");
  outfn3.append(".dat");
  outfn4.append(".dat");
  outfn5.append(".dat");

  std::cout << "Writing results to " << outfn2 << std::endl;
  out.open(outfn.c_str());
  out2.open(outfn2.c_str());
  out3.open(outfn3.c_str());
  out4.open(outfn4.c_str());
  out5.open(outfn5.c_str());

  //Print out ==> Should not be parallelized!!!
  for (int i = 0; i < n_ell; i++)
  {
    out3 << ell_array[i] << " " << powerspectrum_array[i] << std::endl;
    out4 << ell_array[i] << " " << bispectrum_array_diag[i] <<std::endl;
    out5 << ell_array[i] << " " <<bispectrum_covariance_array_diag[i] <<std::endl;
    for (int j = 0; j < n_ell; j++)
    {
      for (int k = 0; k < n_ell; k++)
      {
        out << ell_array[i] << " "
            << ell_array[j] << " "
            << ell_array[k] << " "
            << bispectrum_array[i * n_ell * n_ell + j * n_ell + k] << " "
            << std::endl;
        out2 << ell_array[i] << " "
             << ell_array[j] << " "
             << ell_array[k] << " "
             << bispectrum_covariance_array[i * n_ell * n_ell + j * n_ell + k] << " "
             << std::endl;
      };
    };
  };
}