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
 * @author Laila Linke
 */
int main(int argc, char *argv[]) {
  // Read in command line

  const char *message = R"( 
calculateBispectrum.x : Wrong number of command line parameters (Needed: 5)
Argument 1: Filename for cosmological parameters (ASCII, see necessary_files/MR_cosmo.dat for an example)
Argument 2: Filename for n(z) (ASCII, see necessary_files/nz_MR.dat for an example)
Argument 3: Configuration of triangles: equilateral, flattened, squeezed_[sidelength], all
Argument 4: 0: compute at the center of bin, 1: Average over triangles in bin
Argument 5: Output filename (directory needs to exist)

WARNING: Ell-scales are hard-coded right now.
)";

  if (argc < 5) // Give out error message if too few CLI arguments
  {
    std::cerr << message << std::endl;
    exit(1);
  };

  std::string cosmo_paramfile, outfn, nzfn,configuration;
  bool average;
  
  double ell_array[30] = {100.0, 126.89610031679221, 161.02620275609394, 204.33597178569417, 259.2943797404667, 329.03445623126674, 
    417.53189365604004, 529.8316906283708, 672.3357536499335, 853.1678524172805, 1082.636733874054, 1373.8237958832638, 1743.3288221999874, 2212.21629107045, 2807.2162039411755, 
    3562.2478902624443, 4520.35365636024, 5736.152510448682, 7278.953843983146, 9236.708571873865, 11721.022975334794, 14873.521072935118, 18873.918221350996, 23950.26619987486, 
    30391.95382313195, 38566.20421163472, 48939.00918477499, 62101.694189156166, 78804.62815669904, 100000.0};
  int n_ell = 30;
  
  int npts;
  if(configuration=="all") npts = n_ell*(n_ell+1)*(n_ell+2)/6;
  else npts = n_ell;

  double* result = new double[npts];

  cosmo_paramfile = argv[1];
  nzfn = argv[2];
  configuration = argv[3];
  average = std::stoi(argv[4]);
  outfn = argv[5];

  // Read in cosmology
  cosmology cosmo(cosmo_paramfile);

  // Read in n_z
  std::vector<double> nz;
  read_n_of_z(nzfn, n_redshift_bins, cosmo.zmax, nz);

  // Check if output file can be opened
  std::ofstream out;
  out.open(outfn.c_str());
  if (!out.is_open()) {
    std::cerr << "Couldn't open " << outfn << std::endl;
    exit(1);
  };

  // User output
  std::cout << "Using cosmology from " << cosmo_paramfile << ":" << std::endl;
  std::cout << cosmo;
  std::cout << "Writing to:" << outfn << std::endl;

  // Initialize Bispectrum

  copyConstants();

  std::cout << "Using n(z) from " << nzfn << std::endl;
  set_cosmology(cosmo, &nz);

  calculate_bkappa_array(ell_array,configuration,n_ell,average,result);
  // Output

  if(configuration=="all")
  {
    int step = 0;
    for (int i = 0; i < n_ell; i++) {
      for (int j = i; j < n_ell; j++) {
        for (int k = j; k < n_ell; k++) {
          out << ell_array[i] << " " << ell_array[j] << " " << ell_array[k] << " "
              << result[step] << " " << std::endl;
          step++;
        };
      };
    };
  }
  else
  {
    for (int i = 0; i < npts; i++)
    {
      out << ell_array[i] << " " << ell_array[i] << " " << ell_array[i] << " "
      << result[i] << " " << std::endl;
    }
  }

  return 0;
}
