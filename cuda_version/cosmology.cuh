#ifndef COSMOLOGY_CUH
#define COSMOLOGY_CUH

#include <string>
#include <fstream>
#include <vector>

/**
 * Class containing cosmological parameters
 * Can read them from file
 */
class cosmology
{
public:
  double h; /**< dimensionless Hubble constant*/
  double sigma8; /**< Powerspectrum normalisation \f$\sigma_8\f$*/
  double omb; /**< dimensionless baryon density parameter \f$\Omega_b\f$*/
  double omc; /**< dimensionless density parameter of CDM*/
  double ns; /**< Power spectrum spectral index*/
  double w; /**< Eq. of state of Dark Energy*/
  double om; /**< dimensionless matter density parameter*/
  double ow; /**< dimensionless density parameter of Dark Energy*/
  double zmax; /**< Maximal redshift of simulation */

  cosmology(){}; //Empty constructor

  cosmology(const std::string& fn_parameters); // Constructor from filename (reads in)


};

//Output
std::ostream& operator<<(std::ostream& out, const cosmology& cosmo);





// Read in of n(z) (Assumes linear binning in file!)
void read_n_of_z(const std::string& fn, const double& dz, const int& n_bins, std::vector<double>& nz);


// Read in of thetas (in arcmin!)
void read_thetas(const std::string& fn, std::vector<double>& thetas);


#endif // COSMOLOGY_CUH
