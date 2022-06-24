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
  double h;      /**< dimensionless Hubble constant*/
  double sigma8; /**< Powerspectrum normalisation \f$\sigma_8\f$*/
  double omb;    /**< dimensionless baryon density parameter \f$\Omega_b\f$*/
  double omc;    /**< dimensionless density parameter of CDM*/
  double ns;     /**< Power spectrum spectral index*/
  double w;      /**< Eq. of state of Dark Energy*/
  double om;     /**< dimensionless matter density parameter*/
  double ow;     /**< dimensionless density parameter of Dark Energy*/
  double zmax;   /**< Maximal redshift of simulation */

  cosmology(){}; // Empty constructor

  cosmology(const std::string &fn_parameters); // Constructor from filename (reads in)
};

// Output
std::ostream &operator<<(std::ostream &out, const cosmology &cosmo);

class covarianceParameters
{
public:
  double thetaMax; //[rad] this is the radius for a circular survey and the sidelength for a square survey
  double shapenoise_sigma;
  double galaxy_density; // rad^-2
  bool shapenoiseOnly;
  double thetaMax_smaller; // [rad]
  double area;

  covarianceParameters(){};                    // Empty constructor
  covarianceParameters(const std::string &fn); // Constructor from filename (reads in)
};

// Output
std::ostream &operator<<(std::ostream &out, const covarianceParameters &covPar);

struct configGamma
{
  int rsteps, usteps, vsteps;
  double umin, umax, vmin, vmax, rmin, rmax;
};

void read_gamma_config(const std::string &fn, configGamma &config);

std::ostream &operator<<(std::ostream &out, const configGamma &config);

#endif // COSMOLOGY_CUH
