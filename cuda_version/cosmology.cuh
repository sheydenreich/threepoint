#ifndef COSMOLOGY_CUH
#define COSMOLOGY_CUH

#include <string>
#include <fstream>
#include <vector>


/**
 * @brief Class containing cosmological parameters
 * Class can read the parameters from a file
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
  double A_IA;   /**< Intrinsic alignment amplitude*/

  cosmology(){}; // Empty constructor

  cosmology(const std::string &fn_parameters); // Constructor from filename (reads in)
};

/**
 * @brief Operator to write cosmological parameter file
 * 
 * @param out Output stream
 * @param cosmo Cosmology to be written to file
 * @return std::ostream& 
 */
std::ostream &operator<<(std::ostream &out, const cosmology &cosmo);

/**
 * @brief Class containing survey parameters that are important for the covariance
 * 
 */
class covarianceParameters
{
public:
  double thetaMax; //[rad] this is the radius for a circular survey and the sidelength for a square survey
  double shapenoise_sigma; // Galaxy ellipticity standard deviation (for single component)
  double galaxy_density; // number density of galaxies rad^-2
  bool shapenoiseOnly; // Whether a constant powerspectrum is used
  double thetaMax_smaller; // smaller sidelength for a rectangular survey [rad]
  double area; // Survey area [rad^2]

  covarianceParameters(){};                    // Empty constructor
  covarianceParameters(const std::string &fn); // Constructor from filename (reads in)
};

/**
 * @brief Operator to write covariance parameter file
 * 
 * @param out Output stream
 * @param covPar covariance parameters to be written to file
 * @return std::ostream& 
 */
std::ostream &operator<<(std::ostream &out, const covarianceParameters &covPar);




#endif // COSMOLOGY_CUH
