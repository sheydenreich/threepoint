#ifndef COSMOLOGY_HPP
#define COSMOLOGY_HPP

#include <string>
#include <fstream>

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


#endif // COSMOLOGY_HPP