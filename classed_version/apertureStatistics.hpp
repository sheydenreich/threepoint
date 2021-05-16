#ifndef APERTURESTATISTICS_HPP
#define APERTURESTATISTICS_HPP

#define CUBATURE true
#define INTEGRATE4D true
//Switches for Parallelization
//Make sure at the most one is on!!! Also: NO PARALLELIZATION IF GSL IS USED FOR BISPEC INTEGRATION!!!
#define PARALLEL_INTEGRATION true
#define PARALLEL_RADII false
#include "bispectrum.hpp"



/**
 * @brief Class computing <Map^3> from a bispectrum. 
 * This class takes a matter bispectrum as input and computes the 3pt Aperture Statistics
 */
class ApertureStatistics
{
private:

  /**
   * @brief GSL workspace for l1 integration
   * Is allocated in constructor
   */
  gsl_integration_workspace * w_l1;
  /**
   * @brief GSL workspace for l2 integration
   * Is allocated in constructor
   */
  gsl_integration_workspace * w_l2;
  /**
   * @brief GSL workspace for phi integration
   * Is allocated in constructor
   */
  gsl_integration_workspace * w_phi;



  /**
   * @brief Filter function in Fourier space
   * \f$ \hat{u}(\eta)=\frac{\eta^2}{2}\exp(-\frac{\eta^2}{2}) \f$
   * @param eta Parameter of filter function (=l*theta)
   * @return Value of filter function
   */
  double uHat(const double& eta);

public: //Once debugging is finished, these members should be private!


  /**
   * @brief Bispectrum for which MapMapMap is calculated 
   */
  BispectrumCalculator* Bispectrum_;
  
    /*****Temporary variables for integrations****/
  
  double l1_; //!<Temporary ell1
  double l2_; //!<Temporary ell2

    /*****Integral borders******/
  
  double phiMin=0;//!< Minimal phi [rad]
  double phiMax=6.28318; //!< Maximal phi [rad]
  double lMin=1; //!<Minimal ell
  double lMax=1e4; //!< Maximal ell (Overwritten by 10/min(theta) in inegration)
  
  /**
   * @brief Integrand of MapMapMap
   * @warning This is different from Eq 58 in Schneider, Kilbinger & Lombardi (2003) because the Bispectrum is defined differently!
   * \f$ \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3)]\f$
   * @param l1 ell1
   * @param l2 ell2
   * @param phi angle between l1 and l2 [rad]
   * @param thetas Aperture radii [rad]
   * @return value of integrand
   */
  double integrand(const double& l1, const double& l2, const double& phi, double* thetas);

  /**
   * @brief 4d- Integrand of MapMapMap (combining limber-integration with the integrand function)
   * @warning This is different from Eq 58 in Schneider, Kilbinger & Lombardi (2003) because the Bispectrum is defined differently!
   * \f$ \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3)]\f$
   * @param l1 ell1
   * @param l2 ell2
   * @param phi angle between l1 and l2 [rad]
   * @param z redshift
   * @param thetas Aperture radii [rad]
   * @return value of integrand
   */
  double integrand_4d(const double& l1, const double& l2, const double& phi, const double& z, double* thetas);


  /**
   * @brief MapMapMap integrand, integrated over phi
   * \f$ \int \mathrm{d}\phi \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3) + \mathrm{2 terms}]\f$
   * @param l1 ell1
   * @param l2 ell2
   * @param thetas Aperture radii [rad]
   * @return value of integrand
   */
  double integral_phi(double l1, double l2, double* thetas);

    /**
   * @brief MapMapMap integrand, integrated over phi and l2
   * \f$ \int \mathrm{d} \ell_2 \int \mathrm{d}\phi \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3) + \mathrm{2 terms}]\f$
   * @param l1 ell1
   * @param thetas Aperture radii [rad]
   * @return value of integrand
   */
  double integral_l2(double l1, double* thetas);

  /**
   * @brief MapMapMap integrand, integrated over phi, l2 and l1
   * \f$ \int \mathrm{d} \ell_1 \int \mathrm{d} \ell_2 \int \mathrm{d}\phi \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3) + \mathrm{2 terms}]\f$
   * @param thetas Aperture radii [rad]
   * @return value of integrand
   */
  double integral_l1(double* thetas);


  /**
   * @brief MapMapMap integrand for phi-integration, formulated for use of GSL integration routine
   * Calls integral_phi(l1_, l2_)
   * @param phi Phi [rad]
   * @param thisPtr Pointer to ApertureStatisticsContainer that is integrated
   */
  static double integrand_phi(double phi, void * thisPtr);

    /**
   * @brief MapMapMap integrand for l2-integration, formulated for use of GSL integration routine
   * Calls integral_l2(l1_)
   * @param l2 ell2
   * @param thisPtr Pointer to ApertureStatisticsContainer that is integrated
   */
  static double integrand_l2(double l2, void * thisPtr);

  /**
   * @brief MapMapMap integrand for l1-integration, formulated for use of GSL integration routine
   * Calls integral_l1()
   * @param l1 ell1
   * @param thisPtr Pointer to ApertureStatisticsCOntainer that is integrated
   */
  static double integrand_l1(double l1, void* thisPtr);


  /**
   * @brief Integrand for cubature library
   * See https://github.com/stevengj/cubature for documentation
   * @param ndim Number of dimensions of integral (here: 3, automatically assigned by integration)
   * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
   * @param vars Array containing integration variables
   * @param thisPts Pointer to ApertureStatisticsContainer instance
   * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
   * @param value Value of integral
   * @return 0 on success
   */
  static int integrand(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value); 

  /**
   * @brief Integrand for cubature library, same as the integrand-function, but the limber-integration is also performed via cubature
   * See https://github.com/stevengj/cubature for documentation
   * @param ndim Number of dimensions of integral (here: 3, automatically assigned by integration)
   * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
   * @param vars Array containing integration variables
   * @param thisPts Pointer to ApertureStatisticsContainer instance
   * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
   * @param value Value of integral
   * @return 0 on success
   */
  static int integrand_4d(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value); 

  
public:

  /**
   * @brief Constructor from cosmology
   */
  ApertureStatistics(BispectrumCalculator* Bispectrum);


  /**
   * @brief Aperturestatistics calculated from Bispectrum
   * @warning This is NOT Eq 58 from Schneider, Kilbinger & Lombardi (2003), but a third of that, due to the bispectrum definition
   * If CUBATURE is true, this uses the pcubature routine from the cubature library
   * If CUBATURE is false / not defined, this uses GSL and three separate integrals over each dimension (SLOOOOOW AF)
   * @param theta1 Aperture Radius 1 [rad]
   * @param theta2 Aperture Radius 2 [rad]
   * @param theta3 Aperture Radius 3 [rad]
   */
   double MapMapMap(double* thetas);
};



/**
 * @brief This struct contains an instance of the aperture statistics and aperture radii
 * It is needed for the integration with cubature
 */
struct ApertureStatisticsContainer
{
  /** Aperturestatistics to be calculated*/
  ApertureStatistics* aperturestatistics;

  /** Apertureradii [rad]*/
  double* thetas;
};








#endif //APERTURESTATISTICS_HPP
