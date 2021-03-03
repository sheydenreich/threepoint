#ifndef APERTURESTATISTICS_HPP
#define APERTURESTATISTICS_HPP

#define CUBATURE true
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
  double theta1_; //!< Aperture radius [rad]
  double theta2_; //!< Aperture radius [rad]
  double theta3_; //!< Aperture radius [rad]

    /*****Integral borders******/
  
  double phiMin=0;//!< Minimal phi [rad]
  double phiMax=6.28318; //!< Maximal phi [rad]
  double lMin=1; //!<Minimal ell
  double lMax=1e4; //!< Maximal ell (Overwritten by 10/min(theta) in inegration)
  
  /**
   * @brief Integrand of MapMapMap
   * Given in Schneider, Kilbinger & Lombardi (2003), Eq. 58
   * \f$ \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3) + \mathrm{2 terms}]\f$
   * @param l1 ell1
   * @param l2 ell2
   * @param phi angle between l1 and l2 [rad]
   * @return value of integrand
   */
  double integrand(const double& l1, const double& l2, const double& phi);

  /**
   * @brief MapMapMap integrand, integrated over phi
   * \f$ \int \mathrm{d}\phi \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3) + \mathrm{2 terms}]\f$
   * @param l1 ell1
   * @param l2 ell2
   * @return value of integrand
   */
  double integral_phi(double l1, double l2);

    /**
   * @brief MapMapMap integrand, integrated over phi and l2
   * \f$ \int \mathrm{d} \ell_2 \int \mathrm{d}\phi \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3) + \mathrm{2 terms}]\f$
   * @param l1 ell1
   * @return value of integrand
   */
  double integral_l2(double l1);

  /**
   * @brief MapMapMap integrand, integrated over phi, l2 and l1
   * \f$ \int \mathrm{d} \ell_1 \int \mathrm{d} \ell_2 \int \mathrm{d}\phi \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3) + \mathrm{2 terms}]\f$
   * @return value of integrand
   */
  double integral_l1();


  /**
   * @brief MapMapMap integrand for phi-integration, formulated for use of GSL integration routine
   * Calls integral_phi(l1_, l2_)
   * @param phi Phi [rad]
   * @param thisPtr Pointer to ApertureStatistics class that is integrated
   */
  static double integrand_phi(double phi, void * thisPtr);

    /**
   * @brief MapMapMap integrand for l2-integration, formulated for use of GSL integration routine
   * Calls integral_l2(l1_)
   * @param l2 ell2
   * @param thisPtr Pointer to ApertureStatistics class that is integrated
   */
  static double integrand_l2(double l2, void * thisPtr);

  /**
   * @brief MapMapMap integrand for l1-integration, formulated for use of GSL integration routine
   * Calls integral_l1()
   * @param l1 ell1
   * @param thisPtr Pointer to ApertureStatistics class that is integrated
   */
  static double integrand_l1(double l1, void* thisPtr);


  /**
   * @brief Integrand for cubature library
   * See https://github.com/stevengj/cubature for documentation
   * @param ndim Number of dimensions of integral (here: 3, automatically assigned by integration)
   * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
   * @param vars Array containing integration variables
   * @param thisPts Pointer to ApertureStatistics Instance that is integrated
   * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
   * @param value Value of integral
   * @return 0 on success
   */
  static int integrand(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value); 
  
public:

  /**
   * @brief Constructor from cosmology
   */
  ApertureStatistics(BispectrumCalculator* Bispectrum);


  /**
   * @brief Aperturestatistics calculated from Bispectrum using Eq. 58 from Schneider, Kilbinger & Lombardi (2003)
   * If CUBATURE is true, this uses the pcubature routine from the cubature library
   * If CUBATURE is false / not defined, this uses GSL and three separate integrals over each dimension (SLOOOOOW AF)
   * @param theta1 Aperture Radius 1 [rad]
   * @param theta2 Aperture Radius 2 [rad]
   * @param theta3 Aperture Radius 3 [rad]
   */
   double MapMapMap(const double& theta1, const double& theta2, const double& theta3);
};













#endif //APERTURESTATISTICS_HPP
