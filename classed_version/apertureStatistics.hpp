#ifndef APERTURESTATISTICS_HPP
#define APERTURESTATISTICS_HPP

#define CUBATURE true
#define INTEGRATE4D true
// Switches for Parallelization
// Make sure at the most one is on!!! Also: NO PARALLELIZATION IF GSL IS USED FOR BISPEC INTEGRATION!!!
#define PARALLEL_INTEGRATION true
#define PARALLEL_RADII false
#include "bispectrum.hpp"

#define DO_CYCLIC_PERMUTATIONS false

#define CIRCULAR_SURVEY false

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
  gsl_integration_workspace *w_l1;
  /**
   * @brief GSL workspace for l2 integration
   * Is allocated in constructor
   */
  gsl_integration_workspace *w_l2;
  /**
   * @brief GSL workspace for phi integration
   * Is allocated in constructor
   */
  gsl_integration_workspace *w_phi;

  /**
   * @brief Product of three Filter functions in Fourier space
   * \f$ \hat{u}(\ell_1\thetas[0])*\hat{u}(\ell_2\thetas[1])*\hat{u}(\ell_3\thetas[2]) \f$
   * @param l1 First ell-scale
   * @param l2 Second ell-scale
   * @param l3 Third ell-scale
   * @param thetas filter radii[rad]
   * @return Value of product of filter functions
   */
  double uHat_product(const double &l1, const double &l2, const double &l3, std::vector<double> thetas);

#if DO_CYCLIC_PERMUTATIONS
  double _uHat_product(const double &l1, const double &l2, const double &l3, double *thetas);
#endif

  /**
   * @brief Sum of all possible Permutations of the product of three Filter functions in Fourier space
   * @param l1 First ell-scale
   * @param l2 Second ell-scale
   * @param l3 Third ell-scale
   * @param thetas filter radii[rad]
   * @return Value of sum of permutations
   */
  double uHat_product_permutations(const double &l1, const double &l2, const double &l3, std::vector<double> thetas);

public: // Once debugging is finished, these members should be private!
  /**
   * @brief Bispectrum for which MapMapMap is calculated
   */
  BispectrumCalculator *Bispectrum_;

  /*****Temporary variables for integrations****/

  double l1_; //!< Temporary ell1
  double l2_; //!< Temporary ell2

  /*****Integral borders******/

  double phiMin = 0;        //!< Minimal phi [rad]
  double phiMax = 2 * M_PI; //!< Maximal phi [rad]
  double lMin = 1;          //!< Minimal ell
  double lMax = 1e4;        //!< Maximal ell (Overwritten by 10/min(theta) in inegration)

  /**
   * @brief Filter function in Fourier space
   * \f$ \hat{u}(\eta)=\frac{\eta^2}{2}\exp(-\frac{\eta^2}{2}) \f$
   * @param eta Parameter of filter function (=l*theta)
   * @return Value of filter function
   */
  double uHat(const double &eta);

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
  double integrand(const double &l1, const double &l2, const double &phi, std::vector<double> thetas);

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
  double integrand_4d(const double &l1, const double &l2, const double &phi, const double &z, std::vector<double> thetas);

  /**
   * @brief 4d- Integrand of MapMapMap Covariance (combining limber-integration with the integrand function)
   * @param l1 ell1
   * @param l2 ell2
   * @param phi angle between l1 and l2 [rad]
   * @param z redshift
   * @param thetas_123 Aperture radii [rad]
   * @param thetas_456 Aperture radii [rad]
   * @return value of integrand
   */
  double integrand_Gaussian_Aperture_Covariance(const double &l1, const double &l2, const double &phi, const double &z,
                                                std::vector<double> thetas_123, std::vector<double> thetas_456);

  /**
   * @brief MapMapMap integrand, integrated over phi
   * \f$ \int \mathrm{d}\phi \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3) + \mathrm{2 terms}]\f$
   * @param l1 ell1
   * @param l2 ell2
   * @param thetas Aperture radii [rad]
   * @return value of integrand
   */
  double integral_phi(double l1, double l2, std::vector<double> thetas);

  /**
   * @brief MapMapMap integrand, integrated over phi and l2
   * \f$ \int \mathrm{d} \ell_2 \int \mathrm{d}\phi \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3) + \mathrm{2 terms}]\f$
   * @param l1 ell1
   * @param thetas Aperture radii [rad]
   * @return value of integrand
   */
  double integral_l2(double l1, std::vector<double> thetas);

  /**
   * @brief MapMapMap integrand, integrated over phi, l2 and l1
   * \f$ \int \mathrm{d} \ell_1 \int \mathrm{d} \ell_2 \int \mathrm{d}\phi \ell_1 \ell_2 b(\ell_1, \ell_2, \phi)[\hat{u}(\theta_1\ell_1)\hat{u}(\theta_2\ell_2)\hat{u}(\theta_3\ell_3) + \mathrm{2 terms}]\f$
   * @param thetas Aperture radii [rad]
   * @return value of integrand
   */
  double integral_l1(std::vector<double> thetas);

  /**
   * @brief MapMapMap integrand for phi-integration, formulated for use of GSL integration routine
   * Calls integral_phi(l1_, l2_)
   * @param phi Phi [rad]
   * @param thisPtr Pointer to ApertureStatisticsContainer that is integrated
   */
  static double integrand_phi(double phi, void *thisPtr);

  /**
   * @brief MapMapMap integrand for l2-integration, formulated for use of GSL integration routine
   * Calls integral_l2(l1_)
   * @param l2 ell2
   * @param thisPtr Pointer to ApertureStatisticsContainer that is integrated
   */
  static double integrand_l2(double l2, void *thisPtr);

  /**
   * @brief MapMapMap integrand for l1-integration, formulated for use of GSL integration routine
   * Calls integral_l1()
   * @param l1 ell1
   * @param thisPtr Pointer to ApertureStatisticsCOntainer that is integrated
   */
  static double integrand_l1(double l1, void *thisPtr);

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
  static int integrand(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

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
  static int integrand_4d(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

  /**
   * @brief Wrapper of the Gaussian Aperture Covariance integrand to the cubature library
   * See https://github.com/stevengj/cubature for documentation
   * @param ndim Number of dimensions of integral (here: 4, automatically assigned by integration)
   * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
   * @param vars Array containing integration variables
   * @param thisPts Pointer to ApertureStatisticsContainer instance
   * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration)
   * @param value Value of integral
   * @return 0 on success
   */
  static int integrand_Gaussian_Aperture_Covariance(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

public:
  /**
   * @brief Constructor from cosmology
   * @param Bispectrum bispectrum calculator object, already initialized
   */
  ApertureStatistics(BispectrumCalculator *Bispectrum);

  /**
   * @brief Aperturestatistics calculated from Bispectrum
   * @warning This is NOT Eq 58 from Schneider, Kilbinger & Lombardi (2003), but a third of that, due to the bispectrum definition
   * If CUBATURE is true, this uses the pcubature routine from the cubature library
   * If CUBATURE is false / not defined, this uses GSL and three separate integrals over each dimension (SLOOOOOW AF)
   * @param thetas Aperture Radii, array should contain 3 values [rad]
   */
  double MapMapMap(const std::vector<double> &thetas);

  /**
   * @brief Gaussian Aperturestatistics covariance calculated from non-linear Power spectrum
   * If CUBATURE is true, this uses the pcubature routine from the cubature library
   * @param thetas_123 Aperture Radii, array should contain 3 values [rad]
   * @param thetas_456 Aperture Radii, array should contain 3 values [rad]
   * @param survey_area Survey Area [rad^2]
   */
  double MapMapMap_covariance_Gauss(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456, double survey_area);

  /**
   * @brief Geometric factor in covariance calculation for square survey, Eq. 137 in document.pdf
   *
   * @param ellX first component of wavevector [1/rad]
   * @param ellY second component of wavevector [1/rad]
   * @param thetaMax sidelength of survey [rad]
   * @return double
   */
  double G(double ellX, double ellY, double thetaMax);

  double G_circular(double ell, double thetaMax);

  double integrand_L1(double a, double b, double c, double d, double e, double f, double thetaMax,
                      double theta1, double theta2, double theta3, double theta4, double theta5, double theta6);

  double integrand_L2_A(double ell, double theta1, double theta2);

  double integrand_L2_B(double ellX, double ellY, double thetaMax, double theta1, double theta2);

  double integrand_L1_circular(double a, double b, double c, double d, double e, double f, double thetaMax,
                               double theta1, double theta2, double theta3, double theta4, double theta5, double theta6);

  double integrand_L2_B_circular(double ell, double thetaMax, double theta1, double theta2);

  double integrand_L4_circular(double v, double ell2, double ell4, double ell5, double alphaV, double alpha2, double alpha4,
                               double alpha5, double thetaMax, double theta1, double theta2, double theta3, double theta4, double theta5, double theta6);

  static int integrand_L1(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

  static int integrand_L2_A(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

  static int integrand_L2_B(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

  static int integrand_L4(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

  double L1(double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double thetaMax);

  double L2(double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double thetaMax);

  double L4(double theta1, double theta2, double theta3, double theta4, double theta5, double theta6, double thetaMax);

  double L1_total(const std::vector<double> &thetas123, const std::vector<double> &thetas456, double thetaMax);

  double L2_total(const std::vector<double> &thetas123, const std::vector<double> &thetas456, double thetaMax);

  double L4_total(const std::vector<double> &thetas123, const std::vector<double> &thetas456, double thetaMax);

  double Cov(const std::vector<double> &thetas123, const std::vector<double> &thetas456, double thetaMax);

  double Cov_NG(const std::vector<double> &thetas123, const std::vector<double> &thetas456, double thetaMax);

  double integrand_NonGaussian_Aperture_Covariance(const double &l1, const double &l2, const double &l5, const double &phi1, const double &phi2, const double &z,
                                                   const double &theta1, const double &theta2, const double &theta3,
                                                   const double &theta4, const double &theta5, const double &theta6);

  static int integrand_NonGaussian_Aperture_Covariance(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value);

  double MapMapMap_covariance_NonGauss(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456, double survey_area);
};

/**
 * @brief This struct contains an instance of the aperture statistics and aperture radii
 * It is needed for the integration with cubature
 */
struct ApertureStatisticsContainer
{
  /** Aperturestatistics to be calculated*/
  ApertureStatistics *aperturestatistics;

  /** Apertureradii [rad]*/
  std::vector<double> thetas;

  /** For covariance: Apertureradii [rad]*/
  std::vector<double> thetas2;

  /** For covariance: Sidelength of square survey [rad]*/
  double thetaMax;
};

#endif // APERTURESTATISTICS_HPP
