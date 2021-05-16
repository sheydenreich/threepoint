#ifndef GAMMAHEADERDEF
#define GAMMAHEADERDEF

#include "bispectrum.hpp"
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_math.h>
#include <complex>
#include "cubature.h"


/** If true, the R-integration is performed via Ogata et al. (2005). This is recommended. 
  * If false, cubature performs a 4-dimensional integration. This normally leads to memory issues at scales >10 arcmin since the
  * integrand needs too many subdivisions.
*/
#define USE_OGATA true



class GammaCalculator
{

public:

  /**
  * @brief Constructor for the class
  * @param Bispectrum pointer to instance of a BispectrumCalculator class
  * @param prec_h first accuracy parameter for Ogata et al. integration.
  * determines the bin-width in the bessel-integration
  * recommended: between 0.05 and 0.2
  * @param prec_k second accuracy parameter for Ogata et al. integration.
  * determines the number of roots taken into account in the bessel integration.
  * does not increase precision when larger than 3.5
  * @param triangle_center center of the triangle for the shear 3pcfs. 
  * choices: "orthocenter", "centroid"
  * the latter is recommended due to compatibility with treecorr (https://rmjarvis.github.io/TreeCorr/)
  */
  GammaCalculator(BispectrumCalculator* Bispectrum, double prec_h, double prec_k, std::string triangle_center);

  /**
  * @brief First shear three-point correlation function Gamma^0
  * the integration method is determined by the USE_OGATA define
  * the triangle center is specified at the initialization of the class
  * @param x1 sidelength of triangle [rad]
  * @param x2 sidelength of triangle [rad]
  * @param x3 sidelength of triangle [rad]
  * @return value of Gamma^0
  */
  std::complex<double> gamma0(double x1, double x2, double x3); 

  /**
  * @brief First shear three-point correlation function Gamma^1
  * the integration method is determined by the USE_OGATA define
  * the triangle center is specified at the initialization of the class
  * @param x1 sidelength of triangle [rad]
  * @param x2 sidelength of triangle [rad]
  * @param x3 sidelength of triangle [rad]
  * @return value of Gamma^1
  */
  std::complex<double> gamma1(double x1, double x2, double x3); 

  /**
  * @brief First shear three-point correlation function Gamma^2
  * the integration method is determined by the USE_OGATA define
  * the triangle center is specified at the initialization of the class
  * @param x1 sidelength of triangle [rad]
  * @param x2 sidelength of triangle [rad]
  * @param x3 sidelength of triangle [rad]
  * @return value of Gamma^2
  */
  std::complex<double> gamma2(double x1, double x2, double x3); 

  /**
  * @brief First shear three-point correlation function Gamma^3
  * the integration method is determined by the USE_OGATA define
  * the triangle center is specified at the initialization of the class
  * @param x1 sidelength of triangle [rad]
  * @param x2 sidelength of triangle [rad]
  * @param x3 sidelength of triangle [rad]
  * @return value of Gamma^3
  */
  std::complex<double> gamma3(double x1, double x2, double x3); 

  // For testing: analytical ggg correlation

  /**
  * @brief First shear three-point correlation function Gamma^0
  * for the test-case of an analytic lensing potential
  * @param r1 sidelength of triangle [rad]
  * @param r2 sidelength of triangle [rad]
  * @param r3 sidelength of triangle [rad]
  * @return value of Gamma^0
  */
  std::complex<double> ggg(double r1, double r2, double r3); 

  /**
  * @brief First shear three-point correlation function Gamma^1
  * for the test-case of an analytic lensing potential
  * @param r1 sidelength of triangle [rad]
  * @param r2 sidelength of triangle [rad]
  * @param r3 sidelength of triangle [rad]
  * @return value of Gamma^1
  */
  std::complex<double> gstargg(double r1, double r2, double r3); 

  /**
  * @brief First shear three-point correlation function Gamma^2
  * for the test-case of an analytic lensing potential
  * @param r1 sidelength of triangle [rad]
  * @param r2 sidelength of triangle [rad]
  * @param r3 sidelength of triangle [rad]
  * @return value of Gamma^2
  */
  std::complex<double> ggstarg(double r1, double r2, double r3);

  /**
  * @brief First shear three-point correlation function Gamma^3
  * for the test-case of an analytic lensing potential
  * @param r1 sidelength of triangle [rad]
  * @param r2 sidelength of triangle [rad]
  * @param r3 sidelength of triangle [rad]
  * @return value of Gamma^3
  */
  std::complex<double> gggstar(double r1, double r2, double r3);

private:
    // integration precision for bessel integral
    double prec_h;
    unsigned int prec_k;

    // pre-computed weights for bessel integrations
    double* bessel_zeros;
    double* pi_bessel_zeros;
    double* array_psi;
    double* array_psi_J2;
    double* array_bessel;
    double* array_psip;
    double* array_w;
    double* array_product;
    double* array_product_J2;

    // maximum redshift for integration
    double z_max;

    // class for modelling bispectrum
    BispectrumCalculator* Bispectrum_;

    /*********** Geometric Functions ********************/

    /**
    * @brief Interior angle of a triangle with sides an1, an2, opp
    * @param an1 first side adjacent to the angle
    * @param an2 second side adjacent to the angle
    * @param opp side opposite of the angle
    * @return value of the angle
    */
    double interior_angle(double an1, double an2, double opp);

    /**
    * @brief Height of a triangle with sides an1, an2, opp
    * @param an1 first side adjacent to height of triangle
    * @param an2 first side adjacent to height of triangle
    * @param opp side perpendicular to height of triangle
    * @return value of the height
    */
    double height_of_triangle(double an1, double an2, double opp);


    /*********** Functions for Bessel Integration ********************/


    /**
    * @brief computes the weights for the Ogata et al. Bessel integration
    * @param prec_h_arg determines the bin width relative to the distance between two roots of the Bessel function
    * @param prec_k_arg determines, how many roots of the Bessel function are taken into account
    * @return 0 on success
    */
    int initialize_bessel(double prec_h_arg, double prec_k_arg);


    /* functions for Ogata et al. Bessel integration */
    /**
    * @brief psi, and d/dt psi as in Ogata et al.
    */
    double psi(double t);
    double psip(double t);



    /*************** Terms that appear in the Schneider et al. Integration ****************/

    /**
    * @brief alpha3 as defined in Eq. (12) from Schneider et al.
    * @param psi phase of the vector (ell1,ell2)
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param phi angle between l1 and l2 [rad]
    * @param varpsi psi3 [rad], see Fig. 1 in Schneider et al. 
    * @return value of the function
    */
    double alpha(double psi, double x1, double x2, double phi, double varpsi);

    /**
    * @brief betabar as defined in Eq. (9) from Schneider et al.
    * @param psi phase of the vector (ell1,ell2)
    * @param phi angle between l1 and l2 [rad]
    */
    double betabar(double psi, double phi);

    /**
    * @brief exp(2i*betabar) as defined in Eq. (9) from Schneider et al.
    * @param psi phase of the vector (ell1,ell2)
    * @param phi angle between l1 and l2 [rad]
    * @return value of the function
    */
    std::complex<double> exponential_of_betabar(double psi, double phi);

    /**
    * @brief prefactor for integration of Gamma0 in Eq. (15) of Schneider et al.
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @param psi phase of the vector (ell1,ell2)
    * @param phi angle between l1 and l2 [rad]
    * @param varpsi psi3 [rad], see Fig. 1 in Schneider et al. 
    * @return exp(i(phi1-phi2-6alpha3))
    */
    std::complex<double> exponential_prefactor(double x1, double x2, double x3, double psi, double phi, double varpsi);
    
    /**
    * @brief A3/|ell| defined in Eq. (12) of Schneider et al.
    * @param psi phase of the vector (ell1,ell2)
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param phi angle between l1 and l2 [rad]
    * @param varpsi psi3 [rad], see Fig. 1 in Schneider et al. 
    * @return A3/|ell|
    */
    double A(double psi, double x1, double x2, double phi, double varpsi);


    /************************ Integration Methods *******************************/

    /* Gamma0 from 4-dimensional (r,phi,psi,z) integration with cubature */
    
    /**
    * @brief one of the three prefactors of the Bispectrum in the Integrand of Eq. (15) in Schneider et al.
    * @param r modulus of the vector (ell1,ell2)
    * @param phi angle between l1 and l2 [rad]
    * @param psi phase of the vector (ell1,ell2)
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @return value of the prefactor
    */
    std::complex<double> integrand_gamma0_r_phi_psi_one_x(double r, double phi, double psi, double x1, double x2, double x3);

    /**
    * @brief The complete Integrand of Eq. (15) in Schneider et al. (including a limber-integration for the kappa-Bispectrum)
    * @param r modulus of the vector (ell1,ell2)
    * @param phi angle between l1 and l2 [rad]
    * @param psi phase of the vector (ell1,ell2)
    * @param z redshift
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @return value of the Integrand
    */
    std::complex<double> integrand_gamma0_cubature(double r, double phi, double psi, double z, double x1, double x2, double x3);

    /**
    * @brief Wrapper of the Integrand for cubature library
    * See https://github.com/stevengj/cubature for documentation
    * @param ndim Number of dimensions of integral (here: 4)
    * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
    * @param vars Array containing integration variables
    * @param fdata Pointer to GammaCalculatorContainer container instance
    * @param fdim Dimensions of integral output (here: 2)
    * @param value Value of integral
    * @return 0 on success
    */
    static int integrand_gamma0_cubature(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value);
    
    /**
    * @brief Value of the Integral in Eq. (15) when performing a 4-dimensional integration using cubature
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @return value of the Integral
    */
    std::complex<double> gamma0_from_cubature(double x1, double x2, double x3);
    
    
    /* Gamma0 from 3-dimensional (z,phi,psi) integration with cubature, R-integration with ogata */

    /**
    * @brief \int_0^\infty dR J_6(A_3 R) b_delta(R\cos(psi),R\sin(psi),phi,z), where b_\delta is the matter bispectrum
    * The Integration is performed using Ogata et al.
    * @param z redshift
    * @param phi angle between l1 and l2 [rad]
    * @param psi phase of the vector (ell1,ell2)
    * @param A3 value of the term A3
    * @return value of the Integranl
    */
    double integrated_bdelta_times_rcubed_J6(double z, double phi, double psi, double A3);

    /**
    * @brief One term of the Integrand of Eq. (18) in Schneider et al. (including a limber-integration for the kappa-Bispectrum)
    * where the |ell|-integration has already been performed using Ogata et al.
    * @param z redshift
    * @param phi angle between l1 and l2 [rad]
    * @param psi phase of the vector (ell1,ell2)
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @return value of the Integrand
    */
    std::complex<double> integrand_z_phi_psi_one_x(double z, double phi, double psi, double x1, double x2, double x3);

    /**
    * @brief Complete Integrand of Eq. (18) in Schneider et al. (including a limber-integration for the kappa-Bispectrum)
    * where the |ell|-integration has already been performed using Ogata et al.
    * @param z redshift
    * @param phi angle between l1 and l2 [rad]
    * @param psi phase of the vector (ell1,ell2)
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @return value of the Integrand
    */
    std::complex<double> integrand_z_phi_psi(double z, double phi, double psi, double x1, double x2, double x3);

    /**
    * @brief Wrapper of the Integrand for cubature library
    * See https://github.com/stevengj/cubature for documentation
    * @param ndim Number of dimensions of integral (here: 3)
    * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
    * @param vars Array containing integration variables
    * @param fdata Pointer to GammaCalculatorContainer container instance
    * @param fdim Dimensions of integral output (here: 2)
    * @param value Value of integral
    * @return 0 on success
    */
    static int integrand_gamma0_cubature_and_ogata(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value);

    /**
    * @brief Value of the Integral in Eq. (15) when performing a 3-dimensional (z,phi,psi)-integration using cubature,
    * and an |ell|-integration using Ogata et al.
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @return value of the Integral
    */
    std::complex<double> gamma0_from_cubature_and_ogata(double x1, double x2, double x3);


    /* Gamma1 from 3-dimensional (r,phi,psi) integration with cubature, z-integration with Gaussian Quadrature */
    
    /**
    * @brief Integrand of Eq. (18) in Schneider et al.
    * @param r modulus of the vector (ell1,ell2)
    * @param phi angle between l1 and l2 [rad]
    * @param psi phase of the vector (ell1,ell2)
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @return value of the prefactor
    */
    std::complex<double> integrand_r_phi_psi_gamma1(double r, double phi, double psi, double x1, double x2, double x3);
    
    /**
    * @brief Wrapper of the Integrand for cubature library
    * See https://github.com/stevengj/cubature for documentation
    * @param ndim Number of dimensions of integral (here: 3)
    * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
    * @param vars Array containing integration variables
    * @param fdata Pointer to GammaCalculatorContainer container instance
    * @param fdim Dimensions of integral output (here: 2)
    * @param value Value of integral
    * @return 0 on success
    */
    static int integrand_gamma1_no_ogata_no_limber(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value);
    
    /**
    * @brief Value of the Integral in Eq. (18) when performing a 3-dimensional integration using cubature
    * and the limber-integration using Gaussian Quadrature (96pt)
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @return value of the Integral
    */
    std::complex<double> gamma1_from_cubature(double x1, double x2, double x3);

    /* Gamma1 from 3-dimensional (z,phi,psi) integration with cubature, R-integration with ogata */

    /**
    * @brief \int_0^\infty dR J_2(A_3 R) b_delta(R\cos(psi),R\sin(psi),phi,z), where b_\delta is the matter bispectrum
    * The Integration is performed using Ogata et al.
    * @param z redshift
    * @param phi angle between l1 and l2 [rad]
    * @param psi phase of the vector (ell1,ell2)
    * @param A3 value of the term A3
    * @return value of the Integranl
    */
    double integrated_bdelta_times_rcubed_J2(double z, double phi, double psi, double A3);
    
    /**
    * @brief Complete Integrand of Eq. (18) in Schneider et al. (including a limber-integration for the kappa-Bispectrum)
    * where the |ell|-integration has already been performed using Ogata et al.
    * @param z redshift
    * @param phi angle between l1 and l2 [rad]
    * @param psi phase of the vector (ell1,ell2)
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @return value of the Integrand
    */
    std::complex<double> integrand_z_phi_psi_gamma1(double z, double phi, double psi, double x1, double x2, double x3);
    
    /**
    * @brief Wrapper of the Integrand for cubature library
    * See https://github.com/stevengj/cubature for documentation
    * @param ndim Number of dimensions of integral (here: 3)
    * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
    * @param vars Array containing integration variables
    * @param fdata Pointer to GammaCalculatorContainer container instance
    * @param fdim Dimensions of integral output (here: 2)
    * @param value Value of integral
    * @return 0 on success
    */
    static int integrand_gamma1_cubature_and_ogata(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value);
    
    /**
    * @brief Value of the Integral in Eq. (18) when performing a 3-dimensional integration using cubature
    * and the |ell|-Integration using Ogata et al.
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @return value of the Integral
    */
    std::complex<double> gamma1_from_cubature_and_ogata(double x1, double x2, double x3);

    /* Analytic test functions */

    /**
    * @brief Value for gamma0 when derived from an analytic three-point lensing potential (reference coming)
    * for one choice of the free parameter a
    * @param r1 sidelength of triangle [rad]
    * @param r2 sidelength of triangle [rad]
    * @param r3 sidelength of triangle [rad]
    * @param a free scaling parameter
    * @return value of gamma0
    */
    std::complex<double> ggg_single_a(double r1, double r2, double r3, double a);

    /**
    * @brief Value for gamma1 when derived from an analytic three-point lensing potential (reference coming)
    * for one choice of the free parameter a
    * @param r1 sidelength of triangle [rad]
    * @param r2 sidelength of triangle [rad]
    * @param r3 sidelength of triangle [rad]
    * @param a free scaling parameter
    * @return value of gamma1
    */
    std::complex<double> gggstar_single_a(double r1, double r2, double r3, long double a);

    /* Rotating from orthocenter to centroid */

    /** 
    * @brief if true, the gamma_i are computed with the centroid as triangle center.
    * this is recommended, as treecorr (https://rmjarvis.github.io/TreeCorr/) chooses this parametrization.
    * Also, when computing third-order aperture masses from correlation functions using either
    * Jarvis et al. or Schneider et al., the three-point correlation functions need to be
    * measured with respect to the centroid.
    */
    bool convert_orthocenter_to_centroid_bool;

    /**
    * @brief Computes one of the rotation angles for converting the gammas from orthocenter
    * to centroid using Eq. (13) of Schneider et al. (2013) 
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @return value of rotation angle
    */
    double one_rotation_angle_otc(double x1, double x2, double x3);

    /**
    * @brief Converts a shear correlation function from orthocenter
    * to centroid using Eq. (15) of Schneider et al. (2013) 
    * @param gamma correlation function with respect to orthocenter
    * @param x1 sidelength of triangle [rad]
    * @param x2 sidelength of triangle [rad]
    * @param x3 sidelength of triangle [rad]
    * @param conjugate_phi1 true if the first shear component is complex conjugate
    * (as is the case in Gamma^1), flase of not (Gamma^0)
    * @return correlation function with respect to centroid
    */
    std::complex<double> convert_orthocenter_to_centroid(std::complex<double> gamma, double x1, double x2, double x3, bool conjugate_phi1);

};

struct GammaCalculatorContainer
{
    double x1,x2,x3;
    GammaCalculator* gammaCalculator;
};

#endif
