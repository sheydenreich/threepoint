#ifndef APERTURESTATISTICSCOVARIANCE_HPP
#define APERTURESTATISTICSCOVARIANCE_HPP

#include "apertureStatistics.hpp"

#include <string>
#include <vector>

class ApertureStatisticsCovariance
{
public:
    std::string type; // defines survey geometry, can be 'circle', 'square', 'infinite'

    /**
     * @brief Gives extent of survey [rad].
     * if type = 'circular', this is the radius.
     * if type = 'square', this is the side length.
     * if type = 'infinite', this is sqrt(A).
     */
    double thetaMax;

    double lMin;

    ApertureStatistics *apertureStatistics; // Instance of apertureStatistics, needed to access uHat and bispectrum

    ApertureStatisticsCovariance(){}; // Empty Constructor

    /**
     * @brief Construct a new Aperture Statistics Covariance object
     *
     * @param type_ Survey geometry, can be 'circle', 'square', 'infinite'. Exception is thrown, if other value is passed
     * @param thetaMax_ Extend of survey [rad], radius for circle, sidelength for square, sqrt(A) for infinite
     * @param apertureStatistics_ Object defining the type of aperture statistics and bispectrum
     */
    ApertureStatisticsCovariance(const std::string &type_, const double &thetaMax_, ApertureStatistics *apertureStatistics_);

    /**
     * @brief Writes a covariance matrix (or one part of it) to a file
     *
     * @param values Values of the covariance matrix, sorted in row-major order
     * @param N Number of rows  (= Number of cols)
     * @param filename File to which will be written. Exception is thrown, if file cannot be created (e.g. if folder doesn't exist)
     */
    void writeCov(const std::vector<double> &values, const int &N, const std::string &filename);

    /**
     * @brief Calculate the Gaussian part of the covariance for one aperture radii combination
     *
     * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
     * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
     * @return double Cov_Gaussian(thetas_123, thetas_456) = T1_total(thetas_123, thetas_456) + T2_total(thetas_123, thetas_456)
     */
    double Cov_Gaussian(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

    /**
     * @brief Calculate the Bispectrum-dependent Non-Gaussian part of the covariance for one aperture radii combination.
     * This is implemented only for type='circle' and type='infinite'. Other types throw an exception.
     *
     * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
     * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
     * @return double Cov_NonGaussian(thetas_123, thetas_456) =T4_total(thetas_123, thetas_456)
     */
    double Cov_NonGaussian(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

    /**
     * @brief Calculates the first Term in the Gaussian Covariance with all permutations
     *
     * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
     * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
     * @return double T_1, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_1 + 5 Permutations
     */
    double T1_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

    /**
     * @brief Calculates the second Term in the Gaussian Covariance with all permutations. Returns 0 for type='infinite'
     *
     * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
     * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
     * @return double T_2, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_2 + 8 Permutations
     */
    double T2_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

    /**
     * @brief Calculates the first Term in the NonGaussian Covariance with all permutations. Throws an exception if type='square'.
     *
     * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
     * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
     * @return double T_4, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_4 + 8 Permutations
     */
    double T4_total(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);

    /**
     * @brief First Term of Gaussian Covariance for one permutation
     *
     * @param theta1 Aperture radius [rad]
     * @param theta2 Aperture radius [rad]
     * @param theta3 Aperture radius [rad]
     * @param theta4 Aperture radius [rad]
     * @param theta5 Aperture radius [rad]
     * @param theta6 Aperture radius [rad]
     */
    double T1(const double &theta1, const double &theta2, const double &theta3,
              const double &theta4, const double &theta5, const double &theta6);

    /**
     * @brief Second Term of Gaussian Covariance for one permutation
     *
     * @param theta1 Aperture radius [rad]
     * @param theta2 Aperture radius [rad]
     * @param theta3 Aperture radius [rad]
     * @param theta4 Aperture radius [rad]
     * @param theta5 Aperture radius [rad]
     * @param theta6 Aperture radius [rad]
     */
    double T2(const double &theta1, const double &theta2, const double &theta3,
              const double &theta4, const double &theta5, const double &theta6);

    /**
     * @brief First Term of NonGaussian Covariance for one permutation
     *
     * @param theta1 Aperture radius [rad]
     * @param theta2 Aperture radius [rad]
     * @param theta3 Aperture radius [rad]
     * @param theta4 Aperture radius [rad]
     * @param theta5 Aperture radius [rad]
     * @param theta6 Aperture radius [rad]
     */
    double T4(const double &theta1, const double &theta2, const double &theta3,
              const double &theta4, const double &theta5, const double &theta6);

private:
    /**
     * @brief Wrapper of integrand_T1 for the cubature library
     * See https://github.com/stevengj/cubature for documentation
     * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
     * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
     * @param vars Array containing integration variables
     * @param container Pointer to ApertureStatisticsCovarianceContainer instance
     * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
     * @param value Value of integral
     * @return 0 on success
     */
    static int integrand_T1(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

    /**
     * @brief Wrapper of integrand_T2_part1 for the cubature library
     * See https://github.com/stevengj/cubature for documentation
     * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
     * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
     * @param vars Array containing integration variables
     * @param container Pointer to ApertureStatisticsCovarianceContainer instance
     * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
     * @param value Value of integral
     * @return 0 on success
     */
    static int integrand_T2_part1(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

    /**
     * @brief Wrapper of integrand_T2_part2 for the cubature library
     * See https://github.com/stevengj/cubature for documentation
     * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
     * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
     * @param vars Array containing integration variables
     * @param container Pointer to ApertureStatisticsCovarianceContainer instance
     * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
     * @param value Value of integral
     * @return 0 on success
     */
    static int integrand_T2_part2(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

    /**
     * @brief Wrapper of integrand_T4 for the cubature library
     * See https://github.com/stevengj/cubature for documentation
     * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
     * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
     * @param vars Array containing integration variables
     * @param container Pointer to ApertureStatisticsCovarianceContainer instance
     * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
     * @param value Value of integral
     * @return 0 on success
     */
    static int integrand_T4(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

    /**
     * @brief Integrand of Term1 for circular survey.
     *
     * @param a =(ellvec_1+ellvec_2+ellvec_3)_x
     * @param b =(ellvec_1+ellvec_2+ellvec_3)_x
     * @param c =ellvec_2x
     * @param d =ellvec_2y
     * @param e =ellvec_3x
     * @param f =ellvec_3y
     * @param theta1 Aperture radius [rad]
     * @param theta2 Aperture radius [rad]
     * @param theta3 Aperture radius [rad]
     * @param theta4 Aperture radius [rad]
     * @param theta5 Aperture radius [rad]
     * @param theta6 Aperture radius [rad]
     */
    double integrand_T1_circle(const double &a, const double &b, const double &c, const double &d, const double &e, const double &f,
                               const double &theta1, const double &theta2, const double &theta3, const double &theta4,
                               const double &theta5, const double &theta6);

    /**
     * @brief Integrand of Term1 for square survey.
     *
     * @param a =(ellvec_1+ellvec_2+ellvec_3)_x
     * @param b =(ellvec_1+ellvec_2+ellvec_3)_x
     * @param c =ellvec_2x
     * @param d =ellvec_2y
     * @param e =ellvec_3x
     * @param f =ellvec_3y
     * @param theta1 Aperture radius [rad]
     * @param theta2 Aperture radius [rad]
     * @param theta3 Aperture radius [rad]
     * @param theta4 Aperture radius [rad]
     * @param theta5 Aperture radius [rad]
     * @param theta6 Aperture radius [rad]
     */
    double integrand_T1_square(const double &a, const double &b, const double &c, const double &d, const double &e, const double &f,
                               const double &theta1, const double &theta2, const double &theta3, const double &theta4,
                               const double &theta5, const double &theta6);

    /**
     * @brief Integrand of Term1 for infinite survey.
     *
     * @param l1 ell1
     * @param l2 ell2
     * @param phi angle between l1 and l2 [rad]
     * @param theta1 Aperture radius [rad]
     * @param theta2 Aperture radius [rad]
     * @param theta3 Aperture radius [rad]
     * @param theta4 Aperture radius [rad]
     * @param theta5 Aperture radius [rad]
     * @param theta6 Aperture radius [rad]
     */
    double integrand_T1_infinite(const double &l1, const double &l2, const double &phi, 
                                 const double &theta1, const double &theta2, const double &theta3, const double &theta4,
                                 const double &theta5, const double &theta6);

    /**
     * @brief Integrand for first part of Term 2, applicable to both circular and square surveyy
     *
     * @param ell ell
     * @param theta1 Aperture radius [rad]
     * @param theta2 Aperture radius [rad]
     */
    double integrand_T2_part1(const double &ell, const double &theta1, const double &theta2);

    /**
     * @brief Integrand for second part of Term 2 for circular survey
     *
     * @param ell ell
     * @param theta1 Aperture radius [rad]
     * @param theta2 Aperture radius [rad]
     */
    double integrand_T2_part2_circle(const double &ell, const double &theta1, const double &theta2);

    /**
     * @brief Integrand for second part of Term 2 for square survey
     *
     * @param ellX ell_x
     * @param ellY ell_y
     * @param theta1 Aperture radius [rad]
     * @param theta2 Aperture radius [rad]
     */
    double integrand_T2_part2_square(const double &ellX, const double &ellY, const double &theta1, const double &theta2);

    /**
     * @brief Integrand for Term 4 for circular survey
     *
     * @param v |ellvec_1-ellvec_4|
     * @param ell2 |ellvec_2|
     * @param ell4 |ellvec_4|
     * @param ell5  |ellvec_5|
     * @param alphaV polar angle of v
     * @param alpha2 polar angle of ellvec_2
     * @param alpha4 polar angle of ellvec_4
     * @param alpha5 polar angle of ellvec_5
     * @param theta1 Aperture radius [rad]
     * @param theta2 Aperture radius [rad]
     * @param theta3 Aperture radius [rad]
     * @param theta4 Aperture radius [rad]
     * @param theta5 Aperture radius [rad]
     * @param theta6 Aperture radius [rad]
     */
    double integrand_T4_circle(const double &v, const double &ell2, const double &ell4, const double &ell5,
                               const double &alphaV, const double &alpha2, const double &alpha4, const double &alpha5,
                               const double &theta1, const double &theta2, const double &theta3,
                               const double &theta4, const double &theta5, const double &theta6);

    /**
     * @brief Integrand for Term 4 for an infinite survey
     *
     * @param l1 ell1
     * @param l2 ell2
     * @param l5 ell5
     * @param phi1 Angle between ell1 and ell2
     * @param phi2 Angle between ell1 and ell5
     * @param theta1 Aperture radius [rad]
     * @param theta2 Aperture radius [rad]
     * @param theta3 Aperture radius [rad]
     * @param theta4 Aperture radius [rad]
     * @param theta5 Aperture radius [rad]
     * @param theta6 Aperture radius [rad]
     */
    double integrand_T4_infinite(const double &l1, const double &l2, const double &l5, const double &phi1, const double &phi2,
                                 const double &theta1, const double &theta2, const double &theta3,
                                 const double &theta4, const double &theta5, const double &theta6);

    /**
     * @brief Geometric factor for circular survey
     *
     * @param ell |ellvec|
     */
    double G_circle(const double &ell);

    /**
     * @brief Geometric factor for square survey
     *
     * @param ellX ell_x
     * @param ellY ell_y
     */
    double G_square(const double &ellX, const double &ellY);
};


class ApertureStatisticsCovarianceContainer
{
    public:
    // Covariance to be calculated
    ApertureStatisticsCovariance * apertureStatisticsCovariance;

    // First aperture radii [rad]
    std::vector<double> thetas_123;

    // Second aperture radii [rad]
    std::vector<double> thetas_456;
};

#endif // APERTURESTATISTICSCOVARIANCE_HPP