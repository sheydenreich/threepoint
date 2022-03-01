#ifndef APERTURESTATISTICSCOVARIANCE_CUH
#define APERTURESTATISTICSCOVARIANCE_CUH

#include <string>
#include <vector>

extern int type; // defines survey geometry, can be 0, 1, or 2, corresponding to: 'circle', 'square', 'infinite'

/**
* @brief Gives extent of survey [rad].
* if type = 'circular', this is the radius.
* if type = 'square', this is the side length.
* if type = 'infinite', this is sqrt(A).
*/
extern __constant__ double dev_thetaMax;
extern double thetaMax;
extern __constant__ double dev_lMin;
extern double lMin;

extern __constant__ double dev_sigmaW;
extern double sigmaW, VW, chiMax;

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
      * @brief Calculates the second Term in the Non-Gaussian Covariance with all permutations. Throws an exception if type is not 'infinite
      * 
      * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
      * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
      * @return double T_5, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_5 + 8 Permutations
      */
      double T5_total(const std::vector<double> & thetas_123, const std::vector<double> &thetas_456);


           /**
      * @brief Calculates the third Term in the Non-Gaussian Covariance with all permutations. Throws an exception if type is not 'square'
      * 
      * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
      * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
      * @return double T_6, total(theta1, theta2, theta3, theta4, theta5, theta6) = T_6 + 5 Permutations
      */
      double T6_total(const std::vector<double> & thetas_123, const std::vector<double> &thetas_456);

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


     /**
      * @brief Second Term of NonGaussian Covariance for one permutation
      *
      * @param theta1 Aperture radius [rad]
      * @param theta2 Aperture radius [rad]
      * @param theta3 Aperture radius [rad]
      * @param theta4 Aperture radius [rad]
      * @param theta5 Aperture radius [rad]
      * @param theta6 Aperture radius [rad]
      */
      double T5(const double &theta1, const double &theta2, const double &theta3,
         const double &theta4, const double &theta5, const double &theta6);

     /**
      * @brief Third Term of NonGaussian Covariance for one permutation
      *
      * @param theta1 Aperture radius [rad]
      * @param theta2 Aperture radius [rad]
      * @param theta3 Aperture radius [rad]
      * @param theta4 Aperture radius [rad]
      * @param theta5 Aperture radius [rad]
      * @param theta6 Aperture radius [rad]
      */
      double T6(const double &theta1, const double &theta2, const double &theta3,
         const double &theta4, const double &theta5, const double &theta6);


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
     int integrand_T1(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);

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
     int integrand_T2_part1(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);
 
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
     int integrand_T2_part2(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);
 
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
     int integrand_T4(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);
 
     /**
      * @brief Wrapper of integrand_T5 for the cubature library
      * See https://github.com/stevengj/cubature for documentation
      * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
      * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
      * @param vars Array containing integration variables
      * @param container Pointer to ApertureStatisticsCovarianceContainer instance
      * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
      * @param value Value of integral
      * @return 0 on success
      */
      int integrand_T5(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);



           /**
      * @brief Wrapper of integrand_T6 for the cubature library
      * See https://github.com/stevengj/cubature for documentation
      * @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
      * @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
      * @param vars Array containing integration variables
      * @param container Pointer to ApertureStatisticsCovarianceContainer instance
      * @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
      * @param value Value of integral
      * @return 0 on success
      */
      int integrand_T6(unsigned ndim, size_t npts, const double *vars, void *container, unsigned fdim, double *value);


/**
 * @brief Integrand of Term1 for circular survey
 * 
 * @param vars Integration parameters (6 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T1_circle(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
                                    double theta4, double theta5, double theta6, double* value);


                                    /**
 * @brief Integrand of Term1 for square survey
 * 
 * @param vars Integration parameters (6 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T1_square(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
    double theta4, double theta5, double theta6, double* value);

    /**
 * @brief Integrand of Term1 for infinite survey
 * 
 * @param vars Integration parameters (3 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
__global__ void integrand_T1_infinite(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
    double theta4, double theta5, double theta6, double* value);


/**
 * @brief Integrand for first part of Term2, applicable to both circular and square survey
 * 
 * @param vars Integration parameters (1 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
 __global__ void integrand_T2_part1(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double* value);



    /**
 * @brief Integrand for second part of Term2 for circular survey
 * 
 * @param vars Integration parameters (1 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
 __global__ void integrand_T2_part2_circle(const double* vars, unsigned ndim, int npts, double theta1, double theta2,  double* value);
    

    /**
 * @brief Integrand for second part of Term2 for square survey
 * 
 * @param vars Integration parameters (2 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
 __global__ void integrand_T2_part2_square(const double* vars, unsigned ndim, int npts, double theta1, double theta2,  double* value);
    

    /**
 * @brief Integrand for Term 4 for circular survey
 * 
 * @param vars Integration parameters (8 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
 __global__ void integrand_T4_circle(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
    double theta4, double theta5, double theta6, double* value);
    
    /**
 * @brief Integrand for Term 4 for infinite survey
 * 
 * @param vars Integration parameters (5 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
 __global__ void integrand_T4_infinite(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
    double theta4, double theta5, double theta6, double* value);



    /**
 * @brief Integrand for Term 5 for infinite survey
 * 
 * @param vars Integration parameters (5 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
 __global__ void integrand_T5_infinite(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
   double theta4, double theta5, double theta6, double* value);


       /**
 * @brief Integrand for Term 6 for square survey
 * 
 * @param vars Integration parameters (9 D)
 * @param ndim Number of integration dimensions
 * @param npts Number of integration points
 * @param theta1 Aperture radius [rad]
 * @param theta2 Aperture radius [rad]
 * @param theta3 Aperture radius [rad]
 * @param theta4 Aperture radius [rad]
 * @param theta5 Aperture radius [rad]
 * @param theta6 Aperture radius [rad]
 * @param value Value of integral
 */
 __global__ void integrand_T6_square(const double* vars, unsigned ndim, int npts, double theta1, double theta2, double theta3, 
   double theta4, double theta5, double theta6, double* value);

    /**
     * @brief Geometric factor for circular survey
     *
     * @param ell |ellvec|
     */
__device__ double G_circle(const double& ell);


    /**
     * @brief Geometric factor for square survey
     *
     * @param ellX ell_x
     * @param ellY ell_y
     */
__device__ double G_square(const double& ellX, const double& ellY);



struct ApertureStatisticsCovarianceContainer
{
        // First aperture radii [rad]
        std::vector<double> thetas_123;

        // Second aperture radii [rad]
        std::vector<double> thetas_456;
};


void initCovariance();




// /// THE FOLLOWING THINGS ARE NEEDED FOR THE SUPERSAMPLE COVARIANCE

//    /**
//      * @brief Calculate the super sample covariance for one aperture radii combination.
//      *
//      * @param thetas_123 First three aperture radii [rad]. Exception thrown if not exactly three values.
//      * @param thetas_456 Second three aperture radii [rad]. Exception thrown if not exactly three values.
//      */
//      double Cov_SSC(const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);


//      /**
//       * @brief Redshift integrand of Cov_SSC
//       * 
//       */
//      double integrand_Cov_SSC(const double& z, const std::vector<double> &thetas_123, const std::vector<double> &thetas_456);


//      double f(const double& z, const std::vector<double> &thetas_123, const double& chi);

//      int integrand_f(unsigned ndim, size_t npts, const double* vars, void* container, unsigned fdim, double* value);

//      __global__ void integrand_f(const double* vars, unsigned ndim, int npts, double* value,  double theta1, double theta2, double theta3, double z, double chi);


//    __device__ double halobias(const double& m, const double& z);

//    __device__ double dev_rhobar(const double& z);

//    double rhobar(const double& z);









   

//    void setSurveyVolume();

//    void setSigmaW();

//    double WindowSurvey(const double& k1, const double& k2, const double& k3);


//    int integrand_SigmaW(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value);
   

// struct ApertureStatisticsSSCContainer
// {
//            // aperture radii [rad]
//            std::vector<double> thetas_123;

//            double z; //redshift
//            double chi; //Comoving distance

//            double ell1, ell2, ell3;
// };




// void initSSC();


// double Cov_Bispec_SSC(const double& ell);

// double integrand_Cov_Bispec_SSC(const double& z, const double& ell);

// double I3(const double& ell1, const double& ell2, const double& ell3, const double& z, const double& chi);

// int integrand_I3(unsigned ndim, size_t npts, const double* vars, void* container, unsigned fdim, double* value);

// __global__ void integrand_I3(const double* vars, unsigned ndim, int npts, double* value, double ell1, double ell2, double ell3, double z, double chi);

#endif //APERTURESTATISTICSCOVARIANCE_CUH