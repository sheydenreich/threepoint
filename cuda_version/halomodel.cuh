const double logMmin=9;
const double logMmax=17;//20;//17;
const int n_mbins=128;
extern __constant__ double devLogMmin, devLogMmax;
extern int __constant__ dev_n_mbins;



extern __constant__ double dev_sigma2_array[n_mbins]; 
extern __constant__ double dev_dSigma2dm_array[n_mbins]; 
extern double sigma2_array[n_mbins]; 
extern double dSigma2dm_array[n_mbins]; 


/**
* @brief Initialization function
* Calculates sigma^2(m) and dsigma²/dm and stores them on GPU
* @warning Requires that bispectrum has been initialized!
* 
*/
void initHalomodel();


/**
* @brief Halo mass function (Sheth & Tormen 1999)
* 
* @param m Halo mass [h^-1 Msun]
* @param z Redshift
* @return dn/dm [h^4 Msun^-1 Mpc^-3] 
*/
__host__ __device__ double hmf(const double& m, const double& z);

/**
* Approximation to Si(x) and Ci(x) Functions
* Same as GSL implementation, because they are not implemented in CUDA
* @param x x value
* @param si will contain Si(x)
* @param ci will contain Ci(x)
*/
__host__ __device__  void SiCi(double x, double& si, double& ci);

/**
* @brief r200 Radius of NFW profile
* 
* @param m Halo mass [h^-1 Msun]
* @param z Redshift
* @return r_200 [h^-1 Mpc]
*/
__host__ __device__ double r_200(const double&m, const double&z);

/**
* @brief Fouriertransform of normalized, truncated NFW profile (truncated at r_200)
* 
* @param k Wave vector [h Mpc^-1]
* @param m Halo mass [h^-1 Msun]
* @param z Redshift
*/
__host__ __device__ double u_NFW(const double& k, const double& m, const double& z);


/**
* @brief Concentration-mass/redshift relation of halos. Currently Duffy+(2008)
* 
* @param m Halo mass [h^-1 Msun]
* @param z Redshift
*/
__host__ __device__ double concentration(const double& m, const double& z);

/**
* @brief Matter variance on mass scale m
* I.e. Convolution of P(k) with spherical windowfunction of radius r(m)
* @param m Halo mass [h^-1 Msun]
* @return double 
*/
double sigma2(const double& m);

/**
* @brief Integrand for sigma2, wrapper for cubature
* See https://github.com/stevengj/cubature for documentation
* @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
* @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
* @param vars Array containing integration variables
* @param container Pointer to ApertureStatisticsCovarianceContainer instance
* @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
* @param value Value of integral
* @return 0 on success
*/
int integrand_sigma2(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value);

/**
* @brief calculates sigma2 for various masses, puts them into the sigma2_array and copies them to device
* 
*/
void setSigma2();

/**
* @brief Calculates derivative of sigma² with mass m
* 
* @param m mass [h^-1 Msun]
* @return double [h/Msun]
*/
double dSigma2dm(const double& m);

/**
* @brief Integrand for dSigma2dm, wrapper for cubature
* See https://github.com/stevengj/cubature for documentation
* @param ndim Number of dimensions of integral (automatically determined by integration). Exception is thrown if this is not as expected.
* @param npts Number of integration points that are evaluated at the same time (automatically determined by integration)
* @param vars Array containing integration variables
* @param container Pointer to ApertureStatisticsCovarianceContainer instance
* @param fdim Dimensions of integral output (here: 1, automatically assigned by integration). Exception is thrown if this is not as expected
* @param value Value of integral
* @return 0 on success
*/
int integrand_dSigma2dm(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value);


/**
* @brief calculates dsigma2/dm for various masses, puts them into the sigma2_array and copies them to device
* 
*/
void setdSigma2dm();

/**
* @brief Get sigma2 at the mass m and redshift z. Uses linear interpolation between entries in sigma2 array
* @warning sigma2 array needs to be set beforehand
* 
* @param m mass [h^-1 Msun]
* @param z redshift
* @return double
*/
__host__ __device__ double get_sigma2(const double& m, const double& z);


/**
* @brief Get dsigma2/dm at the mass m and redshift z. Uses linear interpolation between entries in sigma2 array
* @warning sigma2 array needs to be set beforehand
* 
* @param m mass [h^-1 Msun]
* @param z redshift
* @return double
*/
__host__ __device__ double get_dSigma2dm(const double& m, const double& z);


/**
* @brief 1-Halo Term integrand for 2D-Trispectrum. Needs to be integrated over mass m and redshift z to give total 2D-Trispectrum
* 
* @param m mass [h^-1 Msun]
* @param z redshift
* @param l1 ell1 [1/rad]
* @param l2 ell2 [1/rad]
* @param l3 ell3 [1/rad]
* @param l4 ell4 [1/rad]
* @return Integrand for Projected trispectrum (1-halo term only) 
*/
__device__ double trispectrum_integrand(double m, double z, double l1, double l2, 
double l3, double l4);

/**
* @brief 1-Halo Term integrand for 3D-Trispectrum. Needs to be integrated over mass m and redshift z to give total 3D-Trispectrum
* 
* @param m mass [h^-1 Msun]
* @param z redshift
* @param k1 k1 [h/Mpc]
* @param k2 k2 [h/Mpc]
* @param k3 k3 [h/Mpc]
* @param k4 k4 [h/Mpc]
* @return Integrand for trispectrum (1-halo term only) 
*/
__device__ double trispectrum_3D_integrand(double m, double z, double k1, double k2, double k3, double k4);

/**
* @brief 1-Halo Term of projected Trispectrum with performed Limber Integration. Needs to be integrated over mass m for total 2D-Trispectrum
* Integrates over redshift from a to b using GQ on device
*
* @param a Minimal redshift
* @param b Maximal redshift
* @param m mass [h^-1 Msun]
* @param l1 ell1 [1/rad]
* @param l2 ell2 [1/rad]
* @param l3 ell3 [1/rad]
* @param l4 ell4 [1/rad]
* @return Integrand for Projected trispectrum (1-halo term only) 
*/
__device__ double trispectrum_limber_integrated(double a, double b, double m, double l1, double l2, double l3, double l4);


/**
* @brief 1-Halo Term integrand for 2D-Pentaspectrum. Needs to be integrated over mass m and redshift z to give total 2D-Pentaspectrum
* 
* @param m mass [h^-1 Msun]
* @param z redshift
* @param l1 ell1 [1/rad]
* @param l2 ell2 [1/rad]
* @param l3 ell3 [1/rad]
* @param l4 ell4 [1/rad]
* @param l5 ell5 [1/rad]
* @param l6 ell6 [1/rad]
* @return Integrand for Projected Pentaspectrum (1-halo term only) 
*/
__device__ double pentaspectrum_integrand(double m, double z, double l1, double l2, double l3, double l4, double l5, double l6);


/**
* @brief 1-Halo Term of projected Pentaspectrum with performed Limber Integration. Needs to be integrated over mass m for total 2D-Pentaspectrum
* Integrates over redshift from a to b using GQ on device
*
* @param a Minimal redshift
* @param b Maximal redshift
* @param m mass [h^-1 Msun]
* @param l1 ell1 [1/rad]
* @param l2 ell2 [1/rad]
* @param l3 ell3 [1/rad]
* @param l4 ell4 [1/rad]
* @param l5 ell5 [1/rad]
* @param l6 ell6 [1/rad]
* @return Integrand for Projected Pentaspectrum (1-halo term only) 
*/
__device__ double pentaspectrum_limber_integrated(double a, double b, double m, double l1, double l2, double l3, double l4, double l5, double l6);


/**
* @brief 1-Halo Term of projected Pentaspectrum with performed Limber and Mass Integration. 
* Integrates over redshift from zmin to zmax and over log(M) from logMmin to logMmax using GQ on device
*
* @param zmin Minimal redshift
* @param zmax Maximal redshift
* @param logMmin log(minimal mass) [log(h^-1 Msun)]
* @param logMmax log(maximal mass) [log(h^-1 Msun)]
* @param l1 ell1 [1/rad]
* @param l2 ell2 [1/rad]
* @param l3 ell3 [1/rad]
* @param l4 ell4 [1/rad]
* @param l5 ell5 [1/rad]
* @param l6 ell6 [1/rad]
* @return Integrand for Projected Pentaspectrum (1-halo term only) 
*/
__device__ double pentaspectrum_limber_mass_integrated(double zmin, double zmax, double logMmin, double logMmax, double l1, double l2, double l3, double l4, double l5, double l6);


/// FUNCTIONS BELOW ARE FOR TESTING THE HALOMODEL

__device__ double powerspectrum_integrand(double m, double z, double l);

__device__ double powerspectrum_limber_integrated(double a, double b, double m, double l);

int integrand_Powerspectrum(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);


__global__ void integrand_Powerspectrum_kernel(const double* vars, unsigned ndim, int npts, double thetal, double* value);


double Powerspectrum(const double& l);


int integrand_Trispectrum(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);


__global__ void integrand_Trispectrum_kernel(const double* vars, unsigned ndim, int npts, double l1, double l2, double l3, double l4, double* value);


double Trispectrum(const double& l1, const double& l2, const double& l3, const double& l4);


int integrand_Trispectrum_3D(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);


__global__ void integrand_Trispectrum_3D_kernel(const double* vars, unsigned ndim, int npts, double k1, double k2, double k3, double k4, double z, double* value);


double Trispectrum_3D(const double& k1, const double& k2, const double& k3, const double& k4, const double& z);


double Pentaspectrum(const double& l1, const double& l2, const double& l3, const double& l4, const double& l5, const double& l6);


int integrand_Pentaspectrum(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value);


__global__ void integrand_Pentaspectrum_kernel(const double* vars, unsigned ndim, int npts, double l1, double l2, double l3, double l4, double l5, double l6, double* value);


struct SigmaContainer
{
  double R;
  double dR;
};

struct PowerspecContainer
{
double l;
};

struct TrispecContainer
{
double l1;
double l2;
double l3;
double l4;

};


struct TrispecContainer3D
{
double k1;
double k2;
double k3;
double k4;
double z;
};

struct PentaspecContainer
{
double l1;
double l2;
double l3;
double l4;
double l5;
double l6;
};