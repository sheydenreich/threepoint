const double logMmin=9;
const double logMmax=17;
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


  double sigma2(const double& m);
  int integrand_sigma2(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value);

  void setSigma2();

  double dSigma2dm(const double& m);
  int integrand_dSigma2dm(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value);
  
  void setdSigma2dm();

  __host__ __device__ double get_sigma2(const double& m, const double& z);
  __host__ __device__ double get_dSigma2dm(const double& m, const double& z);

  __device__ double trispectrum_integrand(double m, double z, double l1, double l2, 
    double l3, double l4);

__device__ double trispectrum_limber_integrated(double a, double b, double m, double l1, double l2, double l3, double l4);

__device__ double pentaspectrum_integrand(double m, double z, double l1, double l2, 
  double l3, double l4, double l5, double l6);

__device__ double pentaspectrum_limber_integrated(double a, double b, double m, double l1, double l2, double l3, double l4, double l5, double l6);


    struct SigmaContainer
{
   double R;
   double dR;
};