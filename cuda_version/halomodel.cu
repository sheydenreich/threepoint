#include "halomodel.cuh"
#include "cuda_helpers.cuh"
#include "bispectrum.cuh"

#include "cubature.h"

#include <iostream>


__constant__ double devLogMmin, devLogMmax;
int __constant__ dev_n_mbins;


__constant__ double dev_sigma2_array[n_mbins]; 
__constant__ double dev_dSigma2dm_array[n_mbins]; 
double sigma2_array[n_mbins]; 
double dSigma2dm_array[n_mbins]; 


void initHalomodel()
{
    copyConstants();
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(devLogMmin, &logMmin, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(devLogMmax, &logMmax, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_mbins, &n_mbins, sizeof(int)));
    setSigma2();
    std::cerr<<"Finished calculating sigma²(m)"<<std::endl;
    setdSigma2dm();
    std::cerr<<"Finished calculating dSigma²dm"<<std::endl;
    std::cerr<<"Finished all initializations"<<std::endl;
}


__host__ __device__ double hmf(const double& m, const double& z)
{
    double A=0.322;
    double q=0.707;
    double p=0.3;


    
    // Get sigma^2(m, z)
    double sigma2=get_sigma2(m, z);
    //printf("%f %f\n", m, sigma2);
  
    // Get dsigma^2/dm
    double dsigma2=get_dSigma2dm(m, z);
  
    // Get critical density contrast and mean density
#ifdef __CUDA_ARCH__
    double deltac=1.686*(1+0.0123*log10(dev_om_m_of_z(z)));
    double rho_mean=2.7754e11*dev_om;
#else
    double deltac=1.686*(1+0.0123*log10(om_m_of_z(z)));
    double rho_mean=2.7754e11*cosmo.om;
#endif
     
  
    double nu=deltac*deltac/sigma2;

    double result=-rho_mean/m/sigma2*dsigma2*A*(1+pow(q*nu, -p))*sqrt(q*nu/2/M_PI)*exp(-0.5*q*nu);


    /*
     * \f $n(m,z)=-\frac{\bar{\rho}}{m \sigma} \dv{\sigma}{m} A \sqrt{2q/pi} (1+(\frac{\sigma^2}{q\delta_c^2})^p) \frac{\delta_c}{\sigma} \exp(-\frac{q\delta_c^2}{2\sigma^2})$ \f
     */
    return result;
}

__host__ __device__ void SiCi(double x, double& si, double& ci)
  {
  double x2=x*x;
  double x4=x2*x2;
  double x6=x2*x4;
  double x8=x4*x4;
  double x10=x8*x2;
  double x12=x6*x6;
  double x14=x12*x2;

  if(x<4)
    { 
      
      double a=1-4.54393409816329991e-2*x2+1.15457225751016682e-3*x4
	-1.41018536821330254e-5*x6+9.43280809438713025e-8*x8
	-3.53201978997168357e-10*x10+7.08240282274875911e-13*x12
	-6.05338212010422477e-16*x14;

      double b=1+1.01162145739225565e-2*x2+4.99175116169755106e-5*x4
	+1.55654986308745614e-7*x6+3.28067571055789734e-10*x8
	+4.5049097575386581e-13*x10+3.21107051193712168e-16*x12;

      si=x*a/b;

      double gamma=0.5772156649;
      a=-0.25+7.51851524438898291e-3*x2-1.27528342240267686e-4*x4
	+1.05297363846239184e-6*x6-4.68889508144848019e-9*x8
	+1.06480802891189243e-11*x10-9.93728488857585407e-15*x12;
      
      b=1+1.1592605689110735e-2*x2+6.72126800814254432e-5*x4
	+2.55533277086129636e-7*x6+6.97071295760958946e-10*x8
	+1.38536352772778619e-12*x10+1.89106054713059759e-15*x12
	+1.39759616731376855e-18*x14;

      ci=gamma+std::log(x)+x2*a/b;
    }
  else
    {
      double x16=x8*x8;
      double x18=x16*x2;
      double x20=x10*x10;
      double cos_x=cos(x);
      double sin_x=sin(x);

      double f=(1+7.44437068161936700618e2/x2+1.96396372895146869801e5/x4
		+2.37750310125431834034e7/x6+1.43073403821274636888e9/x8
		+4.33736238870432522765e10/x10+6.40533830574022022911e11/x12
		+4.20968180571076940208e12/x14+1.00795182980368574617e13/x16
		+4.94816688199951963482e12/x18-4.94701168645415959931e11/x20)
	/(1+7.46437068161927678031e2/x2+1.97865247031583951450e5/x4
	  +2.41535670165126845144e7/x6+1.47478952192985464958e9/x8
	  +4.58595115847765779830e10/x10+7.08501308149515401563e11/x12
	  +5.06084464593475076774e12/x14+1.43468549171581016479e13/x16
	  +1.11535493509914254097e13/x18)/x;
      
      double g=(1+8.1359520115168615e2/x2+2.35239181626478200e5/x4
		+3.12557570795778731e7/x6+2.06297595146763354e9/x8
		+6.83052205423625007e10/x10+1.09049528450362786e12/x12
		+7.57664583257834349e12/x14+1.81004487464664575e13/x16
		+6.43291613143049485e12/x18-1.36517137670871689e12/x20)/
	(1+8.19595201151451564e2/x2+2.40036752835578777e5/x4
	 +3.26026661647090822e7/x6+2.23355543278099360e9/x8
	 +7.87465017341829930e10/x10+1.39866710696414565e12/x12
	 +1.17164723371736605e13/x14+4.01839087307656620e13/x16
	 +3.99653257887490811e13/x18)/x2;

      si=0.5*M_PI-f*cos_x-g*sin_x;
      ci=f*sin_x-g*cos_x;
    };
  return;
}



__host__ __device__ double r_200(const double&m, const double&z)
{
    double rhobar = 2.7754e11; //critical density[Msun*h²/Mpc³]

#ifdef __CUDA_ARCH__
    rhobar*=dev_om;
#else
    rhobar*=cosmo.om;
#endif

    return pow(0.75/M_PI*m/rhobar/200, 1./3.);
}

__host__ __device__ double u_NFW(const double& k, const double& m, const double& z)
{
    // Get concentration
double c=concentration(m, z);

double arg1=k*r_200(m,z)/c;
double arg2=arg1*(1+c);

//printf("%f %f %f\n", r_200(m, z), m ,z);

double si1, ci1, si2, ci2;
SiCi(arg1, si1, ci1);
SiCi(arg2, si2, ci2);

double term1=sin(arg1)*(si2-si1);
double term2=cos(arg1)*(ci2-ci1);
double term3=-sin(arg1*c)/arg2;
double F=std::log(1.+c)-c/(1.+c);

//printf("%f %f %f %f\n", term1, term2, term3, F);

double result=(term1+term2+term3)/F;
return result;
}

__host__ __device__ double concentration(const double& m, const double& z)
{
    //Using Duffy+ 08 (second mass definition, all halos)
    //To Do: Bullock01 might be better!
    double A= 10.14;
    double Mpiv=2e12; //Msun h⁻1
    double B= -0.081;
    double C= -1.01;
    double result=A*pow(m/Mpiv, B)*pow(1+z, C);
    return result;
}

double sigma2(const double& m)
{
    double rhobar = 2.7754e11*cosmo.om; //critical density[Msun*h²/Mpc³]

  double R=pow(0.75/M_PI*m/rhobar, 1./3.); //[Mpc/h]

  SigmaContainer container;
  container.R=R;
  
  double k_min[1]={0};
  double k_max[1]={1e12};
  double result, error;
  
  hcubature_v(1, integrand_sigma2, &container, 1, k_min, k_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
  
  return result/2/M_PI/M_PI;
}

int integrand_sigma2(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim!=1)
    {
      std::cerr<<"integrand_sigma2: Wrong fdim"<<std::endl;
      exit(1);
    };
  SigmaContainer* container = (SigmaContainer*) thisPtr;

 
  double R=container->R;

  
  #pragma omp parallel for
  for(unsigned int i=0; i<npts; i++)
    {
      double k_=k[i*ndim];
      double W=window(k_*R, 0);
      
      value[i]=k_*k_*W*W*linear_pk(k_);
    };
  
  return 0;
}


void setSigma2()
{
    double rhobar = 2.7754e11*cosmo.om; //critical density[Msun*h²/Mpc³]



    double deltaM=(logMmax-logMmin)/n_mbins;

    for(int i=0; i<n_mbins; i++)
    {


            double m=pow(10, logMmin+i*deltaM);
            double R=pow(0.75/M_PI*m/rhobar, 1./3.);
            double sigma = sigma2(m);
            sigma2_array[i]=sigma;
    }
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma2_array, &sigma2_array, n_mbins*sizeof(double)));
    std::cerr<<"Finished precalculating sigma2(m)"<<std::endl;
}




double dSigma2dm(const double& m)
{
    double rhobar = 2.7754e11; //critical density[Msun*h²/Mpc³]

#ifdef __CUDA_ARCH__
    rhobar*=dev_om;
#else
    rhobar*=cosmo.om;
#endif

  double R=pow(0.75/M_PI*m/rhobar, 1./3.); //[Mpc/h]
  double dR=pow(0.75/M_PI/rhobar, 1./3.)*pow(m, -2./3.)/3.; //[Mpc/h]


  SigmaContainer container;
  container.R=R;
  container.dR=dR;
  
  
  double k_min[1]={0};
  double k_max[1]={1e12};
  double result, error;
  
  hcubature_v(1, integrand_dSigma2dm, &container, 1, k_min, k_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

  return result/M_PI/M_PI;
}


int integrand_dSigma2dm(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim!=1)
    {
      std::cerr<<"integrand_dSigma2dm: Wrong fdim"<<std::endl;
      exit(1);
    };
  SigmaContainer* container = (SigmaContainer*) thisPtr;

 
  double R=container->R;
  double dR=container->dR;
 
  
  #pragma omp parallel for
  for(unsigned int i=0; i<npts; i++)
    {
      double k_=k[i*ndim];
      double W=window(k_*R,0);
      double Wprime=window(k_*R, 4);
      
      value[i]=k_*k_*W*Wprime*dR*linear_pk(k_)*k_;
    };
  
  return 0;
}

void setdSigma2dm()
{


    double deltaM=(logMmax-logMmin)/n_mbins;

    for(int i=0; i<n_mbins; i++)
    {

            double m=pow(10, logMmin+i*deltaM);
        
            dSigma2dm_array[i] = dSigma2dm(m);
            //std::cerr<<i<<" "<<dSigma2dm_[i]<<std::endl;

    }
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_dSigma2dm_array, &dSigma2dm_array, n_mbins*sizeof(double)));
    std::cerr<<"Finished precalculating dsigma2/dm"<<std::endl;
}

__host__ __device__ double get_sigma2(const double& m, const double& z)
{
    double logM=log10(m);


#ifdef __CUDA_ARCH__
    double didx = (logM-devLogMmin) / (devLogMmax-devLogMmin) * (dev_n_mbins - 1);
    int idx = didx;
    didx = didx - idx;
    if (idx == dev_n_mbins - 1)
    {
      idx = dev_n_mbins - 2;
      didx = 1.;
    }
    double sigma=dev_sigma2_array[idx]*(1-didx)+dev_sigma2_array[idx+1]*didx;


    didx = z / dev_z_max * (dev_n_redshift_bins - 1);
    idx = didx;
    didx = didx - idx;
    if (idx == dev_n_redshift_bins - 1)
    {
      idx = dev_n_redshift_bins - 2;
      didx = 1.;
    }

    double D1=dev_D1_array[idx] * (1 - didx) + dev_D1_array[idx + 1] * didx;
#else
    double didx = (logM-logMmin) / (logMmax-logMmin) * (n_mbins - 1);
    int idx = didx;
    didx = didx - idx;
    if (idx == n_mbins - 1)
    {
    idx = n_mbins - 2;
    didx = 1.;
    }
    double sigma=sigma2_array[idx]*(1-didx)+sigma2_array[idx+1]*didx;


    didx = z / z_max * (n_redshift_bins - 1);
    idx = didx;
    didx = didx - idx;
    if (idx == n_redshift_bins - 1)
    {
    idx = n_redshift_bins - 2;
    didx = 1.;
    }

    double D1=D1_array[idx] * (1 - didx) + D1_array[idx + 1] * didx;
#endif


    sigma*=D1*D1;

    return sigma;
}


__host__ __device__ double get_dSigma2dm(const double& m, const double& z)
{
    double logM=log10(m);

#ifdef __CUDA_ARCH__
    double didx = (logM-devLogMmin) / (devLogMmax-devLogMmin) * (dev_n_mbins - 1);
    int idx = didx;
    didx = didx - idx;
    if (idx == dev_n_mbins - 1)
    {
      idx = dev_n_mbins - 2;
      didx = 1.;
    }
    double dsigma=dev_dSigma2dm_array[idx]*(1-didx)+dev_dSigma2dm_array[idx+1]*didx;

    
    didx = z / dev_z_max * (dev_n_redshift_bins - 1);
    idx = didx;
    didx = didx - idx;
    if (idx == dev_n_redshift_bins - 1)
    {
      idx = dev_n_redshift_bins - 2;
      didx = 1.;
    }

    double D1=dev_D1_array[idx] * (1 - didx) + dev_D1_array[idx + 1] * didx;
#else
    double didx = (logM-logMmin) / (logMmax-logMmin) * (n_mbins - 1);
    int idx = didx;
    didx = didx - idx;
    if (idx == n_mbins - 1)
    {
    idx = n_mbins - 2;
    didx = 1.;
    }
    double dsigma=dSigma2dm_array[idx]*(1-didx)+dSigma2dm_array[idx+1]*didx;


    didx = z / z_max * (n_redshift_bins - 1);
    idx = didx;
    didx = didx - idx;
    if (idx == n_redshift_bins - 1)
    {
    idx = n_redshift_bins - 2;
    didx = 1.;
    }

    double D1=D1_array[idx] * (1 - didx) + D1_array[idx + 1] * didx;
#endif
    dsigma*=D1*D1;
    return dsigma;
}


__device__ double trispectrum_integrand(double m, double z, double l1, double l2, double l3, double l4)
{
    double didx = z / dev_z_max * (n_redshift_bins - 1);
    int idx = didx;
    didx = didx - idx;
    if (idx == n_redshift_bins - 1)
    {
      idx = n_redshift_bins - 2;
      didx = 1.;
    }
    double g= dev_g_array[idx] * (1 - didx) + dev_g_array[idx + 1] * didx;
    double chi = dev_f_K_array[idx] * (1-didx) + dev_f_K_array[idx + 1] * didx;
    double rhobar = 2.7754e11; //critical density[Msun*h²/Mpc³]
    rhobar*=dev_om;


    double result=hmf(m, z)*g*g*g*g/chi/chi;
    result*=dev_c_over_H0/dev_E(z);
    result*=u_NFW(l1/chi, m, z)*u_NFW(l2/chi, m, z)*u_NFW(l3/chi, m, z)*u_NFW(l4/chi, m, z);
    result*=m*m*m*m/rhobar/rhobar/rhobar/rhobar;
    result*=pow(1.5*dev_om/dev_c_over_H0/dev_c_over_H0, 4); //Prefactor in h^8

    if(!isfinite(result))
    {
      printf("%e %e %e %e %e %e %e\n", u_NFW(l1/chi, m, z),
      u_NFW(l2/chi, m, z), u_NFW(l3/chi, m, z), u_NFW(l4/chi, m, z) ,l4, chi,result);
    }

    //printf("%f %e %e %e %e %e %e\n",z, m, l1, l2, l3, l4, result);

    return result;
}
