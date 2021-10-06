#include "bispectrum.cuh"
#include "cuda_helpers.cuh"

#include <iostream>
#include <vector>

// Gaussian Quadrature stuff
double A96[48]={                   /* abscissas for 96-point Gauss quadrature */
    0.016276744849603,0.048812985136050,0.081297495464426,0.113695850110666,
    0.145973714654897,0.178096882367619,0.210031310460567,0.241743156163840,
    0.273198812591049,0.304364944354496,0.335208522892625,0.365696861472314,
    0.395797649828909,0.425478988407301,0.454709422167743,0.483457973920596,
    0.511694177154668,0.539388108324358,0.566510418561397,0.593032364777572,
    0.618925840125469,0.644163403784967,0.668718310043916,0.692564536642172,
    0.715676812348968,0.738030643744400,0.759602341176648,0.780369043867433,
    0.800308744139141,0.819400310737932,0.837623511228187,0.854959033434602,
    0.871388505909297,0.886894517402421,0.901460635315852,0.915071423120898,
    0.927712456722309,0.939370339752755,0.950032717784438,0.959688291448743,
    0.968326828463264,0.975939174585137,0.982517263563015,0.988054126329624,
    0.992543900323763,0.995981842987209,0.998364375863182,0.999689503883231};

  double W96[48]={                     /* weights for 96-point Gauss quadrature */
    0.032550614492363,0.032516118713869,0.032447163714064,0.032343822568576,
    0.032206204794030,0.032034456231993,0.031828758894411,0.031589330770727,
    0.031316425596861,0.031010332586314,0.030671376123669,0.030299915420828,
    0.029896344136328,0.029461089958168,0.028994614150555,0.028497411065085,
    0.027970007616848,0.027412962726029,0.026826866725592,0.026212340735672,
    0.025570036005349,0.024900633222484,0.024204841792365,0.023483399085926,
    0.022737069658329,0.021966644438744,0.021172939892191,0.020356797154333,
    0.019519081140145,0.018660679627411,0.017782502316045,0.016885479864245,
    0.015970562902562,0.015038721026995,0.014090941772315,0.013128229566962,
    0.012151604671088,0.011162102099839,0.010160770535008,0.009148671230783,
    0.008126876925699,0.007096470791154,0.006058545504236,0.005014202742928,
    0.003964554338445,0.002910731817935,0.001853960788947,0.000796792065552};

// Values that are put in constant memory on device
// These things stay the same for a kernel run
// Warning: They are read-only for the GPU!

__constant__ double dev_A96[48];// Abscissas for Gauss-Quadrature
__constant__ double dev_W96[48];// Weights for Gauss-Quadrature

__constant__ double dev_eps=1.0e-4; //< Integration accuracy
const double eps=1.0e-4;

__constant__ double dev_dz,dev_z_max; //Redshift bin and maximal redshift
double dz, z_max;

// Cosmological Parameters
__constant__ double dev_h,dev_sigma8,dev_omb,dev_omc,dev_ns,dev_w,dev_om,dev_ow,dev_norm;
double h,sigma8,omb,omc,ns,w,om,ow,norm_P;

__constant__ int dev_n_redshift_bins; // Number of redshift bins
//const int n_redshift_bins=512; // Number of redshift bins (hardcoded here!)


__constant__ double dev_H0_over_c = 100./299792.; //Hubble constant/speed of light [h s/m]
const double c_over_H0 = 2997.92;
__constant__ double dev_c_over_H0 = 2997.92; //Speed of light / Hubble constant [h^-1 m/s]

__constant__ double dev_f_K_array[n_redshift_bins]; // Array for comoving distance
__constant__ double dev_g_array[n_redshift_bins]; // Array for lensing efficacy g
__constant__ double dev_n_eff_array[n_redshift_bins]; // Array for n_eff
__constant__ double dev_r_sigma_array[n_redshift_bins]; // Array for r(sigma)
__constant__ double dev_D1_array[n_redshift_bins]; // Array for growth factor
__constant__ double dev_ncur_array[n_redshift_bins]; // Array for C in Halofit


__device__ double bkappa(double ell1, double ell2, double ell3)
{
  if(ell1 == 0 || ell2 == 0 || ell3 == 0) return 0; //WARNING! THIS MIGHT SCREW WITH THE INTEGRATION ROUTINE!
  double prefactor = 27./8. * pow(dev_om,3) * pow(dev_H0_over_c,5);
  return prefactor*GQ96_of_bdelta(0, dev_z_max, ell1, ell2, ell3);
}

__device__ double GQ96_of_bdelta(double a,double b, double ell1, double ell2, double ell3)
{
  double cx=(a+b)/2;
  double dx=(b-a)/2;
  double q=0;
  for(int i=0;i<48;i++)
    q+=dev_W96[i]*(integrand_bkappa(cx-dx*dev_A96[i],ell1, ell2, ell3)+integrand_bkappa(cx+dx*dev_A96[i],ell1, ell2, ell3));
  return q*dx;
}

__device__ double integrand_bkappa(double z, double ell1, double ell2, double ell3)
{
    if(z<1.0e-7) return 0.;
    if(ell1<=1.0e-10||ell2<=1.0e-10||ell3<=1.0e-10)
    {
      return 0;
    } 

    double didx = z/dev_z_max*(dev_n_redshift_bins-1);
    int idx = didx;
    didx = didx - idx;
    if(idx==dev_n_redshift_bins-1){
        idx = dev_n_redshift_bins-2;
        didx = 1.;
    }
    double g_value = g_interpolated(idx,didx);
    double f_K_value = f_K_interpolated(idx,didx);
    double result = pow(g_value*(1.+z),3)*bispec(ell1/f_K_value,ell2/f_K_value,ell3/f_K_value,z,idx,didx)/f_K_value/dev_E(z);
    if(isnan(result))
    {
      //      printf("%lf, %lf, %lf \n", z, dev_z_max, dev_n_redshift_bins);
      //      printf("%lf, %lf \n", idx, didx);
      printf("nan in bispec! %lf, %lf, %lf, %lf, %.3f \n",f_K_value,ell1,ell2,ell3,z);
      return 0;
    }
    return result;
}

__device__ double g_interpolated(int idx, double didx)
{
    return dev_g_array[idx]*(1-didx) + dev_g_array[idx+1]*didx;
}

__device__ double f_K_interpolated(int idx, double didx)
{
    return dev_f_K_array[idx]*(1-didx) + dev_f_K_array[idx+1]*didx;
}


__device__ double bispec(double k1, double k2, double k3, double z, int idx, double didx)
{
  int i,j;
  double q[4],qt,logsigma8z,r1,r2;
  double an,bn,cn,en,fn,gn,hn,mn,nn,pn,alphan,betan,mun,nun,BS1h,BS3h,PSE[4];
  double r_sigma,n_eff,D1;
  compute_coefficients(idx, didx, &D1, &r_sigma, &n_eff);

  if(z>10.) return bispec_tree(k1,k2,k3,z,D1);


  q[1]=k1*r_sigma, q[2]=k2*r_sigma, q[3]=k3*r_sigma;  // dimensionless wavenumbers

  // sorting q[i] so that q[1]>=q[2]>=q[3]
  for(i=1;i<=3;i++){
    for(j=i+1;j<=3;j++){
      if(q[i]<q[j]){
	qt=q[j];
        q[j]=q[i];
	q[i]=qt;
      }}}
  r1=q[3]/q[1], r2=(q[2]+q[3]-q[1])/q[1];   // Eq.(B8)

  q[1]=k1*r_sigma, q[2]=k2*r_sigma, q[3]=k3*r_sigma;
  logsigma8z=log10(D1*dev_sigma8);

  // 1-halo term parameters in Eq.(B7)
  an=pow(10.,-2.167-2.944*logsigma8z-1.106*pow(logsigma8z,2)-2.865*pow(logsigma8z,3)-0.310*pow(r1,pow(10.,0.182+0.57*n_eff)));
  bn=pow(10.,-3.428-2.681*logsigma8z+1.624*pow(logsigma8z,2)-0.095*pow(logsigma8z,3));
  cn=pow(10.,0.159-1.107*n_eff);
  alphan=pow(10.,-4.348-3.006*n_eff-0.5745*pow(n_eff,2)+pow(10.,-0.9+0.2*n_eff)*pow(r2,2));
  if(alphan>1.-(2./3.)*dev_ns) alphan=1.-(2./3.)*dev_ns;
  betan=pow(10.,-1.731-2.845*n_eff-1.4995*pow(n_eff,2)-0.2811*pow(n_eff,3)+0.007*r2);

  // 1-halo term bispectrum in Eq.(B4)
  BS1h=1.;
  for(i=1;i<=3;i++){
    BS1h*=1./(an*pow(q[i],alphan)+bn*pow(q[i],betan))/(1.+1./(cn*q[i]));
  }

  // 3-halo term parameters in Eq.(B9)
  fn=pow(10.,-10.533-16.838*n_eff-9.3048*pow(n_eff,2)-1.8263*pow(n_eff,3));
  gn=pow(10.,2.787+2.405*n_eff+0.4577*pow(n_eff,2));
  hn=pow(10.,-1.118-0.394*n_eff);
  mn=pow(10.,-2.605-2.434*logsigma8z+5.71*pow(logsigma8z,2));
  nn=pow(10.,-4.468-3.08*logsigma8z+1.035*pow(logsigma8z,2));
  mun=pow(10.,15.312+22.977*n_eff+10.9579*pow(n_eff,2)+1.6586*pow(n_eff,3));
  nun=pow(10.,1.347+1.246*n_eff+0.4525*pow(n_eff,2));
  pn=pow(10.,0.071-0.433*n_eff);
  en=pow(10.,-0.632+0.646*n_eff);

  for(i=1;i<=3;i++){
    PSE[i]=(1.+fn*pow(q[i],2))/(1.+gn*q[i]+hn*pow(q[i],2))*pow(D1,2)*dev_linear_pk(q[i]/r_sigma)+1./(mn*pow(q[i],mun)+nn*pow(q[i],nun))/(1.+pow(pn*q[i],-3));  // enhanced P(k) in Eq.(B6)
  }

  // 3-halo term bispectrum in Eq.(B5)
  BS3h=2.*(F2(k1,k2,k3,z,D1,r_sigma)*PSE[1]*PSE[2]+F2(k2,k3,k1,z,D1,r_sigma)*PSE[2]*PSE[3]+F2(k3,k1,k2,z,D1,r_sigma)*PSE[3]*PSE[1]);
  for(i=1;i<=3;i++) BS3h*=1./(1.+en*q[i]);
  // if (BS1h+BS3h>=0)
  return BS1h+BS3h;
  // else return 0;
}

__device__ void compute_coefficients(int idx, double didx, double *D1, double *r_sigma, double *n_eff)
{
     *D1 = dev_D1_array[idx]*(1-didx) + dev_D1_array[idx+1]*didx;
     *r_sigma = dev_r_sigma_array[idx]*(1-didx) + dev_r_sigma_array[idx+1]*didx;
     *n_eff = dev_n_eff_array[idx]*(1-didx) + dev_n_eff_array[idx+1]*didx;
}

__device__ double P_k_nonlinear(double k, double z){
  printf("Warning! This is the bugged version. Not fixed yet.");
  /* get the interpolation coefficients */
  double didx = z/dev_z_max*(n_redshift_bins-1);
  int idx = didx;
  didx = didx - idx;
  if(idx==n_redshift_bins-1){
      idx = n_redshift_bins-2;
      didx = 1.;
  }

  double r_sigma,n_eff,D1;
  compute_coefficients(idx, didx, &D1, &r_sigma, &n_eff);

  double a,b,c,gam,alpha,beta,xnu,y,ysqr,ph,pq,f1,f2,f3;
  double f1a,f2a,f3a,f1b,f2b,f3b,frac;
  double plin,delta_nl;
  double scalefactor,om_m,om_v;
  double nsqr,ncur;

  f1   = pow(dev_om, -0.0307);
  f2   = pow(dev_om, -0.0585);
  f3   = pow(dev_om, 0.0743);


  nsqr = n_eff*n_eff;
  ncur = dev_ncur_array[idx]*(1-didx) + dev_ncur_array[idx+1]*didx; //interpolate ncur
  // ncur = (n_eff+3)*(n_eff+3)+4.*pow(D1,2)*sigmam(r_sigma,3)/pow(sigmam(r_sigma,1),2);
  // printf("n_eff, ncur, knl = %.3f, %.3f, %.3f \n",n_eff,ncur,1./r_sigma);

  // ncur = 0.;

  if(abs(dev_om+dev_ow-1)>1e-4)
  {
    printf("Warning: omw as a function of redshift only implemented for flat universes yet!");
  }

  scalefactor=1./(1.+z);

  om_m = dev_om/(dev_om+dev_ow*pow(scalefactor,-3.*dev_w));   // Omega matter at z
  om_v = 1.-om_m; //omega lambda at z. TODO: implement for non-flat Universes

		f1a=pow(om_m,(-0.0732));
		f2a=pow(om_m,(-0.1423));
		f3a=pow(om_m,(0.0725));
		f1b=pow(om_m,(-0.0307));
		f2b=pow(om_m,(-0.0585));
		f3b=pow(om_m,(0.0743));
		frac=om_v/(1.-om_m);
		f1=frac*f1b + (1-frac)*f1a;
		f2=frac*f2b + (1-frac)*f2a;
		f3=frac*f3b + (1-frac)*f3a;

  a = 1.5222 + 2.8553*n_eff + 2.3706*nsqr + 0.9903*n_eff*nsqr
      + 0.2250*nsqr*nsqr - 0.6038*ncur + 0.1749*om_v*(1.0 + dev_w);
  a = pow(10.0, a);
  b = pow(10.0, -0.5642 + 0.5864*n_eff + 0.5716*nsqr - 1.5474*ncur + 0.2279*om_v*(1.0 + dev_w));
  c = pow(10.0, 0.3698 + 2.0404*n_eff + 0.8161*nsqr + 0.5869*ncur);
  gam = 0.1971 - 0.0843*n_eff + 0.8460*ncur;
  alpha = fabs(6.0835 + 1.3373*n_eff - 0.1959*nsqr - 5.5274*ncur);
  beta  = 2.0379 - 0.7354*n_eff + 0.3157*nsqr + 1.2490*n_eff*nsqr + 0.3980*nsqr*nsqr - 0.1682*ncur;
  xnu   = pow(10.0, 5.2105 + 3.6902*n_eff);

  plin = dev_linear_pk(k)*D1*D1*k*k*k/(2*M_PI*M_PI);


  y = k*r_sigma;
  ysqr = y*y;
  ph = a*pow(y,f1*3)/(1+b*pow(y,f2)+pow(f3*c*y,3-gam));
  ph = ph/(1+xnu/ysqr);
  pq = plin*pow(1+plin,beta)/(1+plin*alpha)*exp(-y/4.0-ysqr/8.0);
  
  delta_nl = pq + ph;

  return (2*M_PI*M_PI*delta_nl/(k*k*k));
}


__device__ double dev_linear_pk(double k) 
{
  double pk,delk,alnu,geff,qeff,L,C;
  k*=dev_h;  // unit conversion from [h/Mpc] to [1/Mpc]

  double fc=dev_omc/dev_om;
  double fb=dev_omb/dev_om;
  double theta=2.728/2.7;
  double pc=0.25*(5.0-sqrt(1.0+24.0*fc));
  double omh2=dev_om*dev_h*dev_h;
  double ombh2=dev_omb*dev_h*dev_h;
  double zeq=2.5e+4*omh2/pow(theta,4);
  double b1=0.313*pow(omh2,-0.419)*(1.0+0.607*pow(omh2,0.674));
  double b2=0.238*pow(omh2,0.223);
  double zd=1291.0*pow(omh2,0.251)/(1.0+0.659*pow(omh2,0.828))*(1.0+b1*pow(ombh2,b2));
  double yd=(1.0+zeq)/(1.0+zd);
  double sh=44.5*log(9.83/(omh2))/sqrt(1.0+10.0*pow(ombh2,0.75));

  alnu=fc*(5.0-2.0*pc)/5.0*(1.0-0.553*fb+0.126*fb*fb*fb)*pow(1.0+yd,-pc)
    *(1.0+0.5*pc*(1.0+1.0/(7.0*(3.0-4.0*pc)))/(1.0+yd));

  geff=omh2*(sqrt(alnu)+(1.0-sqrt(alnu))/(1.0+pow(0.43*k*sh,4)));
  qeff=k/geff*theta*theta;

  L=log(2.718281828+1.84*sqrt(alnu)*qeff/(1.0-0.949*fb));
  C=14.4+325.0/(1.0+60.5*pow(qeff,1.11));

  delk=pow(dev_norm,2)*pow(k*2997.9/dev_h,3.+dev_ns)*pow(L/(L+C*qeff*qeff),2);
  pk=2.0*M_PI*M_PI/(k*k*k)*delk;

  return dev_h*dev_h*dev_h*pk;
}


double linear_pk(double k) 
{
  double pk,delk,alnu,geff,qeff,L,C;
  k*=h;  // unit conversion from [h/Mpc] to [1/Mpc]

  double fc=omc/om;
  double fb=omb/om;
  double theta=2.728/2.7;
  double pc=0.25*(5.0-sqrt(1.0+24.0*fc));
  double omh2=om*h*h;
  double ombh2=omb*h*h;
  double zeq=2.5e+4*omh2/pow(theta,4);
  double b1=0.313*pow(omh2,-0.419)*(1.0+0.607*pow(omh2,0.674));
  double b2=0.238*pow(omh2,0.223);
  double zd=1291.0*pow(omh2,0.251)/(1.0+0.659*pow(omh2,0.828))*(1.0+b1*pow(ombh2,b2));
  double yd=(1.0+zeq)/(1.0+zd);
  double sh=44.5*log(9.83/(omh2))/sqrt(1.0+10.0*pow(ombh2,0.75));

  alnu=fc*(5.0-2.0*pc)/5.0*(1.0-0.553*fb+0.126*fb*fb*fb)*pow(1.0+yd,-pc)
    *(1.0+0.5*pc*(1.0+1.0/(7.0*(3.0-4.0*pc)))/(1.0+yd));

  geff=omh2*(sqrt(alnu)+(1.0-sqrt(alnu))/(1.0+pow(0.43*k*sh,4)));
  qeff=k/geff*theta*theta;

  L=log(2.718281828+1.84*sqrt(alnu)*qeff/(1.0-0.949*fb));
  C=14.4+325.0/(1.0+60.5*pow(qeff,1.11));

  delk=pow(norm_P,2)*pow(k*2997.9/h,3.+ns)*pow(L/(L+C*qeff*qeff),2);
  pk=2.0*M_PI*M_PI/(k*k*k)*delk;

  return h*h*h*pk;
}

__device__ double bispec_tree(double k1, double k2, double k3, double z, double D1)  // tree-level BS [(Mpc/h)^6]
{
  return pow(D1,4)*2.*(F2_tree(k1,k2,k3)*dev_linear_pk(k1)*dev_linear_pk(k2)
		      +F2_tree(k2,k3,k1)*dev_linear_pk(k2)*dev_linear_pk(k3)
		      +F2_tree(k3,k1,k2)*dev_linear_pk(k3)*dev_linear_pk(k1));
}

__device__ double F2(double k1, double k2, double k3, double z, double D1, double r_sigma)
{
  double a,q[4],dn,omz,logsigma8z;

  q[3]=k3*r_sigma;

  logsigma8z=log10(D1*dev_sigma8);
  a=1./(1.+z);
  omz=dev_om/(dev_om+dev_ow*pow(a,-3.*dev_w));   // Omega matter at z

  dn=pow(10.,-0.483+0.892*logsigma8z-0.086*omz);

  return F2_tree(k1,k2,k3)+dn*q[3];
}

__device__ double F2_tree(double k1, double k2, double k3)  // F2 kernel in tree level
{
  double costheta12=0.5*(k3*k3-k1*k1-k2*k2)/(k1*k2);
  return (5./7.)+0.5*costheta12*(k1/k2+k2/k1)+(2./7.)*costheta12*costheta12;
}

double get_om()
{
  double om;
  CudaSafeCall(cudaMemcpyFromSymbol(&om,dev_om,sizeof(double)));
  return om;
}

void set_cosmology(cosmology cosmo, double dz_, bool nz_from_file, std::vector<double>* nz)
{

  if(nz_from_file && nz==nullptr)
    {
      std::cerr<<"set_cosmology: expected n(z) from file, but values not provided"<<std::endl;
      exit(1);
    };
  
  //set cosmology
  h = cosmo.h;
  sigma8 = cosmo.sigma8;
  omb = cosmo.omb;
  omc = cosmo.omc;
  ns = cosmo.ns;
  w = cosmo.w;
  om = cosmo.om;
  ow = cosmo.ow;
  
  // Copy Cosmological Parameters (constant memory)
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_h,&cosmo.h,sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma8,&cosmo.sigma8,sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_omb,&cosmo.omb,sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_omc,&cosmo.omc,sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_ns,&cosmo.ns,sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_w,&cosmo.w,sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_om,&cosmo.om,sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_ow,&cosmo.ow,sizeof(double)));

  // Calculate Norm and copy
  norm_P=1.0;
  norm_P=cosmo.sigma8/sigmam(8.,0);
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_norm,&norm_P,sizeof(double)));

  // Copy redshift binning
  dz=dz_;
  z_max=cosmo.zmax;
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_dz,&dz,sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_z_max,&z_max,sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_redshift_bins,&n_redshift_bins,sizeof(int)));

  // Calculate f_K(z) and g(z)
  double f_K_array[n_redshift_bins];
  double g_array[n_redshift_bins];

  //First: f_K
#pragma omp parallel for
  for(int i=0;i<n_redshift_bins;i++)
    {
      double z_now = i*dz;
      f_K_array[i] = f_K_at_z(z_now);
    };


  //Second: g
#pragma omp parallel for
    for(int i=0;i<n_redshift_bins;i++)
      {
	g_array[i] = 0;
	// perform trapezoidal integration
	for(int j=i;j<n_redshift_bins;j++)
	  {
	    double z_now = j*dz;
	    double nz_znow;
	    if(nz_from_file)
	      {
		nz_znow=nz->at(j);
	      }
	    else
	      {
		nz_znow=n_of_z(z_now);
	      };
	  if(j==i || j==n_redshift_bins-1)
	    {
	      g_array[i] += nz_znow*(f_K_array[j]-f_K_array[i])/f_K_array[j]/2;
	    }
	  else
	    {
	      g_array[i] += nz_znow*(f_K_array[j]-f_K_array[i])/f_K_array[j];
	    }
	}
	g_array[i] = g_array[i]*dz;
      }
    g_array[0] = 1.;

    // Copy f_k and g to device (constant memory)
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_f_K_array,f_K_array,n_redshift_bins*sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_g_array,g_array,n_redshift_bins*sizeof(double)));


    // Calculating Non-linear scales
    double D1_array[n_redshift_bins];
    double r_sigma_array[n_redshift_bins];
    double n_eff_array[n_redshift_bins];
    double ncur_array[n_redshift_bins];

#pragma omp parallel for
    for(int i=0; i<n_redshift_bins;i++)
      {
	double z_now = i*dz;
    
	D1_array[i]=lgr(z_now)/lgr(0.);   // linear growth factor
	r_sigma_array[i]=calc_r_sigma(D1_array[i]);  // =1/k_NL [Mpc/h] in Eq.(B1)
  double d1 = -2.*pow(D1_array[i]*sigmam(r_sigma_array[i],2),2);
	n_eff_array[i]=-3.+2.*pow(D1_array[i]*sigmam(r_sigma_array[i],2),2);   // n_eff in Eq.(B2)
  ncur_array[i] = d1*d1+4.*sigmam(r_sigma_array[i],3)*pow(D1_array[i],2);

      }

    // Copy non-linear scales to device
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_D1_array,D1_array,n_redshift_bins*sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_r_sigma_array,r_sigma_array,n_redshift_bins*sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_eff_array,n_eff_array,n_redshift_bins*sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_ncur_array,ncur_array,n_redshift_bins*sizeof(double)));


}


double f_K_at_z(double z)
{
  return c_over_H0*GQ96_of_Einv(0,z);
}


double n_of_z(double z)
{
  if(z<=0 || z>=z_max) return 0;
  if(slics)
    {
      // Here the correct n(z) for Euclid-like simulations.
      return 1.7865*(pow(z,0.4710)+pow(z,0.4710*5.1843))/(pow(z,5.1843)+0.7259)/2.97653;
      // this is a different n(z), not the one used for our simulations. That one is above.
      // return pow(z,2)*exp(-pow(z/0.637,1.5))*5.80564; //normalization 5.80564 such that int_0^3 dz n(z) is 1
    }
    else
      {
        if(z>=1 && z<1+dz) return 1./dz;
        else return 0;
      }

}

double lgr(double z)  // linear growth factor at z (not normalized at z=0)
{
  int i,j,n;
  double a,a0,x,h,yp;
  double k1[2],k2[2],k3[2],k4[2],y[2],y2[2],y3[2],y4[2];

  a=1./(1.+z);
  a0=1./1100.;

  yp=-1.;
  n=10;

  for(;;){
    n*=2;
    h=(log(a)-log(a0))/n;
  
    x=log(a0);
    y[0]=1., y[1]=0.;
    for(i=0;i<n;i++){
      for(j=0;j<2;j++) k1[j]=h*lgr_func(j,x,y);

      for(j=0;j<2;j++) y2[j]=y[j]+0.5*k1[j];
      for(j=0;j<2;j++) k2[j]=h*lgr_func(j,x+0.5*h,y2);

      for(j=0;j<2;j++) y3[j]=y[j]+0.5*k2[j];
      for(j=0;j<2;j++) k3[j]=h*lgr_func(j,x+0.5*h,y3);

      for(j=0;j<2;j++) y4[j]=y[j]+k3[j];
      for(j=0;j<2;j++) k4[j]=h*lgr_func(j,x+h,y4);
      
      for(j=0;j<2;j++) y[j]+=(k1[j]+k4[j])/6.+(k2[j]+k3[j])/3.;
      x+=h;
    }

    if(fabs(y[0]/yp-1.)<0.1*eps) break;
    yp=y[0];
  }

  return a*y[0];
}


double lgr_func(int j, double la, double y[2])
{
  if(j==0) return y[1];
  if(j==1)
    {
      double g,a;
      a=exp(la);
      g=-0.5*(5.*om+(5.-3*w)*ow*pow(a,-3.*w))*y[1]-1.5*(1.-w)*ow*pow(a,-3.*w)*y[0];
      g=g/(om+ow*pow(a,-3.*w));
      return g;
    };

   // This is only reached, if j is not a valid value
  std::cerr << "lgr_func: j not a valid value. Exiting \n";
  exit(1);
}


double sigmam(double r, int j)   // r[Mpc/h]
{
  int n,i;
  double k1,k2,xx,xxp,xxpp,k,a,b,hh;

  k1=2.*M_PI/r;
  k2=2.*M_PI/r;

  xxpp=-1.0;
  for(;;){
    k1=k1/10.0;
    k2=k2*2.0;

    a=log(k1),b=log(k2);

    xxp=-1.0;
    n=2;
    for(;;){
      n=n*2;
      hh=(b-a)/(double)n;

      xx=0.;
      for(i=1;i<n;i++){
  k=exp(a+hh*i);
  if(j<3)	xx+=k*k*k*linear_pk(k)*pow(window(k*r,j),2);
  else xx+=k*k*k*linear_pk(k)*window(k*r,j);
      }
      if(j<3) xx+=0.5*(k1*k1*k1*linear_pk(k1)*pow(window(k1*r,j),2)+k2*k2*k2*linear_pk(k2)*pow(window(k2*r,j),2));
      else xx+=0.5*(k1*k1*k1*linear_pk(k1)*window(k1*r,j)+k2*k2*k2*linear_pk(k2)*window(k2*r,j));
      
      xx*=hh;

      // if(j==3) std::cout << xx << std::endl;

      if(fabs((xx-xxp)/xx)<eps) break;
      xxp=xx; 
      // printf("\033[2J");
      // printf("%lf",(xx-xxp)/xx);
      // fflush(stdout);
    }

    if(fabs((xx-xxpp)/xx)<eps) break;
    xxpp=xx;
  }

  if(j<3) return sqrt(xx/(2.0*M_PI*M_PI));
  else return xx/(2.0*M_PI*M_PI);
}

double window(double x, int i)
{
  if(i==0) return 3.0/pow(x,3)*(sin(x)-x*cos(x));  // top hat
  if(i==1) return exp(-0.5*x*x);   // gaussian
  if(i==2) return x*exp(-0.5*x*x);  // 1st derivative gaussian
  if(i==3) return x*x*(1-x*x)*exp(-x*x);
  printf("window ran out \n");
  return -1;
}

double calc_r_sigma(double D1)  // return r_sigma[Mpc/h] (=1/k_sigma)
{

  double k,k1,k2;

  k1=k2=1.;
  for(;;){
    if(D1*sigmam(1./k1,1)<1.) break;
    k1*=0.5;
  }
  for(;;){
    if(D1*sigmam(1./k2,1)>1.) break;
    k2*=2.;
  }

  for(;;){
    k=0.5*(k1+k2);
    if(D1*sigmam(1./k,1)<1.) k1=k; 
    else if(D1*sigmam(1./k,1)>1.) k2=k;
    if(D1*sigmam(1./k,1)==1. || fabs(k2/k1-1.)<eps*0.1) break;
  }

  return 1./k;
}

double GQ96_of_Einv(double a,double b)
{ /* 96-pt Gauss qaudrature integrates E^-1(x) from a to b */
  double cx=(a+b)/2;
  double dx=(b-a)/2;
  double q=0;
  for(int i=0;i<48;i++)
    {
      q+=W96[i]*(E_inv(cx-dx*A96[i])+E_inv(cx+dx*A96[i]));
    };
  return(q*dx);
}


double E(double z)
{   //assuming flat universe
    return sqrt(om*pow(1+z,3)+ow);
}

__device__ double dev_E(double z)
{   //assuming flat universe
    return sqrt(dev_om*pow(1+z,3)+dev_ow);
}

double E_inv(double z)
{
    return 1./E(z);
}
