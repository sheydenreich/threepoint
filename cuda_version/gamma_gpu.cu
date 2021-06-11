#include "gamma_gpu.cuh"




#define slics false
#define PERFORM_SUM_REDUCTION

static const double prec_h = 0.1;
static const int prec_k = int(3.5/prec_h);
static const int threadsPerBlock_sum_reduction_idx = 6;

int main()
{
    // Things that should be passed as function arguments
    int n_redshift_bins = 100;
    double z_max;
    if(slics) z_max = 3.;
    else z_max = 1.1;
    double dz = z_max / ((double) n_redshift_bins);

    struct cosmology cosmo;

    if(slics)
    {
        printf("using SLICS cosmology...");
        cosmo.h=0.6898;     // Hubble parameter
        cosmo.sigma8=0.826; // sigma 8
        cosmo.omb=0.0473;   // Omega baryon
        cosmo.omc=0.2905-cosmo.omb;   // Omega CDM
        cosmo.ns=0.969;    // spectral index of linear P(k)
        cosmo.w=-1.0;
        cosmo.om = cosmo.omb+cosmo.omc;
        cosmo.ow = 1-cosmo.om;
    }
    else
    {
        printf("using Millennium cosmology...");
        cosmo.h = 0.73;
        cosmo.sigma8 = 0.9;
        cosmo.omb = 0.045;
        cosmo.omc = 0.25 - cosmo.omb;
        cosmo.ns = 1.;
        cosmo.w = -1.0;
        cosmo.om = cosmo.omc+cosmo.omb;
        cosmo.ow = 1.-cosmo.om;
    }
    fflush(stdout);
    // Norm (maybe also function argument?)
    double norm;
    // Copy GC stuff to GPU
    CudaSafeCall(cudaMemcpyToSymbol(d_A96,&A96,48*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_W96,&W96,48*sizeof(double)));
    // Copy other parameters to GPU
    CudaSafeCall(cudaMemcpyToSymbol(d_z_max,&z_max,sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_dz,&dz,sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_n_redshift_bins,&n_redshift_bins,sizeof(int)));


    // Allocate memory for integration arrays on CPU
    double* f_K_array = new double[n_redshift_bins];
    double* g_array = new double[n_redshift_bins];
    double* D1_array = new double[n_redshift_bins];
    double* r_sigma_array = new double[n_redshift_bins];
    double* n_eff_array = new double[n_redshift_bins];
    // Allocate memory for integration arrays on GPU
    double* d_f_K_array;
    double* d_g_array;
    double* d_D1_array;
    double* d_r_sigma_array;
    double* d_n_eff_array;
    CudaSafeCall(cudaMalloc(&d_f_K_array,n_redshift_bins*sizeof(double)));
    CudaSafeCall(cudaMalloc(&d_g_array,n_redshift_bins*sizeof(double)));
    CudaSafeCall(cudaMalloc(&d_D1_array,n_redshift_bins*sizeof(double)));
    CudaSafeCall(cudaMalloc(&d_r_sigma_array,n_redshift_bins*sizeof(double)));
    CudaSafeCall(cudaMalloc(&d_n_eff_array,n_redshift_bins*sizeof(double)));

    // Allocate memory for bessel integration on CPU
    double* bessel_zeros = new double[prec_k];
    double* pi_bessel_zeros = new double[prec_k];
    double* array_psi = new double[prec_k];
    double* array_bessel = new double[prec_k];
    double* array_psip = new double[prec_k];
    double* array_w = new double[prec_k];
    double* array_product = new double[prec_k];
    double* array_psi_J2 = new double[prec_k];
    double* array_product_J2 = new double[prec_k];

    // Compute the weights
    for(unsigned int i=0;i<prec_k;i++){
        bessel_zeros[i] = gsl_sf_bessel_zero_Jnu(6,i);
        pi_bessel_zeros[i] = bessel_zeros[i]/M_PI;
        array_psi[i] = M_PI*psi(pi_bessel_zeros[i]*prec_h)/prec_h;
        array_bessel[i] = gsl_sf_bessel_Jn(6,array_psi[i]);
        array_psip[i] = psip(prec_h*pi_bessel_zeros[i]);
        array_w[i] = 2/(M_PI*bessel_zeros[i]*pow(gsl_sf_bessel_Jn(7,bessel_zeros[i]),2));
        array_product[i] = array_w[i]*pow(array_psi[i],3)*array_bessel[i]*array_psip[i];
    }

    for(unsigned int i=0;i<prec_k;i++){
        bessel_zeros[i] = gsl_sf_bessel_zero_Jnu(2,i);
        pi_bessel_zeros[i] = bessel_zeros[i]/M_PI;
        array_psi_J2[i] = M_PI*psi(pi_bessel_zeros[i]*prec_h)/prec_h;
        array_bessel[i] = gsl_sf_bessel_Jn(2,array_psi_J2[i]);
        array_psip[i] = psip(prec_h*pi_bessel_zeros[i]);
        array_w[i] = 2/(M_PI*bessel_zeros[i]*pow(gsl_sf_bessel_Jn(3,bessel_zeros[i]),2));
        array_product_J2[i] = array_w[i]*pow(array_psi_J2[i],3)*array_bessel[i]*array_psip[i];
    }
    // Copy the weights to the GPU
    CudaSafeCall(cudaMemcpyToSymbol(d_array_psi,array_psi,prec_k*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_array_product,array_product,prec_k*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_array_psi_J2,array_psi_J2,prec_k*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_array_product_J2,array_product_J2,prec_k*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_prec_k,&prec_k,sizeof(int)));

    // ##################################################################################################
    // #### Below here, things depend on cosmology and have to be repeated for different cosmologies!####
    // ##################################################################################################


    // Write cosmology to constant memory
    CudaSafeCall(cudaMemcpyToSymbol(d_h,&cosmo.h,sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_sigma8,&cosmo.sigma8,sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_omb,&cosmo.omb,sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_omc,&cosmo.omc,sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_ns,&cosmo.ns,sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_w,&cosmo.w,sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_om,&cosmo.om,sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(d_ow,&cosmo.ow,sizeof(double)));
    norm=1.;
    norm*=cosmo.sigma8/sigmam(8.,0,cosmo,norm);
    CudaSafeCall(cudaMemcpyToSymbol(d_norm,&norm,sizeof(double)));

    // Write g and f_k arrays
    write_f_K_array<<<1,n_redshift_bins>>>(d_f_K_array);
    CudaCheckError();
    cudaDeviceSynchronize();
//    printf("test \n");
    write_g_array<<<1,n_redshift_bins>>>(d_g_array,d_f_K_array);
    CudaCheckError();
    // copy g and f_k arrays to constant memory
    CudaSafeCall(cudaMemcpy(f_K_array,d_f_K_array,n_redshift_bins*sizeof(double),cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(g_array,d_g_array,n_redshift_bins*sizeof(double),cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpyToSymbol(c_f_K_array,f_K_array,n_redshift_bins*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(c_g_array,g_array,n_redshift_bins*sizeof(double)));
    // Compute nonlinear scales
    write_nonlinear_scales<<<1,n_redshift_bins>>>(d_D1_array, d_r_sigma_array, d_n_eff_array);
    CudaCheckError();
    // copy nonlinear scales to constant memory
    CudaSafeCall(cudaMemcpy(D1_array,d_D1_array,n_redshift_bins*sizeof(double),cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(r_sigma_array,d_r_sigma_array,n_redshift_bins*sizeof(double),cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(n_eff_array,d_n_eff_array,n_redshift_bins*sizeof(double),cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpyToSymbol(c_D1_array,d_D1_array,n_redshift_bins*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(c_r_sigma_array,d_r_sigma_array,n_redshift_bins*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(c_n_eff_array,d_n_eff_array,n_redshift_bins*sizeof(double)));
    printf("Preparations done. Computing Gamma.\n");


    double x1 = 10.*M_PI/180./60.;
    std::complex<double> testgamma = gamma0(x1,x1,x1,z_max,prec_k);
    std::cout << testgamma << std::endl;
    
    return 0;

    // CALCULATE ALL THE GAMMA

    int steps = 10;
    int usteps = 10;
    int vsteps = 10;
    double* r_array = new double[steps];
    double* u_array = new double[usteps];
    double* v_array = new double[vsteps];

    double rmin = 0.1;
    double rmax = 120.;
    double umin = 0;
    double umax = 1;
    double vmin = 0;
    double vmax = 1;

    double lrmin = log(rmin);
    double lrmax = log(rmax);

    for(int i=0;i<steps;i++)
    {
      double temp = lrmin+(lrmax-lrmin)/steps*(i+0.5);
      r_array[i] = exp(temp);
    }
    for(int i=0;i<usteps;i++)
    {
      u_array[i] = umin+(umax-umin)/usteps*(i+0.5);
    }
    for(int i=0;i<vsteps;i++)
    {
      v_array[i] = vmin+(vmax-vmin)/vsteps*(i+0.5);
    }

    std::complex<double> result_gamma0[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")

    printf("Computing gammas...[100%%]");
    for(int i=0;i<steps;i++){
        for(int j=0;j<usteps;j++){
        	for(int k=0;k<vsteps;k++){
                printf("\b\b\b\b\b\b[%3d%%]",static_cast<int>(100*(1.*i*usteps*vsteps+j*vsteps+k)/(usteps*vsteps*steps)));
        		fflush(stdout);

                    double r,u,v;
                    int id_x = i*usteps*vsteps+j*vsteps+k;
                    // r = triangle_configurations[id_x].r;
                    // u = triangle_configurations[id_x].u;
                    // v = triangle_configurations[id_x].v;]
                    r = r_array[i];
                    u = u_array[j];
                    v = v_array[k];


                    double r2 = r; //THIS IS THE BINNING BY JARVIS. FROM THE WEBSITE, NOT THE PAPER.
                    double r3 = r2*u;
                    double r1 = v*r3+r2;
                    // std::complex<double> res_temp = class_gamma.gamma0(M_PI/180./60.*r1, M_PI/180./60.*r2, M_PI/180./60.*r3);
                    // if(isnan(real(res_temp)) || isnan(imag(res_temp))){
                    //     printf("%lf %lf %lf %lf %lf \n",r1,r2,r3,real(res_temp),imag(res_temp));
                    //     res_temp = std::complex<double>(0.0,0.0);
                    // }
                    // assert(!isnan(real(res_temp)) && !isnan(imag(res_temp)));
                    if(r1!=0 && r2!=0 && r3!=0) 
                    {
                      result_gamma0[id_x] = gamma0(M_PI/180./60.*r1, M_PI/180./60.*r2, M_PI/180./60.*r3,z_max,prec_k);
                    }
            }
        }
    }

    printf("\b\b\b\b\b\b[100%%] Done. Writing results...");
    fflush(stdout);

    FILE *fp;

    fp = fopen("../../../../Sciebo/data/Gammas_0p1_to_120_cuda.dat","w");
    for(int i=0;i<steps;i++){
        for(int j=0;j<usteps;j++){
            for(int k=0;k<vsteps;k++){
                int id_x;
                id_x = i*usteps*vsteps+j*vsteps+k;

                double r,u,v;
                // r = triangle_configurations[id_x].r;
                // u = triangle_configurations[id_x].u;
                // v = triangle_configurations[id_x].v;
                r = r_array[i];
                u = u_array[j];
                v = v_array[k];


                fprintf(fp,"%d %d %d %e %e 0.0 0.0 0.0 0.0 0.0 0.0 %e %e %e\n",i ,j ,k ,real(result_gamma0[id_x]),imag(result_gamma0[id_x]),r,u,v);
            }
        }
    }
    fclose(fp);
    printf("\b\b\b Done.\n");



}

__device__ double GQ96(double F(double),double a,double b)
{ /* 96-pt Gauss qaudrature integrates F(x) from a to b */
    int i;
    double cx,dx,q;
    cx=(a+b)/2;
    dx=(b-a)/2;
    q=0;
    for(i=0;i<48;i++)
      q+=d_W96[i]*(F(cx-dx*d_A96[i])+F(cx+dx*d_A96[i]));
    return(q*dx);
}


__device__ inline double A(double psi, double x1, double x2, double phi, double varpsi){
    return sqrt(pow(cos(psi)*x2,2)+pow(sin(psi)*x1,2)+sin(2*psi)*x1*x2*cos(phi+varpsi));
}


__device__ inline double alpha(double psi, double x1, double x2, double phi, double varpsi){
    double zahler = (cos(psi)*x2-sin(psi)*x1)*sin((phi+varpsi)/2);
    double nenner = (cos(psi)*x2+sin(psi)*x1)*cos((phi+varpsi)/2);
    return atan2(zahler,nenner);
}

__device__ inline double betabar(double psi, double phi){
    double zahler = cos(2*psi)*sin(phi);
    double nenner = cos(phi)+sin(2*psi);
    return 0.5*atan2(zahler,nenner);
}

__device__ inline double interior_angle(double an1, double an2, double opp){
    return acos((pow(an1,2)+pow(an2,2)-pow(opp,2))/(2.0*an1*an2));
}


double inline psi(double t){
    return t*tanh(M_PI*sinh(t)/2);
}

double inline psip(double t){
    double zahler = sinh(M_PI*sinh(t))+M_PI*t*cosh(t);
    double nenner = cosh(M_PI*sinh(t))+1;
    return zahler/nenner;
}

__global__ void write_f_K_array(double* d_f_K_array)
{
    int i = threadIdx.x;
    double z_now = i*d_dz;
    d_f_K_array[i] = f_K_at_z(z_now);
}

__device__ double E(double z)
{   //H(z)/H_0 for a flat Universe, disregarding Omega_r
    return sqrt(d_om*pow(1+z,3)+d_ow);
}

__device__ double E_inv(double z)
{
    return 1/E(z);
}

__device__ double f_K_at_z(double z)
{
    return d_c_over_H0*GQ96(E_inv,0,z);
}

__device__ double n_of_z(double z)
{
    // if(z<=0 || z>=d_z_max) return 0;
//	if(slics)
//	{
//		return pow(z,2)*exp(-pow(z/0.637,1.5))*5.80564; //normalization 5.80564 such that int_0^3 is 1
//	}
//	else
//	{
	    if(z>=1 && z<1+d_dz) return 1./d_dz;
	    else return 0;
//	}
}

__global__ void write_g_array(double* d_g_array, double* d_f_K_array)
{
    int i = threadIdx.x;
    double z_now;
    d_g_array[i] = 0;

    if(i==0)
    {
        d_g_array[i] = 1;
        return;
    }
    else
    {
        // i != 0
        for(int j=i;j<d_n_redshift_bins;j++){
            z_now = j*d_dz;
            if(j==i || j==d_n_redshift_bins-1){
                d_g_array[i] += n_of_z(z_now)*(d_f_K_array[j]-d_f_K_array[i])/d_f_K_array[j]/2;
            }
            else{
                d_g_array[i] += n_of_z(z_now)*(d_f_K_array[j]-d_f_K_array[i])/d_f_K_array[j];
            }
        }
        d_g_array[i] = d_g_array[i]*d_dz;
    }
}

double sigmam(double r, int j, cosmology cosmo, double norm)   // r[Mpc/h]
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
	xx+=k*k*k*linear_pk(k,cosmo,norm)*pow(window(k*r,j),2);
      }
      xx+=0.5*(k1*k1*k1*linear_pk(k1,cosmo,norm)*pow(window(k1*r,j),2)+k2*k2*k2*linear_pk(k2,cosmo,norm)*pow(window(k2*r,j),2));
      xx*=hh;

      if(fabs((xx-xxp)/xx)<eps) break;
      xxp=xx;
    }

    if(fabs((xx-xxpp)/xx)<eps) break;
    xxpp=xx;
  }

  return sqrt(xx/(2.0*M_PI*M_PI));
}

double window(double x, int i)
{
  if(i==0) return 3.0/pow(x,3)*(sin(x)-x*cos(x));  // top hat
  if(i==1) return exp(-0.5*x*x);   // gaussian
  if(i==2) return x*exp(-0.5*x*x);  // 1st derivative gaussian
  printf("window ran out \n");
  return -1;
}

double linear_pk(double k, cosmology cosmo, double norm)   // Eisenstein & Hu (1999) fitting formula without wiggle,      k[h/Mpc], P(k)[(Mpc/h)^3]
{
  double pk,delk,alnu,geff,qeff,L,C;
  k*=cosmo.h;  // unit conversion from [h/Mpc] to [1/Mpc]

  double fc=cosmo.omc/cosmo.om;
  double fb=cosmo.omb/cosmo.om;
  double theta=2.728/2.7;
  double pc=0.25*(5.0-sqrt(1.0+24.0*fc));
  double omh2=cosmo.om*cosmo.h*cosmo.h;
  double ombh2=cosmo.omb*cosmo.h*cosmo.h;
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

  delk=pow(norm,2)*pow(k*2997.9/cosmo.h,3.+cosmo.ns)*pow(L/(L+C*qeff*qeff),2);
  pk=2.0*M_PI*M_PI/(k*k*k)*delk;

  return pow(cosmo.h,3)*pk;
}

__device__ double d_sigmam(double r, int j)   // r[Mpc/h]
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
      xx+=k*k*k*d_linear_pk(k)*pow(d_window(k*r,j),2);
      }
      xx+=0.5*(k1*k1*k1*d_linear_pk(k1)*pow(d_window(k1*r,j),2)+k2*k2*k2*d_linear_pk(k2)*pow(d_window(k2*r,j),2));
      xx*=hh;

      if(fabs((xx-xxp)/xx)<d_eps) break;
      xxp=xx;
    }

    if(fabs((xx-xxpp)/xx)<d_eps) break;
    xxpp=xx;
  }

  return sqrt(xx/(2.0*M_PI*M_PI));
}

__device__ double d_window(double x, int i)
{
  if(i==0) return 3.0/pow(x,3)*(sin(x)-x*cos(x));  // top hat
  if(i==1) return exp(-0.5*x*x);   // gaussian
  if(i==2) return x*exp(-0.5*x*x);  // 1st derivative gaussian
  printf("window ran out \n");
  return -1;
}

__device__ double d_linear_pk(double k)   // Eisenstein & Hu (1999) fitting formula without wiggle,      k[h/Mpc], P(k)[(Mpc/h)^3]
{
  double pk,delk,alnu,geff,qeff,L,C;
  k*=d_h;  // unit conversion from [h/Mpc] to [1/Mpc]

  double fc=d_omc/d_om;
  double fb=d_omb/d_om;
  double theta=2.728/2.7;
  double pc=0.25*(5.0-sqrt(1.0+24.0*fc));
  double omh2=d_om*d_h*d_h;
  double ombh2=d_omb*d_h*d_h;
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

  delk=pow(d_norm,2)*pow(k*2997.9/d_h,3.+d_ns)*pow(L/(L+C*qeff*qeff),2);
  pk=2.0*M_PI*M_PI/(k*k*k)*delk;

  return pow(d_h,3)*pk;
}


__device__ double lgr(double z)  // linear growth factor at z (not normalized at z=0)
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

    if(fabs(y[0]/yp-1.)<0.1*d_eps) break;
    yp=y[0];
  }

  return a*y[0];
}

__device__ double lgr_func(int j, double la, double y[2])
{
  if(j==0) return y[1];

  double g,a;
  a=exp(la);
  g=-0.5*(5.*d_om+(5.-3*d_w)*d_ow*pow(a,-3.*d_w))*y[1]-1.5*(1.-d_w)*d_ow*pow(a,-3.*d_w)*y[0];
  g=g/(d_om+d_ow*pow(a,-3.*d_w));
  if(j==1) return g;
  printf("lgr_func wrong j");
  return -1;
}

__global__ void write_nonlinear_scales(double* d_D1_array, double* d_r_sigma_array, double* d_n_eff_array)
{
    int i = threadIdx.x;
    double z_now = i*d_dz;
    d_D1_array[i]=lgr(z_now)/lgr(0.);   // linear growth factor
    d_r_sigma_array[i]=calc_r_sigma(z_now,d_D1_array[i]);  // =1/k_NL [Mpc/h] in Eq.(B1)
    d_n_eff_array[i]=-3.+2.*pow(d_D1_array[i]*d_sigmam(d_r_sigma_array[i],2),2);   // n_eff in Eq.(B2)

}

__device__ double calc_r_sigma(double z, double D1)  // return r_sigma[Mpc/h] (=1/k_sigma)
{
  double k,k1,k2;

  k1=k2=1.;
  for(;;){
    if(D1*d_sigmam(1./k1,1)<1.) break;
    k1*=0.5;
  }
  for(;;){
    if(D1*d_sigmam(1./k2,1)>1.) break;
    k2*=2.;
  }

  for(;;){
    k=0.5*(k1+k2);
    if(D1*d_sigmam(1./k,1)<1.) k1=k;
    else if(D1*d_sigmam(1./k,1)>1.) k2=k;
    if(D1*d_sigmam(1./k,1)==1. || fabs(k2/k1-1.)<d_eps*0.1) break;
  }

  return 1./k;
}


__device__ double bispec(double k1, double k2, double k3, double z, int idx, double didx)   // non-linear BS w/o baryons [(Mpc/h)^6]
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
  logsigma8z=log10(D1*d_sigma8);

  // 1-halo term parameters in Eq.(B7)
  an=pow(10.,-2.167-2.944*logsigma8z-1.106*pow(logsigma8z,2)-2.865*pow(logsigma8z,3)-0.310*pow(r1,pow(10.,0.182+0.57*n_eff)));
  bn=pow(10.,-3.428-2.681*logsigma8z+1.624*pow(logsigma8z,2)-0.095*pow(logsigma8z,3));
  cn=pow(10.,0.159-1.107*n_eff);
  alphan=pow(10.,-4.348-3.006*n_eff-0.5745*pow(n_eff,2)+pow(10.,-0.9+0.2*n_eff)*pow(r2,2));
  if(alphan>1.-(2./3.)*d_ns) alphan=1.-(2./3.)*d_ns;
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
    PSE[i]=(1.+fn*pow(q[i],2))/(1.+gn*q[i]+hn*pow(q[i],2))*pow(D1,2)*d_linear_pk(q[i]/r_sigma)+1./(mn*pow(q[i],mun)+nn*pow(q[i],nun))/(1.+pow(pn*q[i],-3));  // enhanced P(k) in Eq.(B6)
  }

  // 3-halo term bispectrum in Eq.(B5)
  BS3h=2.*(F2(k1,k2,k3,z,D1,r_sigma)*PSE[1]*PSE[2]+F2(k2,k3,k1,z,D1,r_sigma)*PSE[2]*PSE[3]+F2(k3,k1,k2,z,D1,r_sigma)*PSE[3]*PSE[1]);
  for(i=1;i<=3;i++) BS3h*=1./(1.+en*q[i]);
  // if (BS1h+BS3h>=0)
  return BS1h+BS3h;
  // else return 0;
}

std::complex<double> gamma0(double x1, double x2, double x3, double z_max, unsigned int prec_k)
{
  double vals_min[3] = {0,0,0};
  double vals_max[3] = {z_max,2*M_PI,M_PI/2};
  double result[2];
  double error[2];
  struct GammaCudaContainer params;
  params.x1 = x1;
  params.x2 = x2;
  params.x3 = x3;
  params.prec_k = prec_k;
  double epsabs = 0;
  hcubature_v(2,integrand_gamma0,&params,3,vals_min,vals_max,0,epsabs,1e-3,ERROR_L1,result,error);
  return std::complex<double>(result[0],result[1])*27./8.*pow(0.25,3)*pow(100./299792.,5)/3./(2*pow(2*M_PI,3));
}


int integrand_gamma0(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value)
{
    struct GammaCudaContainer params = *((GammaCudaContainer*) fdata);

    // std::cout << npts << std::endl;

    double x1 = params.x1;
    double x2 = params.x2;
    double x3 = params.x3;

    unsigned int prec_k = params.prec_k;
    double* d_vars;
    double* d_value;
    
    int threadsPerBlock_k = 4;
    int threadsPerBlock_idx = 64;
    int max_blocksPerGrid_idx = pow(2,15);
    dim3 threadsPerBlock(threadsPerBlock_k,threadsPerBlock_idx);

    int blocksPerGrid_k = static_cast<int>(ceil(prec_k*1./threadsPerBlock_k));
    int blocksPerGrid_idx = static_cast<int>(ceil(npts*1./threadsPerBlock_idx));

    unsigned int incr = 0;
    // allocate memory
    double calculationsPerIncrement = max_blocksPerGrid_idx*threadsPerBlock_idx;

    CudaSafeCall(cudaMalloc(&d_value,fdim*npts*sizeof(double)));
    CudaSafeCall(cudaMalloc(&d_vars,ndim*calculationsPerIncrement*sizeof(double)));

    dim3 threadsPerBlock_sum_reduction(prec_k-1,threadsPerBlock_sum_reduction_idx);
    dim3 blocksPerGrid_sum_reduction(1,static_cast<int>(ceil(npts*1./threadsPerBlock_sum_reduction.y)));



    double myNpts = npts;

    while(blocksPerGrid_idx>max_blocksPerGrid_idx)
    {
      std::cout << "WARNING: block size too large." << std::endl; //TODO: shared computation also on this one
      // offset the parameters and the result array
      const double* off_vars = vars + ndim*incr;
      double* off_result_array = value + fdim*incr;

      // copy the parameters
      CudaSafeCall(cudaMemcpy(d_vars,off_vars,ndim*calculationsPerIncrement*sizeof(double),cudaMemcpyHostToDevice));
  
      // perform calculations
      dim3 blocksPerGrid(blocksPerGrid_k,max_blocksPerGrid_idx);
      // std::cout << blocksPerGrid.x << ", " << blocksPerGrid.y << ", " << blocksPerGrid.z << "\t ," <<
      // threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z << std::endl;
      #ifdef PERFORM_SUM_REDUCTION
        compute_integrand_gamma0_with_sum_reduction<<<blocksPerGrid_sum_reduction,threadsPerBlock_sum_reduction>>>(d_vars,d_value,myNpts,x1,x2,x3);
      #else
        compute_integrand_gamma0<<<blocksPerGrid,threadsPerBlock>>>(d_vars,d_value,myNpts,x1,x2,x3);
      #endif //PERFORM_SUM_REDUCTION

      CudaCheckError();

      // copy the result
      CudaSafeCall(cudaMemcpy(off_result_array,d_value,fdim*calculationsPerIncrement*sizeof(double),cudaMemcpyDeviceToHost));
      // set array back to zero
      CudaSafeCall(cudaMemset(d_value,0,fdim*calculationsPerIncrement*sizeof(double)));
      

      // adjust the increment
      incr += calculationsPerIncrement;
      blocksPerGrid_idx -= max_blocksPerGrid_idx;
      myNpts -= calculationsPerIncrement;
    }

    // perform the rest of the calculations
    calculationsPerIncrement = blocksPerGrid_idx*threadsPerBlock_idx;

    // offset the parameters and the result array
    const double* off_vars = vars + ndim*incr;
    double* off_result_array = value + fdim*incr;
    
    // std::cout << "Copy: " << off_vars[0]-vars[ndim*incr] << ", " << incr << ", " << npts << "," << myNpts << std::endl;

    // copy the parameters
    CudaSafeCall(cudaMemcpy(d_vars,off_vars,ndim*myNpts*sizeof(double),cudaMemcpyHostToDevice));


    // perform calculations
    dim3 blocksPerGrid(blocksPerGrid_k,blocksPerGrid_idx);


    // std::cout << "final: " << blocksPerGrid.x << ", " << blocksPerGrid.y << ", " << blocksPerGrid.z << "\t ," <<
    // threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z << std::endl;
    
    // set array to zero
    CudaSafeCall(cudaMemset(d_value,0,fdim*myNpts*sizeof(double)));

    #ifdef PERFORM_SUM_REDUCTION
      compute_integrand_gamma0_with_sum_reduction<<<blocksPerGrid_sum_reduction,threadsPerBlock_sum_reduction>>>(d_vars,d_value,myNpts,x1,x2,x3);
    #else
      compute_integrand_gamma0<<<blocksPerGrid,threadsPerBlock>>>(d_vars,d_value,myNpts,x1,x2,x3);
    #endif //PERFORM_SUM_REDUCTION

    CudaCheckError();


    // copy the result
    CudaSafeCall(cudaMemcpy(off_result_array,d_value,fdim*myNpts*sizeof(double),cudaMemcpyDeviceToHost));

    // free allocated memory
    CudaSafeCall(cudaFree(d_vars));
    CudaSafeCall(cudaFree(d_value));

    // std::cout << npts << ", " << x1 << ", " << x2 << ", " << x3 << ", " << value[6] << ", " << value[7] 
    // << ", " << vars[9] << ", " << vars[10] << ", " << vars[11] << std::endl;
    // std::cout << "npts: " << npts << std::endl;
    // std::cout << "blocks: " 
    // << blocksPerGrid.x*threadsPerBlock.x << "/" 
    // << blocksPerGrid.y*threadsPerBlock.y << "/" 
    // << blocksPerGrid.z*threadsPerBlock.z << std::endl;
    // fflush(stdout);
    // std::cout << npts << ", " << x1 << ", " << value[2*5] << ", " << value[2*5+1] 
    // << ", " << vars[3*5] << ", " << vars[3*5+1] << ", " << vars[3*5+2] << std::endl;
    // for(int i=0;i<npts;i++)
    // {
    //   if(i==incr*fdim) std::cout << "***************************************************************" << std::endl;
    //   if(i==incr*ndim) std::cout << "###############################################################" << std::endl;
    //   std::cout << i << "/" << incr << " -- z: " << vars[i*3] << ", phi: " << vars[i*3+1] << ", psi: " << vars[i*3+2] << ", integrand: " <<
    //   value[i*2] << " + " << value[i*2+1] << "i" << std::endl; 
    //   fflush(stdout);
    // }
    return 0;
}


__global__ void compute_integrand_gamma0_with_sum_reduction(double* d_vars, double* d_result_array, unsigned int max_idx, double x1, double x2, double x3)
{
    unsigned int idx = blockDim.y*blockIdx.y+threadIdx.y;
    unsigned int k = blockDim.x*blockIdx.x+threadIdx.x+1;
    int tidx = threadIdx.y;
    int idk = threadIdx.x; //idk = k-1
  
    // if(k==1 && idx > 70) printf("%d \n",idx);
    if(k>=d_prec_k) return;
    if(idx>=max_idx) return;
    double z=d_vars[idx*3];
    double phi=d_vars[idx*3+1];
    double psi=d_vars[idx*3+2];

    cuDoubleComplex result = full_integrand_gamma0(phi,psi,z,k,x1,x2,x3);

    __shared__ double r[threadsPerBlock_sum_reduction_idx*(prec_k-1)];
    __shared__ double r2[threadsPerBlock_sum_reduction_idx*(prec_k-1)];
    r[tidx*(prec_k-1)+idk] = cuCreal(result);
    r2[tidx*(prec_k-1)+idk] = cuCimag(result);
  
    __syncthreads();
    for (int size = (prec_k-1)/2; size>0; size/=2) { //uniform
        if (idk<size)
        {
          r[tidx*(prec_k-1)+idk] += r[tidx*(prec_k-1)+idk+size];
          r2[tidx*(prec_k-1)+idk] += r2[tidx*(prec_k-1)+idk+size];
        }
        __syncthreads();
    }
    if (idk == 0)
    {
      d_result_array[idx*2] = r[tidx*(prec_k-1)];
      d_result_array[idx*2+1] = r2[tidx*(prec_k-1)];
    }    
  
    return;
}

#ifndef PERFORM_SUM_REDUCTION

__global__ void compute_integrand_gamma0(double* d_vars, double* d_result_array, unsigned int max_idx, double x1, double x2, double x3)
{
    unsigned int idx = blockDim.y*blockIdx.y+threadIdx.y;
    unsigned int k = blockDim.x*blockIdx.x+threadIdx.x+1;

    if(k>=d_prec_k) return;
    if(idx>=max_idx) return;
    double z=d_vars[idx*3];
    double phi=d_vars[idx*3+1];
    double psi=d_vars[idx*3+2];

    cuDoubleComplex result = full_integrand_gamma0(phi,psi,z,k,x1,x2,x3);

    
    atomicAdd(&d_result_array[idx*2],cuCreal(result));
    atomicAdd(&d_result_array[idx*2+1],cuCimag(result));

    return;
}

#endif // PERFORM_SUM_REDUCTION

// __global__ void doubleSumSingleBlock(const double *in, double *out, int max_idx)
// {
//   unsigned int idx = blockDim.y*blockIdx.y+threadIdx.y;
//   int tidx = threadIdx.y;
//   int idk = threadIdx.x; //idk = k-1
//   // printf("Block: %d/%d \n",blockDim.y,threadsPerBlock_sum_reduction_idx);
//   if(idx>=max_idx) 
//   {
//     // if(idk==0) printf("%d %d \n",idx,max_idx);
//    return; 
//   }
//   // int sum = 0;
//   // for (int i = idx; i < arraySize; i += blockSize)
//   //     sum += in[i];
//   __shared__ double r[threadsPerBlock_sum_reduction_idx*(prec_k-1)];
//   __shared__ double r2[threadsPerBlock_sum_reduction_idx*(prec_k-1)];
//   r[tidx*(prec_k-1)+idk] = in[idx*2+idk*max_idx*2];
//   r2[tidx*(prec_k-1)+idk] = in[idx*2+1+idk*max_idx*2];

//   __syncthreads();
//   for (int size = (prec_k-1)/2; size>0; size/=2) { //uniform
//       if (idk<size)
//       {
//         r[tidx*(prec_k-1)+idk] += r[tidx*(prec_k-1)+idk+size];
//         r2[tidx*(prec_k-1)+idk] += r2[tidx*(prec_k-1)+idk+size];
//       }
//       __syncthreads();
//   }
//   if (idk == 0)
//   {
//     out[idx*2] = r[tidx*(prec_k-1)];
//     out[idx*2+1] = r2[tidx*(prec_k-1)];
//   }    
// }


__device__ cuDoubleComplex one_integrand_gamma0(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3)
{
    double varpsi = interior_angle(x1,x2,x3);
    double A3 = A(psi,x1,x2,phi,varpsi);
    cuDoubleComplex prefac = prefactor(x1,x2,x3,phi,psi);
    double ell1 = d_array_psi[k]/A3*cos(psi);
    double ell2 = d_array_psi[k]/A3*sin(psi);
    double ell3 = ell1*ell1+ell2*ell2+2*ell1*ell2*cos(phi);
    if(ell3 <= 0) ell3 = 0;
    else ell3 = sqrt(ell3);

    double bis = integrand_bkappa(z,ell1,ell2,ell3)*d_array_product[k]*M_PI/pow(A3,4);

    if(isnan(cuCreal(prefac)) || isnan(cuCimag(prefac)) || isnan(bis))
      printf("%.3e, %.3e, %.3e, %d, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e \n",phi,psi,z,k,ell1,ell2,ell3,integrand_bkappa(z,ell1,ell2,ell3),d_array_product[k],
      cuCreal(prefac),cuCimag(prefac));
    return cuCmul(prefac,bis);
}

__device__ __inline__ cuDoubleComplex full_integrand_gamma0(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3)
{
    return cuCadd(cuCadd(one_integrand_gamma0(phi,psi,z,k,x1,x2,x3),one_integrand_gamma0(phi,psi,z,k,x2,x3,x1)),one_integrand_gamma0(phi,psi,z,k,x3,x1,x2));
}

__device__ cuDoubleComplex prefactor(double x1, double x2, double x3, double phi, double psi)
{
    double varpsi = interior_angle(x1,x2,x3);
    cuDoubleComplex exponential = exp_of_imag(interior_angle(x2,x3,x1)-interior_angle(x1,x3,x2)-6*alpha(psi,x1,x2,phi,varpsi));
    cuDoubleComplex prefactor_phi = exp_of_imag(2.*betabar(psi,phi));
    double prefactor_psi = sin(2*psi);
    return cuCmul(cuCmul(exponential,prefactor_phi),prefactor_psi);
}




__device__ double bispec_tree(double k1, double k2, double k3, double z, double D1)  // tree-level BS [(Mpc/h)^6]
{
  return pow(D1,4)*2.*(F2_tree(k1,k2,k3)*d_linear_pk(k1)*d_linear_pk(k2)
		      +F2_tree(k2,k3,k1)*d_linear_pk(k2)*d_linear_pk(k3)
		      +F2_tree(k3,k1,k2)*d_linear_pk(k3)*d_linear_pk(k1));
}


__device__ double F2(double k1, double k2, double k3, double z, double D1, double r_sigma)
{
  double a,q[4],dn,omz,logsigma8z;

  q[3]=k3*r_sigma;

  logsigma8z=log10(D1*d_sigma8);
  a=1./(1.+z);
  omz=d_om/(d_om+d_ow*pow(a,-3.*d_w));   // Omega matter at z

  dn=pow(10.,-0.483+0.892*logsigma8z-0.086*omz);

  return F2_tree(k1,k2,k3)+dn*q[3];
}


__device__ double F2_tree(double k1, double k2, double k3)  // F2 kernel in tree level
{
  double costheta12=0.5*(k3*k3-k1*k1-k2*k2)/(k1*k2);
  return (5./7.)+0.5*costheta12*(k1/k2+k2/k1)+(2./7.)*costheta12*costheta12;
}

__device__ int compute_coefficients(int idx, double didx, double *D1, double *r_sigma, double *n_eff){
     *D1 = c_D1_array[idx]*(1-didx) + c_D1_array[idx+1]*didx;
     *r_sigma = c_r_sigma_array[idx]*(1-didx) + c_r_sigma_array[idx+1]*didx;
     *n_eff = c_n_eff_array[idx]*(1-didx) + c_n_eff_array[idx+1]*didx;
      return 1;
}


__device__ double integrand_bkappa(double z, double ell1, double ell2, double ell3){
    if(z<1.0e-7) return 0.;
    if(ell1<=1.0e-10||ell2<=1.0e-10||ell3<=1.0e-10)
    {
      // printf("%.3e %.3e %.3e \n",ell1,ell2,ell3);
      return 0;
    } 
    assert(ell1>0&&ell2>0&&ell3>0);
    double didx = z/d_z_max*(d_n_redshift_bins-1);
    int idx = didx;
    didx = didx - idx;
    if(idx==d_n_redshift_bins-1){
        idx = d_n_redshift_bins-2;
        didx = 1.;
    }
    double g_value = g_interpolated(idx,didx);
    double f_K_value = f_K_interpolated(idx,didx);
    double result = pow(g_value*(1.+z),3)*bispec(ell1/f_K_value,ell2/f_K_value,ell3/f_K_value,z,idx,didx)/f_K_value/E(z);
    if(isnan(result))
    {
      printf("nan in bispec! %lf, %lf, %lf, %lf, %.3f \n",f_K_value,ell1,ell2,ell3,z);
      return 0;
    }
    assert(isfinite(result));
    return result;
}

__device__ double g_interpolated(int idx, double didx){
    return c_g_array[idx]*(1-didx) + c_g_array[idx+1]*didx;
}

__device__ double f_K_interpolated(int idx, double didx){
    return c_f_K_array[idx]*(1-didx) + c_f_K_array[idx+1]*didx;
}

__host__ __device__ static __inline__ cuDoubleComplex cuCmul(cuDoubleComplex x,double y)
{
    cuDoubleComplex prod;
    prod = make_cuDoubleComplex ((cuCreal(x) * y),(cuCimag(x) * y));
    return prod;
}

__host__ __device__ static __inline__ cuDoubleComplex exp_of_imag(double imag_part)
{
    cuDoubleComplex result;
    result = make_cuDoubleComplex(cos(imag_part), sin(imag_part));
    return result;
}