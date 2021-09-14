#include "bispectrum.hpp"
#include <cmath>
#include <gsl/gsl_errno.h>
#include "cubature.h"


bool is_triangle(double ell1, double ell2, double ell3)
{
  if(abs(ell1-ell2)>ell3 || ell3 > (ell1+ell2)) return false;
  if(abs(ell2-ell3)>ell1 || ell1 > (ell2+ell3)) return false;
  if(abs(ell1-ell3)>ell2 || ell2 > (ell1+ell3)) return false;
  return true;
}


double number_of_triangles(double ell1, double ell2, double ell3){ //TODO: maybe write this into a utility.cpp?
    if(is_triangle(ell1,ell2,ell3))
    {
        double lambda = 2*ell1*ell1*ell2*ell2 + 2*ell1*ell1*ell3*ell3 + 2*ell2*ell2*ell3*ell3 - pow(ell1,4) - pow(ell2,4) - pow(ell3,4);
        lambda = 4./sqrt(lambda);
        return lambda;
    }
    else return 0;
}


// Constructor for bispectrum class
BispectrumCalculator::BispectrumCalculator()
{
  return;
}


BispectrumCalculator::BispectrumCalculator(cosmology cosmo, int n_z, double z_max_arg, bool fast_calculations_arg)
{
  std::cout<<"Start initializing Bispectrum"<<std::endl;
  initialize(cosmo, n_z, z_max_arg, fast_calculations_arg);
  std::cout<<"Finished initializing Bispectrum"<<std::endl;
}


void BispectrumCalculator::initialize(cosmology cosmo, int n_z, double z_max_arg, bool fast_calculations_arg)
{
    if(initialized) return;
    fast_calculations = fast_calculations_arg;
    n_redshift_bins = n_z;
    z_max = z_max_arg;
    dz = z_max / ((double) n_redshift_bins);
    //  printf("Initializing BispectrumCalculator... Filling the arrays... \n");
    // Allocate memory
    std::cout<<"Allocating Memory"<<std::endl;
    f_K_array = new double[n_redshift_bins];
    g_array = new double[n_redshift_bins];
    D1_array = new double[n_redshift_bins];
    r_sigma_array = new double[n_redshift_bins];
    n_eff_array = new double[n_redshift_bins];
    ncur_array = new double[n_redshift_bins];
    std::cout<<"Memory is allocated"<<std::endl;
    set_cosmology(cosmo);
    initialized = true;
}

void BispectrumCalculator::set_cosmology(cosmology cosmo)
{
  std::cout<<"Started setting cosmology"<<std::endl;
  // Necessary temporary variables
  double z_now;
  // Assign the correct cosmology
  h = cosmo.h;
  sigma8 = cosmo.sigma8;
  omb = cosmo.omb;
  omc = cosmo.omc;
  ns = cosmo.ns;
  w = cosmo.w;
  om = cosmo.om;
  ow = cosmo.ow;
  // Calculate f_K(z) and g(z) on a grid
  for(int i=0;i<n_redshift_bins;i++){ //fill the chi arrays
    z_now = i*dz;
    f_K_array[i] = f_K_at_z(z_now);
    assert(isfinite(f_K_array[i]));
    }
  
  for(int i=0;i<n_redshift_bins;i++){ //fill the lenseff arrays
    g_array[i] = 0;
    // perform trapezoidal integration
    for(int j=i;j<n_redshift_bins;j++){
      z_now = j*dz;
      if(j==i || j==n_redshift_bins-1){
	g_array[i] += n_of_z(z_now)*(f_K_array[j]-f_K_array[i])/f_K_array[j]/2;
      }
      else{
	g_array[i] += n_of_z(z_now)*(f_K_array[j]-f_K_array[i])/f_K_array[j];
      }
            // if(z_now>=1 && z_now<1+dz)
            // {
            //   printf("%.3f \n",z_now);
            // // if(n_of_z(z_now)!=0)
            //   printf("%.3e \n",n_of_z(z_now));
            // }
    }
    g_array[i] = g_array[i]*dz;
    // printf("%.3f \n",g_array[i]);
    }
  g_array[0] = 1.;
  // Compute normalization of power spectrum
  norm=1.;
  norm*=sigma8/sigmam(8.,0);
  // Computation of non-linear scales on a grid
  printf("Computing nonlinear scales...\n");
  // Fill the allocated arrays
  for(int i=0; i<n_redshift_bins;i++){
    printf("\b\b\b\b\b\b\b\b %3d/%3d",i,n_redshift_bins);
    fflush(stdout);
    z_now = i*dz;
    
    D1_array[i]=lgr(z_now)/lgr(0.);   // linear growth factor
    r_sigma_array[i]=calc_r_sigma(D1_array[i]);  // =1/k_NL [Mpc/h] in Eq.(B1)
    double d1 = -2.*pow(D1_array[i]*sigmam(r_sigma_array[i],2),2);
    n_eff_array[i]=-3.+2.*pow(D1_array[i]*sigmam(r_sigma_array[i],2),2);   // n_eff in Eq.(B2)
    ncur_array[i] = d1*d1+4.*sigmam(r_sigma_array[i],3)*pow(D1_array[i],2);
  }
  std::cout<<"Finished setting Cosmology"<<std::endl;
}


// Stuff needed for limber integration
double BispectrumCalculator::E(double z)
{   //assuming flat universe
    return sqrt(om*pow(1+z,3)+ow);
}

double BispectrumCalculator::E_inv(double z)
{
    return 1./E(z);
}

double BispectrumCalculator::f_K_at_z(double z)
{
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(10000);
    double result,error;

    BispectrumCalculator* ptr2 = this;
    auto ptr = [=](double x)->double{return ptr2->E_inv(x);};
    gsl_function_pp<decltype(ptr)> Fp(ptr);
    gsl_function *F = static_cast<gsl_function*>(&Fp);
    gsl_integration_qag(F, 0, z, 0, 1.0e-6, 10000, 1, w, &result, &error);
    gsl_integration_workspace_free(w);
    // printf("%.3f -- %.3f \n",z,result);
    // printf("%.3f, %.3f \n",om,ow);
    // result = GQ96_of_Einv(0,z);

    return c_over_H0*result;
}

void BispectrumCalculator::read_nofz(char filename[255]){
    // TODO: n_z_array_z and n_z_array_data are not allocated yet!
    FILE *fp;

    printf("Reading nofz \n");

    fp = fopen(filename,"r");

    if(fp == NULL){
        perror("Error while opening file. \n");
        exit(EXIT_FAILURE);
    }

    for(int i=0;i<len_n_z_array;i++){
        fscanf(fp, "%lf %lf",&n_z_array_z[i],&n_z_array_data[i]);
    }

    fclose(fp);

    normalize_nofz();
    return;
}

void BispectrumCalculator::normalize_nofz(){
    double dz_read = n_z_array_z[1]-n_z_array_z[0];
    double sum_z = 0;
    for(int i=0;i<len_n_z_array;i++){
        sum_z += n_z_array_data[i];
    }
    sum_z *= dz_read;
    for(int i=0;i<len_n_z_array;i++){
        n_z_array_data[i] = n_z_array_data[i]/sum_z;
    }
}

double BispectrumCalculator::n_of_z(double z){
    if(z<=0 || z>=z_max) return 0;
    if(slics)
    {
      // Here the correct n(z) for Euclid-like simulations.
      return (1.7865*(pow(z,0.4710)+pow(z,0.4710*5.1843))/(pow(z,5.1843)+0.7259))/2.97653;
      // this is a different n(z), not the one used for our simulations. That one is above.
      // return pow(z,2)*exp(-pow(z/0.637,1.5))*5.80564; //normalization 5.80564 such that int_0^3 dz n(z) is 1
    }
    else
    {
        if(z>=1 && z<1+dz) return 1./dz;
        else return 0;
    }

    // if(z<0 || z>=z_max) return 0;
    // double diff_z = z/z_max*(len_n_z_array-1.);
    // int pos_z = diff_z;
    // diff_z = diff_z-pos_z;
    // return n_z_array_data[pos_z]*(1.-diff_z)+n_z_array_data[pos_z+1]*diff_z;
}

double BispectrumCalculator::bispectrum_analytic_single_a(double l1, double l2, double phi, double a){
        // FOR TESTING
	    double bis_pref = -pow(M_PI,2)/(8*72*pow(a,8));
	    double pref = pow(l1,2)*pow(l2,2)*(pow(l1,2)+pow(l2,2)+2*l1*l2*cos(phi));
	    double expon = exp(-(pow(l1,2)+pow(l2,2)+l1*l2*cos(phi))/(6*a));
	//     if(isnan(bis_pref*pref*expon/pow(2.*M_PI,3.)/2.)) printf("%lf %lf %lf -- %lf %lf %lf\n",l1,l2,phi, bis_pref, pref, expon);
	    return bis_pref*pref*expon; // /pow(2.*M_PI,3.)/2.;
}


double BispectrumCalculator::bkappa(double ell1,double ell2, double ell3){
  if(ell1 == 0 || ell2 == 0 || ell3 == 0) return 0; //WARNING! THIS MIGHT SCREW WITH THE INTEGRATION ROUTINE!
  if(test_analytical){
		double phi = acos((ell1*ell1+ell2*ell2-ell3*ell3)/(2*ell1*ell2));
    
    double weights[9] = {0,1.0e+8,1.0e+6,1.0e+4,1.0e+2,1.0e+0,5.0e-3,1.0e-5,1.0e-6};
    double a_vals[9] = {3.0e+3,1.0e+4,3.0e+4,1.0e+5,3.0e+5,1.0e+6,3.0e+6,1.0e+7,3.0e+7};
    
    double temp = 0;
    for(int i=0;i<9;i++)
    {
        temp += weights[i]*bispectrum_analytic_single_a(ell1,ell2,phi,a_vals[i]);
    }
    return temp*(3.*pow(2*M_PI,2));
  }
  else{
    assert(isfinite(ell1) && isfinite(ell2) && isfinite(ell3));

    double result;
    double prefactor = 27./8. * pow(om,3) * pow(H0_over_c,5);
    struct ell_params ells = {ell1,ell2,ell3};
       

    // this is a fixed 96-point gaussian quadrature integral. This is threadsafe.
    result = GQ96_of_bdelta(0,z_max,ells);
    
    // This computes an adaptive gaussian quadrature integral. It is NOT threadsafe.
    /*    BispectrumCalculator* ptr2 = this;
    auto ptr = [=](double x)->double{return ptr2->integrand_bkappa(x,ells);};
    gsl_function_pp<decltype(ptr)> Fp(ptr);
    gsl_function *F = static_cast<gsl_function*>(&Fp);   

    size_t neval;
    gsl_integration_qng(F, 0, z_max, 0, 1.0e-3, &result, &error, &neval);

    // gsl_set_error_handler_off();
    // int status = gsl_integration_qag(F, 0, z_max, 0, 1.0e-4, 1000, 1, w_bkappa, &result, &error);
    // if(status)
    // {
    //   fprintf(stderr,"Error at ells: %.6e %.6e %.6e \n",ells.ell1,ells.ell2,ells.ell3);
    //      std::cerr<<"Errorcode:"<<gsl_strerror(status)<<std::endl;
    //  }*/
    
    return prefactor*result;
    }
}


double BispectrumCalculator::integrand_bkappa(double z, ell_params p){
    double ell1 = (p.ell1);
    double ell2 = (p.ell2);
    double ell3 = (p.ell3);

  if(test_analytical)
  { /* Returns bkappa_analytical if z<1, else returns 0 */
    if(z>1) return 0; //fix normalization: int_0^1 f(y) dx = f(y)
		double phi = M_PI-acos((ell1*ell1+ell2*ell2-ell3*ell3)/(2*ell1*ell2));
    if(isnan(phi))
    {
      phi=0;
      // std::cerr << "phi is nan in integrand_bkappa. (ell1,ell2,ell3)=" << ell1 << ", " << ell2 << ", " << ell3 << std::endl;
    } 
    
    double weights[9] = {0,1.0e+8,1.0e+6,1.0e+4,1.0e+2,1.0e+0,5.0e-3,1.0e-5,1.0e-6};
    double a_vals[9] = {3.0e+3,1.0e+4,3.0e+4,1.0e+5,3.0e+5,1.0e+6,3.0e+6,1.0e+7,3.0e+7};
    
    double temp = 0;
    for(int i=0;i<9;i++)
    {
        temp += weights[i]*bispectrum_analytic_single_a(ell1,ell2,phi,a_vals[i]);
    }
    temp /= (27./8.*pow(get_om(),3)*pow(100./299792.,5)); //prefactor from limber integration

    return temp*3.;
  }
  else 
  {
    if(z==0) return 0.;
    if(ell3==0) return 0.;
    double didx = z/z_max*(n_redshift_bins-1);
    int idx = didx;
    didx = didx - idx;
    if(idx==n_redshift_bins-1){
        idx = n_redshift_bins-2;
        didx = 1.;
    }
    double g_value = g_interpolated(idx,didx);
    double f_K_value = f_K_interpolated(idx,didx);
    if(f_K_value==0)
      {
	std::cerr<<"integrand_bkappa: f_K becomes 0"<<std::endl;
	std::cerr<<"idx:"<<idx<<std::endl;
	std::cerr<<"didx:"<<didx<<std::endl;
	std::cerr<<"z:"<<z<<std::endl;
	exit(1);
      };
    double result = pow(g_value*(1.+z),3)*bispec(ell1/f_K_value,ell2/f_K_value,ell3/f_K_value,z,idx,didx)/f_K_value/E(z);

    if(!isfinite(result))
      {
	std::cout<<g_value<<" "
		 <<z<<" "
		 <<ell1<<" "
		 <<ell2<<" "
		 <<ell3<<" "
		 <<f_K_value<<std::endl;
      };
    assert(isfinite(result));

    return result;
  }
}

double BispectrumCalculator::g_interpolated(int idx, double didx){
  // interpolates the lens efficiency
    return g_array[idx]*(1-didx) + g_array[idx+1]*didx;
}

double BispectrumCalculator::f_K_interpolated(int idx, double didx){
  // interpolates the f_K function
    return f_K_array[idx]*(1-didx) + f_K_array[idx+1]*didx;
}

double BispectrumCalculator::limber_integrand_prefactor(double z, double g_value)
{
  return 9./4.*H0_over_c*H0_over_c*H0_over_c*om*om*(1.+z)*(1.+z)*g_value*g_value/E(z);
}

double BispectrumCalculator::limber_integrand_triple_power_spectrum(double ell1, double ell2, double ell3, double z)
{
  if(z<1e-5) return 0;
  double didx = z/z_max*(n_redshift_bins-1);
  int idx = didx;
  didx = didx - idx;
  if(idx==n_redshift_bins-1){
      idx = n_redshift_bins-2;
      didx = 1.;
  }
  double g_value = g_interpolated(idx,didx);

  double f_K_value = f_K_interpolated(idx,didx);

  if(f_K_value==0)
      {
	std::cerr<<"integrand_bkappa: f_K becomes 0"<<std::endl;
	std::cerr<<"idx:"<<idx<<std::endl;
	std::cerr<<"didx:"<<didx<<std::endl;
	std::cerr<<"z:"<<z<<std::endl;
	exit(1);
      };

  double prefactor = limber_integrand_prefactor(z,g_value);
  return pow(prefactor,3)*P_k_nonlinear(ell1/f_K_value, z)*P_k_nonlinear(ell2/f_K_value, z)*P_k_nonlinear(ell3/f_K_value, z);
}

double BispectrumCalculator::limber_integrand_power_spectrum(double ell, double z, bool nonlinear)
{
  if(z<1e-5) return 0;
  double didx = z/z_max*(n_redshift_bins-1);
  int idx = didx;
  didx = didx - idx;
  if(idx==n_redshift_bins-1){
      idx = n_redshift_bins-2;
      didx = 1.;
  }
  double g_value = g_interpolated(idx,didx);

  double f_K_value = f_K_interpolated(idx,didx);

  if(f_K_value==0)
      {
	std::cerr<<"limber_integrand_power_spectrum: f_K becomes 0"<<std::endl;
	std::cerr<<"idx:"<<idx<<std::endl;
	std::cerr<<"didx:"<<didx<<std::endl;
	std::cerr<<"z:"<<z<<std::endl;
	exit(1);
      };

  double prefactor = limber_integrand_prefactor(z,g_value);
  if(nonlinear) return prefactor*P_k_nonlinear(ell/f_K_value, z);
  else return prefactor*linear_pk_at_z(ell/f_K_value, z);
}

int BispectrumCalculator::limber_integrand_power_spectrum(unsigned ndim, size_t npts, const double* vars, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim != 1)
    {
      std::cerr<<"BispectrumCalculator::limber_integrand_power_spectrum: Wrong number of function dimensions"<<std::endl;
      exit(1);
    };
  if(ndim != 1)
    {
      std::cerr<<"BispectrumCalculator::limber_integrand_power_spectrum: Wrong number of variable dimensions"<<std::endl;
      exit(1);
    };
  
  BispectrumContainer* container = (BispectrumContainer*) thisPtr;

  BispectrumCalculator* bispectrum = container->bispectrum;
  double ell = container->ell;

#if PARALLEL_INTEGRATION
#pragma omp parallel for
#endif
  for( unsigned int i=0; i<npts; i++)
    {
      double z=vars[i*ndim];
      value[i]=bispectrum->limber_integrand_power_spectrum(ell,z,true);
    }
  return 0; //Success :)
  
}

double BispectrumCalculator::convergence_power_spectrum(double ell)
{
  double result,error;
  double vals_min[1] = {0};
  double vals_max[1] = {z_max};
  BispectrumContainer container;
  container.bispectrum = this;
  container.ell = ell;
  hcubature_v(1, limber_integrand_power_spectrum, &container, 1, vals_min, vals_max, 0, 0, 1e-6, ERROR_L1, &result, &error);
  return result;
}

double delta_distrib(double ell1, double ell2, double ell3, double ell4, double ell5, double ell6)
{
  if(fabs(ell1/ell4-1.)<1e-6 && fabs(ell2/ell5-1.)<1e-6 && fabs(ell3/ell6-1.)<1e-6) return 1.;
  else return 0;
}

double delta_distrib_permutations(double ell1, double ell2, double ell3, double ell4, double ell5, double ell6)
{
  double result = 0;
  result += delta_distrib(ell1, ell2, ell3, ell4, ell5, ell6);
  result += delta_distrib(ell1, ell2, ell3, ell4, ell6, ell5);
  result += delta_distrib(ell1, ell2, ell3, ell5, ell4, ell6);
  result += delta_distrib(ell1, ell2, ell3, ell5, ell6, ell4);
  result += delta_distrib(ell1, ell2, ell3, ell6, ell4, ell5);
  result += delta_distrib(ell1, ell2, ell3, ell6, ell5, ell4);
  return result;
}

double BispectrumCalculator::bispectrumCovariance(double ell1, double ell2, double ell3, 
                                                  double ell4, double ell5, double ell6,
                                                  double delta_ell1, double delta_ell2, double delta_ell3,
                                                  double delta_ell4, double delta_ell5, double delta_ell6,
                                                  double survey_area)
{
  double product_power_spectra = convergence_power_spectrum(ell1)*convergence_power_spectrum(ell2)*convergence_power_spectrum(ell3);
  double lambda_inv = 1./number_of_triangles(ell1, ell2, ell3);
  double prefactor = pow(2*M_PI,3)/(survey_area*ell1*ell2*ell3*delta_ell1*delta_ell2*delta_ell3);
  double delta = delta_distrib_permutations(ell1, ell2, ell3, ell4, ell5, ell6);
  return prefactor*lambda_inv*delta*product_power_spectra;
}

double BispectrumCalculator::om_m_of_z(double z)
{
  double aa = 1./(1+z);
  return om/(om+aa*(aa*aa*ow+(1-om-ow)));
}

double BispectrumCalculator::om_v_of_z(double z)
{
  double aa = 1./(1+z);
  return ow*pow(aa,3)/(om+aa*(aa*aa*ow+(1-om-ow)));
}

double BispectrumCalculator::P_k_nonlinear(double k, double z){
  /* get the interpolation coefficients */
  double didx = z/z_max*(n_redshift_bins-1);
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

  // f1   = pow(om, -0.0307);
  // f2   = pow(om, -0.0585);
  // f3   = pow(om, 0.0743);


  nsqr = n_eff*n_eff;
  ncur = ncur_array[idx]*(1-didx) + ncur_array[idx+1]*didx; //interpolate ncur
  // ncur = (n_eff+3)*(n_eff+3)+4.*pow(D1,2)*sigmam(r_sigma,3)/pow(sigmam(r_sigma,1),2);
  // ncur = 0.;

  if(abs(om+ow-1)>1e-4)
  {
    std::cerr << "Warning: omw as a function of redshift only implemented for flat universes yet!";
    exit(1);
  }

  om_m = om_m_of_z(z);
  om_v = om_v_of_z(z);

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
      + 0.2250*nsqr*nsqr - 0.6038*ncur + 0.1749*om_v*(1.0 + w);
  a = pow(10.0, a);
  b = pow(10.0, -0.5642 + 0.5864*n_eff + 0.5716*nsqr - 1.5474*ncur + 0.2279*om_v*(1.0 + w));
  c = pow(10.0, 0.3698 + 2.0404*n_eff + 0.8161*nsqr + 0.5869*ncur);
  gam = 0.1971 - 0.0843*n_eff + 0.8460*ncur;
  alpha = fabs(6.0835 + 1.3373*n_eff - 0.1959*nsqr - 5.5274*ncur);
  beta  = 2.0379 - 0.7354*n_eff + 0.3157*nsqr + 1.2490*n_eff*nsqr + 0.3980*nsqr*nsqr - 0.1682*ncur;
  xnu   = pow(10.0, 5.2105 + 3.6902*n_eff);

  plin = linear_pk(k)*D1*D1*k*k*k/(2*M_PI*M_PI);


  y = k*r_sigma;
  ysqr = y*y;
  ph = a*pow(y,f1*3)/(1+b*pow(y,f2)+pow(f3*c*y,3-gam));
  ph = ph/(1+xnu/ysqr);
  pq = plin*pow(1+plin,beta)/(1+plin*alpha)*exp(-y/4.0-ysqr/8.0);
  
  delta_nl = pq + ph;

  return (2*M_PI*M_PI*delta_nl/(k*k*k));
}

double BispectrumCalculator::linear_pk_at_z(double k, double z)
{
  /* get the interpolation coefficients */
  double didx = z/z_max*(n_redshift_bins-1);
  int idx = didx;
  didx = didx - idx;
  if(idx==n_redshift_bins-1){
      idx = n_redshift_bins-2;
      didx = 1.;
  }

  double r_sigma,n_eff,D1;
  compute_coefficients(idx, didx, &D1, &r_sigma, &n_eff);
  
  return D1*D1*linear_pk(k);
  // return linear_pk(k);
}

double BispectrumCalculator::bispec(double k1, double k2, double k3, double z, int idx, double didx)   // non-linear BS w/o baryons [(Mpc/h)^6]
{
  int i,j;
  double q[4],qt,logsigma8z,r1,r2;
  double an,bn,cn,en,fn,gn,hn,mn,nn,pn,alphan,betan,mun,nun,BS1h,BS3h,PSE[4];
  double r_sigma,n_eff,D1;
  compute_coefficients(idx, didx, &D1, &r_sigma, &n_eff);

  if(z>10.) return bispec_tree(k1,k2,k3,D1); 


  
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
  logsigma8z=log10(D1*sigma8);
  
  // 1-halo term parameters in Eq.(B7)
  an=pow(10.,-2.167-2.944*logsigma8z-1.106*pow(logsigma8z,2)-2.865*pow(logsigma8z,3)-0.310*pow(r1,pow(10.,0.182+0.57*n_eff)));
  bn=pow(10.,-3.428-2.681*logsigma8z+1.624*pow(logsigma8z,2)-0.095*pow(logsigma8z,3));
  cn=pow(10.,0.159-1.107*n_eff);
  alphan=pow(10.,-4.348-3.006*n_eff-0.5745*pow(n_eff,2)+pow(10.,-0.9+0.2*n_eff)*pow(r2,2));
  if(alphan>1.-(2./3.)*ns) alphan=1.-(2./3.)*ns;
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
    PSE[i]=(1.+fn*pow(q[i],2))/(1.+gn*q[i]+hn*pow(q[i],2))*pow(D1,2)*linear_pk(q[i]/r_sigma)+1./(mn*pow(q[i],mun)+nn*pow(q[i],nun))/(1.+pow(pn*q[i],-3));  // enhanced P(k) in Eq.(B6) 
  }

  // 3-halo term bispectrum in Eq.(B5)
  BS3h=2.*(F2(k1,k2,k3,z,D1,r_sigma)*PSE[1]*PSE[2]+F2(k2,k3,k1,z,D1,r_sigma)*PSE[2]*PSE[3]+F2(k3,k1,k2,z,D1,r_sigma)*PSE[3]*PSE[1]);
  for(i=1;i<=3;i++) BS3h*=1./(1.+en*q[i]);
  // if (BS1h+BS3h>=0)
  return BS1h+BS3h;
  // else return 0;
}


double BispectrumCalculator::bispec_tree(double k1, double k2, double k3, double D1)  // tree-level BS [(Mpc/h)^6]
{
  return pow(D1,4)*2.*(F2_tree(k1,k2,k3)*linear_pk(k1)*linear_pk(k2)
		      +F2_tree(k2,k3,k1)*linear_pk(k2)*linear_pk(k3)
		      +F2_tree(k3,k1,k2)*linear_pk(k3)*linear_pk(k1));
}


double BispectrumCalculator::F2(double k1, double k2, double k3, double z, double D1, double r_sigma)
{
  double a,q[4],dn,omz,logsigma8z;

  q[3]=k3*r_sigma;
  
  logsigma8z=log10(D1*sigma8);
  a=1./(1.+z);
  omz=om/(om+ow*pow(a,-3.*w));   // Omega matter at z

  dn=pow(10.,-0.483+0.892*logsigma8z-0.086*omz);

  return F2_tree(k1,k2,k3)+dn*q[3];
}


double BispectrumCalculator::F2_tree(double k1, double k2, double k3)  // F2 kernel in tree level 
{
  double costheta12=0.5*(k3*k3-k1*k1-k2*k2)/(k1*k2);
  return (5./7.)+0.5*costheta12*(k1/k2+k2/k1)+(2./7.)*costheta12*costheta12;
}


double BispectrumCalculator::baryon_ratio(double k1, double k2, double k3, double z)   // bispectrum ratio with to without baryons  // k[h/Mpc]
{
  int i;
  double a,A0,A1,mu0,mu1,sigma0,sigma1,alpha0,alpha2,beta2,ks,k[4],x[4],Rb;

  if(z>5.) return 1.;  // baryon_ratio is calbrated at z=0-5   
  
  a=1./(1.+z);
  k[1]=k1, k[2]=k2, k[3]=k3;
  for(i=1;i<=3;i++) x[i]=log10(k[i]);  
  
  if(a>0.5) A0=0.068*pow(a-0.5,0.47);
  else A0=0.;
  mu0=0.018*a+0.837*a*a;
  sigma0=0.881*mu0;
  alpha0=2.346;
  
  if(a>0.2) A1=1.052*pow(a-0.2,1.41);
  else A1=0.;
  mu1=fabs(0.172+3.048*a-0.675*a*a);
  sigma1=(0.494-0.039*a)*mu1;
  
  ks=29.90-38.73*a+24.30*a*a;
  alpha2=2.25;
  beta2=0.563/(pow(a/0.060,0.02)+1.)/alpha2;

  Rb=1.;
  for(i=1;i<=3;i++){
    Rb*=A0*exp(-pow(fabs(x[i]-mu0)/sigma0,alpha0))-A1*exp(-pow(fabs(x[i]-mu1)/sigma1,2))+pow(1.+pow(k[i]/ks,alpha2),beta2);   // Eq.(C1)
  }

  return Rb;
}

  
double BispectrumCalculator::calc_r_sigma(double D1)  // return r_sigma[Mpc/h] (=1/k_sigma)
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


double BispectrumCalculator::linear_pk(double k)  // linear P(k)   k[h/Mpc], P(k)[(Mpc/h)^3]
{
//   if(n_data!=0) return linear_pk_data(k); 
  return linear_pk_eh(k);
}

double BispectrumCalculator::linear_pk_eh(double k)   // Eisenstein & Hu (1999) fitting formula without wiggle,      k[h/Mpc], P(k)[(Mpc/h)^3]
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

  delk=pow(norm,2)*pow(k*2997.9/h,3.+ns)*pow(L/(L+C*qeff*qeff),2);
  pk=2.0*M_PI*M_PI/(k*k*k)*delk;

  return pow(h,3)*pk;
}

double BispectrumCalculator::sigmam(double r, int j)   // r[Mpc/h]
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


double BispectrumCalculator::window(double x, int i)
{
  if(i==0) return 3.0/pow(x,3)*(sin(x)-x*cos(x));  // top hat
  if(i==1) return exp(-0.5*x*x);   // gaussian
  if(i==2) return x*exp(-0.5*x*x);  // 1st derivative gaussian
  // if(i==3) return (x*x-1.)*exp(-0.5*x*x); //2nd derivative gaussian
  if(i==3) return x*x*(1-x*x)*exp(-x*x);
  // This is only reached, if i is not a valid value between 0 and 3
  std::cerr << "BispectrumCalculator::window: Window function not specified. Exiting \n";
  exit(1);
}


double BispectrumCalculator::lgr(double z)  // linear growth factor at z (not normalized at z=0)
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


double BispectrumCalculator::lgr_func(int j, double la, double y[2])
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
  std::cerr << "BispectrumCalculator::lgr_func: j not a valid value. Exiting \n";
  exit(1);
}

void BispectrumCalculator::compute_coefficients(int idx, double didx, double *D1, double *r_sigma, double *n_eff){
    // Computes the non-linear scales on a grid that can be interpolated
    *D1 = D1_array[idx]*(1-didx) + D1_array[idx+1]*didx;
    *r_sigma = r_sigma_array[idx]*(1-didx) + r_sigma_array[idx+1]*didx;
    *n_eff = n_eff_array[idx]*(1-didx) + n_eff_array[idx+1]*didx;
}


double BispectrumCalculator::GQ96_of_Einv(double a,double b)
  { /* 96-pt Gauss qaudrature integrates E^-1(x) from a to b */
  int i;
  double cx,dx,q;
  cx=(a+b)/2;
  dx=(b-a)/2;
  q=0;
  for(i=0;i<48;i++)
    q+=W96[i]*(E_inv(cx-dx*A96[i])+E_inv(cx+dx*A96[i]));
  return(q*dx);
  }

double BispectrumCalculator::GQ96_of_bdelta(double a,double b,ell_params ells)
  { /* 96-pt Gauss qaudrature integrates bdelta(x,ells) from x=a to b */
  int i;
  double cx,dx,q;
  cx=(a+b)/2;
  dx=(b-a)/2;
  q=0;
  for(i=0;i<48;i++)
    q+=W96[i]*(integrand_bkappa(cx-dx*A96[i],ells)+integrand_bkappa(cx+dx*A96[i],ells));
  return(q*dx);
  }

double BispectrumCalculator::get_om()
  {
    return om;
  }

double BispectrumCalculator::get_z_max()
  {
    return z_max;
  }
