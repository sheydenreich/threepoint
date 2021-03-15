#include "gamma.hpp"
#include "bispectrum.hpp"

GammaCalculator::GammaCalculator(cosmology cosmo, double prec_h, double prec_k, bool fast_calculations_arg, int n_z, double z_max)
{
    Bispectrum_Class.initialize(cosmo,n_z,z_max,fast_calculations_arg);
    initialize_bessel(prec_h,prec_k);
}


int GammaCalculator::initialize_bessel(double prec_h_arg, double prec_k_arg){
    /* This function is used to compute the integral int_0^infty r^3 f(r) J_6(r)*/
    // ##############################################################
    // INITIALIZE BESSEL INTEGRALS
    // ##############################################################
    printf("Initializing bessel arrays...");
    prec_h = prec_h_arg;
    prec_k = int(prec_k_arg/prec_h);
    bessel_zeros = new double[prec_k];
    pi_bessel_zeros = new double[prec_k];
    array_psi = new double[prec_k];
    array_bessel = new double[prec_k];
    array_psip = new double[prec_k];
    array_w = new double[prec_k];
    array_product = new double[prec_k];

    for(int i=0;i<prec_k;i++){
        bessel_zeros[i] = gsl_sf_bessel_zero_Jnu(6,i);
        pi_bessel_zeros[i] = bessel_zeros[i]/M_PI;
        array_psi[i] = M_PI*psi(pi_bessel_zeros[i]*prec_h)/prec_h;
        array_bessel[i] = gsl_sf_bessel_Jn(6,array_psi[i]);
        array_psip[i] = psip(prec_h*pi_bessel_zeros[i]);
        array_w[i] = 2/(M_PI*bessel_zeros[i]*pow(gsl_sf_bessel_Jn(7,bessel_zeros[i]),2));
        array_product[i] = array_w[i]*pow(array_psi[i],3)*array_bessel[i]*array_psip[i];
    }
    printf("Done \n");
    return 1;
}

double GammaCalculator::w_function(unsigned int k, double *bessel_zeros){
    double denominator;
    denominator = M_PI*bessel_zeros[k]*gsl_sf_bessel_Jn(7,bessel_zeros[k]);
    return 2/denominator;
}

double GammaCalculator::psi(double t){
    return t*tanh(M_PI*sinh(t)/2);
}

double GammaCalculator::psip(double t){
    double zahler = sinh(M_PI*sinh(t))+M_PI*t*cosh(t);
    double nenner = cosh(M_PI*sinh(t))+1;
    return zahler/nenner;
    // return tanh(0.5*M_PI*sinh(t))+0.5*M_PI*t*cosh(t); // /pow(cosh(0.5*M_PI*sinh(t)),2);
}

double GammaCalculator::bispectrum(double l1, double l2, double phi){
    double l3 = sqrt(l1*l1+l2*l2-2*l1*l2*cos(phi));
    if(isnan(l3))
    {
        // printf("NAN in l3 with (l1,l2,phi)=(%f, %f, %f) \n",l1,l2,phi);
        l3 = 0;
    } 
    return Bispectrum_Class.bkappa(l1,l2,l3)/(3.*pow(2*M_PI,2));
}


double GammaCalculator::integrated_bispec(double psi, double phi, double A3){
    /* This computes int_0^infty dr r^3 b(r/A3,phi,psi) J_6(r)*/
    double bis = 0;
    double cpsi = cos(psi);
    double spsi = sin(psi);
    double r1,r2,temp;


    for(unsigned long int k=1;k<prec_k;k++){
        r1 = array_psi[k]/A3*cpsi;
        r2 = array_psi[k]/A3*spsi;
        temp = bispectrum(r1, r2, phi)*array_product[k];
        assert(isfinite(temp));
        bis += temp;
    }

    bis = bis*M_PI;
    return bis;
}

double GammaCalculator::A(double psi, double x1, double x2, double phi, double varpsi){
    return sqrt(pow(cos(psi)*x2,2)+pow(sin(psi)*x1,2)+sin(2*psi)*x1*x2*cos(phi+varpsi));
}

double GammaCalculator::alpha(double psi, double x1, double x2, double phi, double varpsi){
    double zahler = (cos(psi)*x2-sin(psi)*x1)*sin((phi+varpsi)/2);
    double nenner = (cos(psi)*x2+sin(psi)*x1)*cos((phi+varpsi)/2);
    return atan2(zahler,nenner);
}

double GammaCalculator::betabar(double psi, double phi){
    double zahler = cos(2*psi)*sin(phi);
    double nenner = cos(phi)+sin(2*psi);
    return 0.5*atan2(zahler,nenner);
}

double GammaCalculator::varpsifunc(double an1, double an2, double opp){
    return acos((pow(an1,2)+pow(an2,2)-pow(opp,2))/(2.0*an1*an2));
}

std::complex<double> GammaCalculator::exponential(double x1, double x2, double x3, double psi, double phi, double varpsi)
{
    return exp(std::complex<double>(0,(varpsifunc(x2,x3,x1)-varpsifunc(x1,x3,x2)-6*alpha(psi,x1,x2,phi,varpsi))));
}

std::complex<double> GammaCalculator::prefactor_phi(double psi, double phi)
{
    return exp(std::complex<double>(0,betabar(psi,phi)*2.));
}

std::complex<double> GammaCalculator::integrand_phi_psi(double phi, double psi, double x1, double x2, double x3){
    double varpsi = varpsifunc(x1,x2,x3);
    double A3 = A(psi,x1,x2,phi,varpsi);
    if(isnan(A3))
    {
        printf("NAN in A3. (x1,x2,x3,phi,psi)=(%f,%f,%f,%f,%f)\n",x1,x2,x3,phi,psi);
    }
    assert(isfinite(A3));
    double bis;
    if(true)
    {
        // perform bessel integration according to Ogata et al.
        bis = integrated_bispec(psi,phi,A3);
        double A34 = pow(A3,4);
        bis/=A34;
    }
    else
    {
        // perform gaussian quadrature bessel integration
        bis = GQ96_bkappa(psi,phi,A3);
    }

    std::complex<double> prefactor = exponential(x1,x2,x3,psi,phi,varpsi)*prefactor_phi(psi,phi);
    double prefactor_psi = sin(2*psi);

    return prefactor_psi*prefactor*bis;
}

double GammaCalculator::r_integral(double phi, double psi, double x1, double x2, double x3){
    /* returns int_0^oo dr r^3 b(r cos(psi),r sin(psi), phi)J_6(A_3 r) */
    double varpsi = varpsifunc(x1,x2,x3);
    double A3 = A(psi,x1,x2,phi,varpsi);
    if(isnan(A3))
    {
        printf("NAN in A3. (x1,x2,x3,phi,psi)=(%f,%f,%f,%f,%f)\n",x1,x2,x3,phi,psi);
    }
    assert(isfinite(A3));
    double bis;
    if(true)
    {
        // perform bessel integration according to Ogata et al.
        bis = integrated_bispec(psi,phi,A3);
        double A34 = pow(A3,4);
        bis/=A34;
    }
    else
    {
        // perform gaussian quadrature bessel integration
        bis = GQ96_bkappa(psi,phi,A3);
    }

    return bis;
}

// The below functions serve to perform the (phi,psi) integration via boost's trapezoidal integral.
std::complex<double> GammaCalculator::integrand_x_psi(double x, double psi, double x1, double x2, double x3){
    /*same as integrand_phi_psi, just with substitution x=cos(phi/2). Thus, x goes from -1 to 1. */
    std::complex<double> result = integrand_phi_psi(2*acos(x),psi,x1,x2,x3)*2./sqrt(1-x*x);
    if (isnan(real(result)) || isnan(imag(result)) || isinf(real(result)) || isinf(imag(result)) )
    {
        printf("NAN with (x,psi,x1,x2,x3) = (%f,%f,%.3e,%.3e,%.3e) \n",x,psi,x1,x2,x3);
        return std::complex<double>(0,0);
    } 
    return result;
}

std::complex<double> GammaCalculator::integrand_psi(double psi, double x1, double x2, double x3){
    using boost::math::quadrature::trapezoidal;
    // auto f = [this,&psi,&x1,&x2,&x3](double x) { return integrand_x_psi(x,psi,x1,x2,x3); };
    auto freal = [this,&psi,&x1,&x2,&x3](double x) { // printf("%.3f %.3f \n",real(integrand_x_psi(x,psi,x1,x2,x3)),imag(integrand_x_psi(x,psi,x1,x2,x3)));
                                                        return real(integrand_x_psi(x,psi,x1,x2,x3)); };
    auto fimag = [this,&psi,&x1,&x2,&x3](double x) { return imag(integrand_x_psi(x,psi,x1,x2,x3)); };
    double integral_min = -1;
    double integral_max = 1.;
    std::complex<double> result = std::complex<double>(trapezoidal(freal,integral_min,integral_max),trapezoidal(fimag,integral_min,integral_max));
    return result;
}

std::complex<double> GammaCalculator::Trapz2D_phi_psi(double x1,double x2, double x3){
    using boost::math::quadrature::trapezoidal;
    namespace pl = std::placeholders;
    auto freal = [this,&x1,&x2,&x3](double psi) { return real(integrand_psi(psi,x1,x2,x3)); };
    auto fimag = [this,&x1,&x2,&x3](double psi) { return imag(integrand_psi(psi,x1,x2,x3)); };
    double integral_min = 0;
    double integral_max = M_PI/2.;
    std::complex<double> result = std::complex<double>(trapezoidal(freal,integral_min,integral_max),trapezoidal(freal,integral_min,integral_max));

    return result;
}



double GammaCalculator::bkappa_rcubed_j6(double r, double psi, double phi, double A3)
{   // returns r^3*J_6(A3*r)*b(r*cos(psi),r*sin(psi),phi)
    double r1,r2;
    r1 = r*cos(psi);
    r2 = r*sin(psi);
    return pow(r,3)*gsl_sf_bessel_Jn(6, A3*r)*bispectrum(r1,r2,phi);
}

double GammaCalculator::GQ96_bkappa(double psi, double phi, double A3)
{ /* 96-pt Gauss qaudrature integrates bkappa_rcubed_j6(x,psi,phi,A3) from x=a to x=b */
  int i;
  double cx,dx,q;
  double a = 0;
  double b = 40000;
  cx=(a+b)/2;
  dx=(b-a)/2;
  q=0;
  for(i=0;i<48;i++)
    q+=W96[i]*(bkappa_rcubed_j6(cx-dx*A96[i],psi,phi,A3)+bkappa_rcubed_j6(cx+dx*A96[i],psi,phi,A3));
  return(q*dx);
}


// std::complex<double> GammaCalculator::GQ962D_phi_psi(double x1,double x2, double x3)
// {/* 96x96-pt 2-D Gauss qaudrature integrates
//             F(phi,psi,x1,x2,x3) over [0,2pi] x [0,pi/2] */
//     int i,j,k;
//     double cx,cy,dx,dy,w,x;
//     std::complex<double> q;
//     cx=M_PI;
//     dx=cx;
//     cy=M_PI/4;
//     dy=cy;
//     q=0;
//     for(i=0;i<48;i++)
//     {
//     for(k=-1;k<=1;k+=2)
//         {
//         x=cx+k*dx*A96[i];
//         w=dy*W96[i];
//         for(j=0;j<48;j++)
//         q+=w*W96[j]*(integrand_phi_psi(x,cy-dy*A96[j],x1,x2,x3)+integrand_phi_psi(x,cy+dy*A96[j],x1,x2,x3));
//         }
//     }
//     return(q*dx);
// }

std::complex<double> GammaCalculator::GQ962D_phi_psi(double x1,double x2, double x3)
{/* 96x96-pt 2-D Gauss qaudrature integrates
            F(phi,psi,x1,x2,x3) over [0,2pi] x [0,pi/2] */
    /* right now, it is using the substitution x=-cos(phi/2). Thus, x goes from -1 to 1. */
    int i,j,k;
    double cx,cy,dx,dy,w,x;
    std::complex<double> q;
    cx=M_PI;
    dx=cx;
    cy=M_PI/4;
    dy=cy;
    q=0;
    for(i=0;i<48;i++)
    {
    for(k=-1;k<=1;k+=2)
        {
        x=cx+k*dx*A96[i];
        w=dy*W96[i];
        for(j=0;j<48;j++)
        q+=w*W96[j]*(integrand_phi_psi(x,cy-dy*A96[j],x1,x2,x3)+integrand_phi_psi(x,cy+dy*A96[j],x1,x2,x3));
        }
    }
    return(q*dx);
}


// double GammaCalculator::integrand_psi(double psi, double x1, double x2, double x3){


//     GammaCalculator* ptr2 = this;
//     {
//         auto ptr = [=](double x)->double{return ptr2->integrand_phi_real(psi,x1,x2,x3,x);};
//         gsl_function_pp<decltype(ptr)> Fp(ptr);
//         gsl_function *F = static_cast<gsl_function*>(&Fp);
//         gsl_integration_qags(F, 2*integral_border, 2*M_PI-2*integral_border, 0, 1.0e-3, 1000, w_psi, &integral1, &error);
//     }
//     {
//         auto ptr = [=](double x)->double{return ptr2->integrand_phi_real(psi,x2,x3,x1,x);};
//         gsl_function_pp<decltype(ptr)> Fp(ptr);
//         gsl_function *F = static_cast<gsl_function*>(&Fp);
//         gsl_integration_qags(F, 2*integral_border, 2*M_PI-2*integral_border, 0, 1.0e-3, 1000, w_psi, &integral2, &error);
//     }
//     {
//         auto ptr = [=](double x)->double{return ptr2->integrand_phi_real(psi,x3,x1,x2,x);};
//         gsl_function_pp<decltype(ptr)> Fp(ptr);
//         gsl_function *F = static_cast<gsl_function*>(&Fp);
//         gsl_integration_qags(F, 2*integral_border, 2*M_PI-2*integral_border, 0, 1.0e-3, 1000, w_psi, &integral3, &error);
//     }

//   return prefactor_psi*(integral1+integral2+integral3);
// }

std::complex<double> GammaCalculator::gamma0(double x1, double x2, double x3){
    std::complex<double> integral1,integral2,integral3;
    std::complex<double> result;
    integral1 = GQ962D_phi_psi(x1,x2,x3);
    integral2 = GQ962D_phi_psi(x2,x3,x1);
    integral3 = GQ962D_phi_psi(x3,x1,x2);
    result = integral1+integral2+integral3;


    
    // {
    //     auto ptr = [=](double x)->double{return ptr2->integrand_psi_real(x,x1,x2,x3);};
    //     gsl_function_pp<decltype(ptr)> Fp(ptr);
    //     gsl_function *F = static_cast<gsl_function*>(&Fp);
    //     gsl_integration_qags(F, integral_border/2., M_PI/2.-integral_border/2., 0, 1.0e-2, 1000, w_psi, &integral1, &error);
    // }
    // {
    //     auto ptr = [=](double x)->double{return ptr2->integrand_psi_imag(x,x1,x2,x3);};
    //     gsl_function_pp<decltype(ptr)> Fp(ptr);
    //     gsl_function *F = static_cast<gsl_function*>(&Fp);
    //     gsl_integration_qags(F, integral_border/2., M_PI/2.-integral_border/2., 0, 1.0e-2, 1000, w_psi, &integral2, &error);
    // }
    // std::complex<double> result(integral1,integral2);

  return result/(2*pow(2*M_PI,3));
}


std::complex<double> GammaCalculator::ggg_single_a(std::complex<double> x, std::complex<double> y, double a){
    // FOR TESTING COMPARISON
    std::complex<double> z = y-x;
    const std::complex<double> i(0,1);
    double phi1 = atan2(imag(x),real(x));
    double phi2 = atan2(-imag(y),-real(y));
    double phi3 = atan2(imag(z),real(z));
    std::complex<double> pref = pow((x-2.*y)*(x+y)*(y-2.*x),2);
    std::complex<double> expon = pow(abs(x),2)+pow(abs(y),2)-real(x)*real(y)-imag(x)*imag(y);
    std::complex<double> phase = exp(2.*i*(phi1+phi2+phi3)); //adjusted, should get correct imaginary part now
    // std::complex<double> temp_result = pref*exp(-2*a*expon)*phase;
    return pref*exp(-2*a*expon)*phase;
}

std::complex<double> GammaCalculator::ggg(std::complex<double> x, std::complex<double> y){
    double weights[9] = {0,1.0e+8,1.0e+6,1.0e+4,1.0e+2,1.0e+0,5.0e-3,1.0e-5,1.0e-6};
    double a_vals[9] = {3.0e+3,1.0e+4,3.0e+4,1.0e+5,3.0e+5,1.0e+6,3.0e+6,1.0e+7,3.0e+7};
    
    std::complex<double> temp(0,0);
    for(int i=0;i<9;i++)
    {
        temp += weights[i]*ggg_single_a(x,y,a_vals[i]);
    }
    return temp;
}

std::complex<double> GammaCalculator::integrand_r_phi_psi_one_x(double r, double phi, double psi, double x1, double x2, double x3)
{
    double varpsi = varpsifunc(x1,x2,x3);
    double A3 = A(psi,x1,x2,phi,varpsi);
    assert(isfinite(A3));
    std::complex<double> prefactor = exponential(x1,x2,x3,psi,phi,varpsi)*prefactor_phi(psi,phi);
    return sin(2*psi)*prefactor*gsl_sf_bessel_Jn(6, A3*r);
}

std::complex<double> GammaCalculator::integrand(double r, double phi, double psi, double z, double x1, double x2, double x3)
{
    double ell1 = r*cos(psi);
    double ell2 = r*sin(psi);
    double ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));
    // std::cout << ell1 << ", " << ell2 << ", " << ell3 << std::endl;
    struct ell_params ells = {ell1,ell2,ell3};
    return Bispectrum_Class.integrand_bkappa(z,ells)*pow(r,3)*(integrand_r_phi_psi_one_x(r, phi, psi, x1, x2, x3)+integrand_r_phi_psi_one_x(r, phi, psi, x2, x3, x1)+integrand_r_phi_psi_one_x(r, phi, psi, x3, x1, x2));
}

int GammaCalculator::integrand(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value)
{
    struct integration_parameter params = *((integration_parameter*) fdata);

    GammaCalculator* gammaCalculator = params.gammaCalculator;
    double x1 = params.x1;
    double x2 = params.x2;
    double x3 = params.x3;

    // std::cout << "Batch-size for evaluation: " << npts << std::endl;

    #pragma omp parallel for
    for( unsigned int i=0; i<npts; i++)
    {
      double r=vars[i*ndim];
      double phi=vars[i*ndim+1];
      double psi=vars[i*ndim+2];
      double z=vars[i*ndim+3];
      std::complex<double> temp = gammaCalculator->integrand(r,phi,psi,z,x1,x2,x3);
      value[fdim*i]=real(temp);
      value[fdim*i+1]=imag(temp);
    }

    return 0;
}



std::complex<double> GammaCalculator::gamma0_from_cubature(double x1, double x2, double x3)
{
    double vals_min[4] = {0,0,0,0};
    double vals_max[4] = {500000,2*M_PI,M_PI/2,1};
    double result[2];
    double error[2];
    struct integration_parameter params;
    params.gammaCalculator = this;
    params.x1 = x1;
    params.x2 = x2;
    params.x3 = x3;
    double epsabs = 0;
    if(test_analytical) epsabs = 1.0e-8;
    hcubature_v(2,integrand,&params,4,vals_min,vals_max,0,epsabs,1e-3,ERROR_L1,result,error);

    if(!test_analytical) std::cout << x1 << ", " << x2 << ", " << x3 << ": " << result[0] << " + " << result[1] << " i +/- " << error[0] << " + " << error[1] <<" i" << std::endl;

    return std::complex<double>(result[0],result[1]);
}

std::complex<double> GammaCalculator::integrand_ell1_ell2_phi_one_x(double ell1, double ell2, double phi, double x1, double x2, double x3)
{
    double psi = atan2(ell2,ell1);
    double varpsi = varpsifunc(x1,x2,x3);
    double A3 = A(psi,x1,x2,phi,varpsi);
    assert(isfinite(A3));
    std::complex<double> prefactor = exponential(x1,x2,x3,psi,phi,varpsi)*prefactor_phi(psi,phi);
    return ell1*ell2*prefactor*gsl_sf_bessel_Jn(6, A3*sqrt(ell1*ell1+ell2*ell2));
}

std::complex<double> GammaCalculator::integrand_ell(double ell1, double ell2, double phi, double z, double x1, double x2, double x3)
{
    double ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));
    // std::cout << ell1 << ", " << ell2 << ", " << ell3 << std::endl;
    struct ell_params ells = {ell1,ell2,ell3};
    return Bispectrum_Class.integrand_bkappa(z,ells)*(integrand_ell1_ell2_phi_one_x(ell1, ell2, phi, x1, x2, x3)+integrand_ell1_ell2_phi_one_x(ell1, ell2, phi, x2, x3, x1)+integrand_ell1_ell2_phi_one_x(ell1, ell2, phi, x3, x1, x2));
}

int GammaCalculator::integrand_ell(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value)
{
    struct integration_parameter params = *((integration_parameter*) fdata);

    GammaCalculator* gammaCalculator = params.gammaCalculator;
    double x1 = params.x1;
    double x2 = params.x2;
    double x3 = params.x3;

    // std::cout << "Batch-size for evaluation: " << npts << std::endl;

    #pragma omp parallel for
    for( unsigned int i=0; i<npts; i++)
    {
      double ell1=vars[i*ndim];
      double ell2=vars[i*ndim+1];
      double phi=vars[i*ndim+2];
      double z=vars[i*ndim+3];
      std::complex<double> temp = gammaCalculator->integrand_ell(ell1,ell2,phi,z,x1,x2,x3);
      value[fdim*i]=real(temp);
      value[fdim*i+1]=imag(temp);
    }

    return 0;
    // std::cout << std::endl;
}

std::complex<double> GammaCalculator::gamma0_from_cubature_ell(double x1, double x2, double x3)
{
    double vals_min[4] = {0,0,0,0};
    double vals_max[4] = {40000,40000,2*M_PI,1};
    double result[2];
    double error[2];
    struct integration_parameter params;
    params.gammaCalculator = this;
    params.x1 = x1;
    params.x2 = x2;
    params.x3 = x3;
    double epsabs = 0;
    if(test_analytical) epsabs = 1.0e-8;
    hcubature_v(2,integrand_ell,&params,4,vals_min,vals_max,0,epsabs,1e-4,ERROR_L1,result,error);

    if(!test_analytical) std::cout << x1 << ", " << x2 << ", " << x3 << ": " << result[0] << " + " << result[1] << " i +/- " << error[0] << " + " << error[1] <<" i" << std::endl;

    return std::complex<double>(result[0],result[1]);
}



double GammaCalculator::integrated_bdelta_times_rcubed_J6(double z, double phi, double psi, double A3)
{ /* this computes int_0^oo dR R^3 b_delta(R cos(psi), R sin(psi), phi) J_6(A_3' R) */
    double bis = 0;
    double cpsi = cos(psi);
    double spsi = sin(psi);
    double ell1,ell2,ell3,temp;

    for(unsigned long int k=1;k<prec_k;k++){
        ell1 = array_psi[k]/A3*cpsi;
        ell2 = array_psi[k]/A3*spsi;
        ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));
        struct ell_params ells = {ell1,ell2,ell3};

        temp = Bispectrum_Class.integrand_bkappa(z,ells)*array_product[k];
        assert(isfinite(temp));
        bis += temp;
    }

    bis = bis*M_PI;
    return bis;
}

std::complex<double> GammaCalculator::integrand_z_phi_psi_one_x(double z, double phi, double psi, double x1, double x2, double x3)
{
    double varpsi = varpsifunc(x1,x2,x3);
    double A3 = A(psi,x1,x2,phi,varpsi);
    assert(isfinite(A3));
    std::complex<double> prefactor = sin(2*psi)*exponential(x1,x2,x3,psi,phi,varpsi)*prefactor_phi(psi,phi)/pow(A3,4);
    std::cout << integrated_bdelta_times_rcubed_J6(z, phi, psi, A3) << std::endl;
    return prefactor*integrated_bdelta_times_rcubed_J6(z, phi, psi, A3);
}

std::complex<double> GammaCalculator::integrand_z_phi_psi(double z, double phi, double psi, double x1, double x2, double x3)
{
    return integrand_z_phi_psi_one_x(z, phi, psi, x1, x2, x3)+integrand_z_phi_psi_one_x(z, phi, psi, x2, x3, x1)+integrand_z_phi_psi_one_x(z, phi, psi, x3, x1, x2);
}

int GammaCalculator::integrand_2(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value)
{
    struct integration_parameter params = *((integration_parameter*) fdata);

    GammaCalculator* gammaCalculator = params.gammaCalculator;
    double x1 = params.x1;
    double x2 = params.x2;
    double x3 = params.x3;

    #pragma omp parallel for
    for( unsigned int i=0; i<npts; i++)
    {
      double z=vars[i*ndim];
      double phi=vars[i*ndim+1];
      double psi=vars[i*ndim+2];
      std::complex<double> temp = gammaCalculator->integrand_z_phi_psi(z, phi, psi, x1, x2, x3);
      value[i*fdim]=real(temp);
      value[i*fdim+1] = imag(temp);
    }

    return 0;
}


std::complex<double> GammaCalculator::gamma0_from_cubature_and_ogata(double x1, double x2, double x3)
{
    double vals_min[3] = {0,0,0};
    double vals_max[3] = {1,2*M_PI,M_PI/2};
    double result[2];
    double error[2];
    struct integration_parameter params;
    params.gammaCalculator = this;
    params.x1 = x1;
    params.x2 = x2;
    params.x3 = x3;
    double epsabs = 0;
    if(test_analytical) epsabs = 1.0e-8;
    hcubature_v(2,integrand_2,&params,3,vals_min,vals_max,0,epsabs,1e-4,ERROR_L1,result,error);
    if(!test_analytical) std::cout << x1 << ", " << x2 << ", " << x3 << ": " << result[0] << " + " << result[1] << " i +/- " << error[0] << " + " << error[1] <<" i" << std::endl;

    return std::complex<double>(result[0],result[1]);
}