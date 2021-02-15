#include "gamma.hpp"

GammaCalculator::GammaCalculator(cosmology cosmo, double prec_h, double prec_k, bool fast_calculations_arg, int n_z, double z_max)
{
    Bispectrum_Class.initialize(cosmo,n_z,z_max,fast_calculations_arg);
    initialize_bessel(prec_h,prec_k);
    lp.initialize(0,8,16);
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

        // printf("%.8e, %.8e, %.8e\n",array_w[i],pi_bessel_zeros[i],array_psip[i]);
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

double GammaCalculator::angle(double x,double y){
    double res = atan2(y,x);
    if(res<0) res = res+2*M_PI;
    return res;
}


double GammaCalculator::bispectrum(double l1, double l2, double phi){
    double l3 = sqrt(l1*l1+l2*l2-2*l1*l2*cos(phi));
    if(isnan(l3))
    {
        printf("NAN in l3 with (l1,l2,phi)=(%f, %f, %f) \n",l1,l2,phi);
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

    for(int k=1;k<prec_k;k++){
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
    return angle(nenner,zahler);
}

double GammaCalculator::betabar(double psi, double phi){
    double zahler = cos(2*psi)*sin(phi);
    double nenner = cos(phi)+sin(2*psi);
    return 0.5*angle(nenner,zahler);
}

double GammaCalculator::varpsifunc(double an1, double an2, double opp){
    return acos((pow(an1,2)+pow(an2,2)-pow(opp,2))/(2.0*an1*an2));
}

std::complex<double> GammaCalculator::exponential(double x1, double x2, double x3, double psi, double phi, double varpsi)
{
    return exp(i_complex*(varpsifunc(x2,x3,x1)-varpsifunc(x1,x3,x2)-6*alpha(psi,x1,x2,phi,varpsi)));
}

std::complex<double> GammaCalculator::prefactor_phi(double psi, double phi)
{
    return exp(i_complex*betabar(psi,phi)*2.);
}

std::complex<double> GammaCalculator::integrand_phi_psi(double phi, double psi, double x1, double x2, double x3){
    double varpsi = varpsifunc(x1,x2,x3);
    double A3 = A(psi,x1,x2,phi,varpsi);
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
  double a = 40000;
  double b = 0;
  cx=(a+b)/2;
  dx=(b-a)/2;
  q=0;
  for(i=0;i<48;i++)
    q+=W96[i]*(bkappa_rcubed_j6(cx-dx*A96[i],psi,phi,A3)+bkappa_rcubed_j6(cx+dx*A96[i],psi,phi,A3));
  return(q*dx);
}


std::complex<double> GammaCalculator::GQ962D_phi_psi(double x1,double x2, double x3)
{/* 96x96-pt 2-D Gauss qaudrature integrates
            F(phi,psi,x1,x2,x3) over [0,2pi] x [0,pi/2] */
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
    std::complex<double> phase = exp(-2.*i*(phi1+phi2+phi3));
    std::complex<double> temp_result = pref*exp(-2*a*expon)*phase;
    return temp_result;
}

std::complex<double> GammaCalculator::ggg(std::complex<double> x, std::complex<double> y){
    return ggg_single_a(x,y,1.0e+6);
}
