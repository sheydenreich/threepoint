#include "gamma.hpp"
#include "bispectrum.hpp"
#include <cmath>
#include <gsl/gsl_sf_bessel.h>

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
    prec_k = uint(prec_k_arg/prec_h);
    bessel_zeros = new double[prec_k];
    pi_bessel_zeros = new double[prec_k];
    array_psi = new double[prec_k];
    array_psi_J2 = new double[prec_k];
    array_bessel = new double[prec_k];
    array_psip = new double[prec_k];
    array_w = new double[prec_k];
    array_product = new double[prec_k];
    array_product_J2 = new double[prec_k];

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
    printf("Done \n");
    return 1;
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
        printf("NAN in l3 with (l1,l2,phi)=(%f, %f, %f) \n",l1,l2,phi);
        l3 = 0;
    } 
    return Bispectrum_Class.bkappa(l1,l2,l3)/3.;
}


double GammaCalculator::integrated_bispec(double psi, double phi, double A3){
    /* This computes int_0^infty dr r^3 b(r/A3,phi,psi) J_6(r)*/
    double bis = 0;
    double cpsi = cos(psi);
    double spsi = sin(psi);
    double r1,r2,temp;

    for(unsigned int k=1;k<prec_k;k++){
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


std::complex<double> GammaCalculator::gamma0(double x1, double x2, double x3){
    std::complex<double> result;

    if(USE_CUBATURE)
    {
        if(USE_OGATA)
        {
            result = gamma0_from_cubature_and_ogata(x1, x2, x3);
        }
        else
        {
            if(PERFORM_ELL1_ELL2)
            {
                result = gamma0_from_cubature_ell(x1, x2, x3);
            }
            else
            {
                result = gamma0_from_cubature(x1, x2, x3);
            }
        }
        result *= 27./8.*pow(Bispectrum_Class.get_om(),3)*pow(100./299792.,5); //prefactor from limber integration
        result /= 3.; //prefactor for bispectrum definition in Eq.(4) of Schneider et al. (2005)
        result /= (2*pow(2*M_PI,3)); //prefactor from modified Eq.(15) of Schneider et al. (2005)
        return result;
    }

    else
    {
        std::complex<double> integral1,integral2,integral3;
        integral1 = GQ962D_phi_psi(x1,x2,x3);
        integral2 = GQ962D_phi_psi(x2,x3,x1);
        integral3 = GQ962D_phi_psi(x3,x1,x2);
        result = integral1+integral2+integral3;

        return result/(2*pow(2*M_PI,3)); //prefactor from modified Eq.(15) of Schneider et al. (2005)
    }
}

std::complex<double> GammaCalculator::gamma1(double x1, double x2, double x3){
    std::complex<double> result;

    if(USE_CUBATURE)
    {
        if(USE_OGATA)
        {
            result = gamma1_from_cubature_and_ogata(x1, x2, x3);
        }
        else
        {
            result = gamma1_from_cubature(x1, x2, x3);

            // std::cerr << "Not yet implemented!" << std::endl;
            // return -1;
            // if(PERFORM_ELL1_ELL2)
            // {
            //     result = gamma0_from_cubature_ell(x1, x2, x3);
            // }
            // else
            // {
            //     result = gamma0_from_cubature(x1, x2, x3);
            // }
        }
        result *= 27./8.*pow(Bispectrum_Class.get_om(),3)*pow(100./299792.,5); //prefactor from limber integration
        result /= 3.; //prefactor for bispectrum definition in Eq.(4) of Schneider et al. (2005)
        result /= (2*pow(2*M_PI,3)); //prefactor from modified Eq.(15) of Schneider et al. (2005)
        return result;
    }

    else
    {
        std::cerr << "Not yet implemented!" << std::endl;
        return -1;
        // std::complex<double> integral1,integral2,integral3;
        // integral1 = GQ962D_phi_psi(x1,x2,x3);
        // integral2 = GQ962D_phi_psi(x2,x3,x1);
        // integral3 = GQ962D_phi_psi(x3,x1,x2);
        // result = integral1+integral2+integral3;

        // return result/(2*pow(2*M_PI,3)); //prefactor from modified Eq.(15) of Schneider et al. (2005)
    }
}

std::complex<double> GammaCalculator::gamma2(double x1, double x2, double x3){
    return gamma1(x2,x3,x1);
}

std::complex<double> GammaCalculator::gamma3(double x1, double x2, double x3){
    return gamma1(x3,x1,x2);
}



std::complex<double> GammaCalculator::gggstar_single_a(double r1, double r2, double r3, long double a)
{
    std::complex<long double> x(r2,0);
    long double height = (r1+r2+r3)*(r2+r3-r1)*(r1-r2+r3)*(r1+r2-r3);
    height = sqrt(height)/(2.*r2);
    long double rest_of_r1 = sqrt(r1*r1-height*height);
    std::complex<long double> y(-rest_of_r1,-height);

    long double X1 = real(x);
    long double X2 = imag(x);
    long double Y1 = real(y);
    long double Y2 = imag(y);

    std::complex<long double> z = -y-x;
    long double phi1 = atan2(imag(x),real(x));
    long double phi2 = atan2(imag(y),real(y));
    long double phi3 = atan2(imag(z),real(z));

    const std::complex<long double> i(0,1);

    std::complex<long double> phase = exp(static_cast<long double>(-2.)*i*(phi1+phi2-phi3));

    std::complex<long double> gggstar = (static_cast<long double>(4.)*pow(a,2)*pow(X1,6) - static_cast<long double>(4.)*pow(a,2)*pow(X2,6) + static_cast<long double>(4)*pow(a,2)*pow(X2,5)*(std::complex<long double>(0,7.)*Y1 - static_cast<long double>(3.)*Y2) 
    + static_cast<long double>(4)*pow(a,2)*pow(X1,5)*(std::complex<long double>(0,2)*X2 + static_cast<long double>(3)*Y1 + std::complex<long double>(0,7)*Y2) + 
     a*pow(X1,4)*(static_cast<long double>(8.) + a*(static_cast<long double>(4.)*pow(X2,2) - std::complex<long double>(0,4)*X2*Y1 - static_cast<long double>(3.)*pow(Y1,2) 
     - static_cast<long double>(44.)*X2*Y2 + std::complex<long double>(0,58.)*Y1*Y2 - static_cast<long double>(77.)*pow(Y2,2))) + 
     std::complex<long double>(0,4.)*X2*(Y1 + std::complex<long double>(0,1)*Y2)*(static_cast<long double>(-6) + pow(a,2)*(Y1 - std::complex<long double>(0,1)*Y2)*pow(Y1 + std::complex<long double>(0,1)*Y2,2)*(static_cast<long double>(7.)*Y1 - std::complex<long double>(0,3)*Y2) 
     + a*(static_cast<long double>(5.)*pow(Y1,2) + std::complex<long double>(0,4)*Y1*Y2 + pow(Y2,2))) + 
     a*pow(X2,4)*(static_cast<long double>(-8.) + a*(77.*pow(Y1,2) + std::complex<long double>(0,58.)*Y1*Y2 + static_cast<long double>(3.)*pow(Y2,2))) + static_cast<long double>(2.)*a*pow(X2,3)*(std::complex<long double>(0,-53.)*a*pow(Y1,3) - static_cast<long double>(2.)*Y2 + static_cast<long double>(53.)*a*pow(Y1,2)*Y2 + static_cast<long double>(13.)*a*pow(Y2,3) + std::complex<long double>(0,1)*Y1*(static_cast<long double>(10.) - static_cast<long double>(13.)*a*pow(Y2,2))) + 
     static_cast<long double>(2.)*pow(Y1 + std::complex<long double>(0,1)*Y2,2)*(static_cast<long double>(-3) + static_cast<long double>(2)*a*(pow(Y1,2) + 
     pow(Y2,2))*(static_cast<long double>(2.) + a*(pow(Y1,2) + pow(Y2,2)))) - pow(X2,2)*(static_cast<long double>(-6.) + a*(Y1 + std::complex<long double>(0,1)*Y2)*(std::complex<long double>(0,24)*Y2 + a*(Y1 + std::complex<long double>(0,1)*Y2)*
     (static_cast<long double>(77.)*pow(Y1,2) - std::complex<long double>(0,58)*Y1*Y2 + static_cast<long double>(3.)*pow(Y2,2)))) + 
     std::complex<long double>(0,2)*a*pow(X1,3)*(static_cast<long double>(8.)*a*pow(X2,3) - std::complex<long double>(0,2)*Y1 + static_cast<long double>(10.)*Y2 + 
     static_cast<long double>(4.)*a*pow(X2,2)*(std::complex<long double>(0,-7)*Y1 + static_cast<long double>(3.)*Y2) + std::complex<long double>(0,1)*a*(Y1 + std::complex<long double>(0,1)*Y2)*
     (static_cast<long double>(13.)*pow(Y1,2) + static_cast<long double>(53.)*pow(Y2,2)) - 
        static_cast<long double>(8.)*X2*(static_cast<long double>(-1.) + a*(static_cast<long double>(4.)*pow(Y1,2) + std::complex<long double>(0,2)*Y1*Y2 
        + static_cast<long double>(6.)*pow(Y2,2)))) + pow(X1,2)*(static_cast<long double>(-6.) - std::complex<long double>(0,12)*a*(X2*(Y1 - std::complex<long double>(0,3)*Y2) 
        + static_cast<long double>(2.)*Y1*(std::complex<long double>(0,-1)*Y1 + Y2)) + pow(a,2)*
         (-4.*pow(X2,4) + 8.*pow(X2,3)*(std::complex<long double>(0,3)*Y1 - 7.*Y2) + 6.*pow(X2,2)*(7.*pow(Y1,2) + std::complex<long double>(0,30)*Y1*Y2 - 7.*pow(Y2,2)) + 
           2.*X2*(std::complex<long double>(0,-1)*Y1 + Y2)*(13.*pow(Y1,2) + std::complex<long double>(0,80)*Y1*Y2 + 53.*pow(Y2,2)) - pow(Y1 + std::complex<long double>(0,1)*Y2,2)*(3.*pow(Y1,2) + std::complex<long double>(0,58)*Y1*Y2 + 77*pow(Y2,2)))) + 
     2.*X1*(std::complex<long double>(0,4)*pow(a,2)*pow(X2,5) + 2.*pow(a,2)*pow(X2,4)*(11.*Y1 - std::complex<long double>(0,1)*Y2) + 
        static_cast<long double>(2.)*(static_cast<long double>(-6.) + a*(Y1 + std::complex<long double>(0,1)*Y2)*(Y1 - std::complex<long double>(0,5)*Y2) + pow(a,2)*(Y1 - std::complex<long double>(0,1)*Y2)*pow(Y1 + std::complex<long double>(0,1)*Y2,2)*(3.*Y1 - std::complex<long double>(0,7)*Y2))*(Y1 + std::complex<long double>(0,1)*Y2) + 
        std::complex<long double>(0,1)*X2*(static_cast<long double>(-6.) + pow(a,2)*pow(Y1 + std::complex<long double>(0,1)*Y2,2)*(29.*pow(Y1,2) - std::complex<long double>(0,74)*Y1*Y2 - 29.*pow(Y2,2)) - 12.*a*(pow(Y1,2) + pow(Y2,2))) - 
        std::complex<long double>(0,8)*a*pow(X2,3)*(static_cast<long double>(-1.) + a*(6*pow(Y1,2) - std::complex<long double>(0,2)*Y1*Y2 + 4.*pow(Y2,2))) - 
        a*pow(X2,2)*(static_cast<long double>(53)*a*pow(Y1,3) - std::complex<long double>(0,27)*a*pow(Y1,2)*Y2 + std::complex<long double>(0,1)*Y2*(static_cast<long double>(6) + static_cast<long double>(13)*a*pow(Y2,2)) + 3.*Y1*(-6. + 31*a*pow(Y2,2)))))/
   (pow(a,2)*exp(2*a*(pow(X1,2) + pow(X2,2) + X1*Y1 + pow(Y1,2) + X2*Y2 + pow(Y2,2))));

    return std::complex<double>(real(phase*gggstar),imag(phase*gggstar));

}

std::complex<double> GammaCalculator::ggg_single_a(double r1, double r2, double r3, double a){
    // FOR TESTING COMPARISON
    std::complex<double> x(r2,0);
    double height = (r1+r2+r3)*(r2+r3-r1)*(r1-r2+r3)*(r1+r2-r3);
    height = sqrt(height)/(2.*r2);
    double rest_of_r1 = sqrt(r1*r1-height*height);
    std::complex<double> y(-rest_of_r1,-height);

    std::complex<double> z = -x-y;
    const std::complex<double> i(0,1);
    double phi1 = atan2(imag(x),real(x));
    double phi2 = atan2(imag(y),real(y));
    double phi3 = atan2(imag(z),real(z));
    std::complex<double> pref = pow((x+2.*y)*(x-y)*(y+2.*x),2);
    std::complex<double> expon = pow(abs(x),2)+pow(abs(y),2)+real(x)*real(y)+imag(x)*imag(y);
    std::complex<double> phase = exp(-2.*i*(phi1+phi2+phi3));
    // std::complex<double> temp_result = pref*exp(-2*a*expon)*phase;
    return pref*exp(-2*a*expon)*phase;
}

std::complex<double> GammaCalculator::ggg(double r1, double r2, double r3){
    double weights[9] = {0,1.0e+8,1.0e+6,1.0e+4,1.0e+2,1.0e+0,5.0e-3,1.0e-5,1.0e-6};
    double a_vals[9] = {3.0e+3,1.0e+4,3.0e+4,1.0e+5,3.0e+5,1.0e+6,3.0e+6,1.0e+7,3.0e+7};
    
    std::complex<double> temp(0,0);
    for(int i=0;i<9;i++)
    {
        temp += weights[i]*ggg_single_a(r1,r2,r3,a_vals[i]);
    }
    return temp;
}

std::complex<double> GammaCalculator::gggstar(double r1, double r2, double r3){
    double weights[9] = {0,1.0e+8,1.0e+6,1.0e+4,1.0e+2,1.0e+0,5.0e-3,1.0e-5,1.0e-6};
    double a_vals[9] = {3.0e+3,1.0e+4,3.0e+4,1.0e+5,3.0e+5,1.0e+6,3.0e+6,1.0e+7,3.0e+7};
    
    std::complex<double> temp(0,0);
    for(int i=0;i<9;i++)
    {
        temp += weights[i]*gggstar_single_a(r1,r2,r3,a_vals[i]);
    }
    return temp;
}

std::complex<double> GammaCalculator::ggstarg(double r1, double r2, double r3){
    return gggstar(r3,r1,r2);
}

std::complex<double> GammaCalculator::gstargg(double r1, double r2, double r3){
    return gggstar(r2,r3,r1);
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
    if(test_analytical) epsabs = 1.0e-7;
    hcubature_v(2,integrand,&params,4,vals_min,vals_max,0,epsabs,1e-4,ERROR_L1,result,error);

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
    if(test_analytical) epsabs = 1.0e-7;
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

    for(unsigned int k=1;k<prec_k;k++){
        ell1 = array_psi[k]/A3*cpsi;
        ell2 = array_psi[k]/A3*spsi;
        ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));
        struct ell_params ells = {ell1,ell2,ell3};

        temp = Bispectrum_Class.integrand_bkappa(z,ells)*array_product[k];
        assert(isfinite(temp));
        bis += temp;
    }

    bis = bis*M_PI/pow(A3,4);
    return bis;
}

double GammaCalculator::integrated_bdelta_times_rcubed_J2(double z, double phi, double psi, double A3)
{ /* this computes int_0^oo dR R^3 b_delta(R cos(psi), R sin(psi), phi) J_2(A_3' R) */
    double bis = 0;
    double cpsi = cos(psi);
    double spsi = sin(psi);
    double ell1,ell2,ell3,temp;

    for(unsigned int k=1;k<prec_k;k++){
        ell1 = array_psi_J2[k]/A3*cpsi;
        ell2 = array_psi_J2[k]/A3*spsi;
        ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));
        struct ell_params ells = {ell1,ell2,ell3};

        temp = Bispectrum_Class.integrand_bkappa(z,ells)*array_product_J2[k];
        assert(isfinite(temp));
        bis += temp;
    }

    bis = bis*M_PI/pow(A3,4);
    return bis;
}

double GammaCalculator::testfunction_integrate_J2()
{
    double bis = 0;
    double temp,x;
    for(unsigned int k=1;k<prec_k;k++){
        x = array_psi_J2[k];
        temp = pow(x,2)*exp(-x)*array_product_J2[k]/pow(x,3);
        assert(isfinite(temp));
        bis += temp;
    }
    return bis*M_PI;
}

std::complex<double> GammaCalculator::integrand_z_phi_psi_gamma1(double z, double phi, double psi, double x1, double x2, double x3)
{

    double varbetabar = betabar(psi,phi);

    double varpsi1 = varpsifunc(x2,x3,x1);
    double varpsi2 = varpsifunc(x3,x1,x2);
    double varpsi3 = varpsifunc(x1,x2,x3);

    double A1 = A(psi,x2,x3,phi,varpsi1);
    double A2 = A(psi,x3,x1,phi,varpsi2);
    double A3 = A(psi,x1,x2,phi,varpsi3);

    double alpha1 = alpha(psi, x2, x3, phi, varpsi1);
    double alpha2 = alpha(psi, x3, x1, phi, varpsi2);
    double alpha3 = alpha(psi, x1, x2, phi, varpsi3);
    
    std::complex<double> integrand_3(0,varpsi1-varpsi2+2*varpsi3+2*(varbetabar-phi-alpha3));
    integrand_3 = exp(integrand_3);
    integrand_3 *= integrated_bdelta_times_rcubed_J2(z,phi,psi,A3);

    std::complex<double> integrand_1(0,varpsi3-varpsi2-2*(varbetabar+alpha1));
    integrand_1 = exp(integrand_1);
    integrand_1 *= integrated_bdelta_times_rcubed_J2(z,phi,psi,A1);

    std::complex<double> integrand_2(0,varpsi3-varpsi1-2*varpsi2+2*(varbetabar+phi-alpha2));
    integrand_2 = exp(integrand_2);
    integrand_2 *= integrated_bdelta_times_rcubed_J2(z,phi,psi,A2);

    return sin(2*psi)*(integrand_1+integrand_2+integrand_3);
}

std::complex<double> GammaCalculator::integrand_r_phi_psi_gamma1(double r, double phi, double psi, double x1, double x2, double x3)
{

    double varbetabar = betabar(psi,phi);

    double varpsi1 = varpsifunc(x2,x3,x1);
    double varpsi2 = varpsifunc(x3,x1,x2);
    double varpsi3 = varpsifunc(x1,x2,x3);

    double A1 = A(psi,x2,x3,phi,varpsi1);
    double A2 = A(psi,x3,x1,phi,varpsi2);
    double A3 = A(psi,x1,x2,phi,varpsi3);

    double alpha1 = alpha(psi, x2, x3, phi, varpsi1);
    double alpha2 = alpha(psi, x3, x1, phi, varpsi2);
    double alpha3 = alpha(psi, x1, x2, phi, varpsi3);

    double ell1,ell2,ell3;
    ell1 = r*cos(psi);
    ell2 = r*sin(psi);
    ell3 = sqrt(ell1*ell1+ell2*ell2-2*ell1*ell2*cos(phi));

    
    std::complex<double> integrand_3(0,varpsi1-varpsi2+2*varpsi3+2*(varbetabar-phi-alpha3));
    integrand_3 = exp(integrand_3);
    integrand_3 *= gsl_sf_bessel_Jn(2, A3*r);

    std::complex<double> integrand_1(0,varpsi3-varpsi2-2*(varbetabar+alpha1));
    integrand_1 = exp(integrand_1);
    integrand_1 *= gsl_sf_bessel_Jn(2, A1*r);

    std::complex<double> integrand_2(0,varpsi3-varpsi1-2*varpsi2+2*(varbetabar+phi-alpha2));
    integrand_2 = exp(integrand_2);
    integrand_2 *= gsl_sf_bessel_Jn(2, A2*r);
    
    double prefactor = 27./8.*pow(Bispectrum_Class.get_om(),3)*pow(100./299792.,5)/(3*2*pow(2*M_PI,3));
    // prefactor = 1;

    // std::cout << abs(Bispectrum_Class.bkappa(ell1,ell2,ell3)*pow(r,3)*(integrand_1+integrand_2+integrand_3)/prefactor) << std::endl;

    return Bispectrum_Class.bkappa(ell1,ell2,ell3)*pow(r,3)*sin(2*psi)*(integrand_1+integrand_2+integrand_3)/prefactor;
}

int GammaCalculator::integrand_gamma1_no_ogata_no_limber(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value)
{
    struct integration_parameter params = *((integration_parameter*) fdata);

    GammaCalculator* gammaCalculator = params.gammaCalculator;
    double x1 = params.x1;
    double x2 = params.x2;
    double x3 = params.x3;

    std::cout << "Evaluating " << npts << " points!" << std::endl;


    #pragma omp parallel for
    for( unsigned int i=0; i<npts; i++)
    {
      double r=vars[i*ndim];
      double phi=vars[i*ndim+1];
      double psi=vars[i*ndim+2];
      std::complex<double> temp = gammaCalculator->integrand_r_phi_psi_gamma1(r, phi, psi, x1, x2, x3);
      value[i*fdim]=real(temp);
      value[i*fdim+1] = imag(temp);
    }

    return 0;
}


int GammaCalculator::integrand_2_gamma1(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value)
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
      std::complex<double> temp = gammaCalculator->integrand_z_phi_psi_gamma1(z, phi, psi, x1, x2, x3);
      value[i*fdim]=real(temp);
      value[i*fdim+1] = imag(temp);
    }

    return 0;
}

std::complex<double> GammaCalculator::gamma1_from_cubature(double x1, double x2, double x3)
{
    double vals_min[3] = {0,0,0};
    double vals_max[3] = {40000,2*M_PI,M_PI/2};
    double result[2];
    double error[2];
    struct integration_parameter params;
    params.gammaCalculator = this;
    params.x1 = x1;
    params.x2 = x2;
    params.x3 = x3;
    double epsabs = 0;
    if(test_analytical) epsabs = 1.0e-4;
    hcubature_v(2,integrand_gamma1_no_ogata_no_limber,&params,3,vals_min,vals_max,0,epsabs,1e-4,ERROR_L1,result,error);
    // if(!test_analytical) std::cout << x1 << ", " << x2 << ", " << x3 << ": " << result[0] << " + " << result[1] << " i +/- " << error[0] << " + " << error[1] <<" i" << std::endl;

    return std::complex<double>(result[0],result[1]);
}


std::complex<double> GammaCalculator::gamma1_from_cubature_and_ogata(double x1, double x2, double x3)
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
    hcubature_v(2,integrand_2_gamma1,&params,3,vals_min,vals_max,0,epsabs,1e-5,ERROR_L1,result,error);
    // if(!test_analytical) std::cout << x1 << ", " << x2 << ", " << x3 << ": " << result[0] << " + " << result[1] << " i +/- " << error[0] << " + " << error[1] <<" i" << std::endl;

    return std::complex<double>(result[0],result[1]);
}


std::complex<double> GammaCalculator::integrand_z_phi_psi_one_x(double z, double phi, double psi, double x1, double x2, double x3)
{
    double varpsi = varpsifunc(x1,x2,x3);
    double A3 = A(psi,x1,x2,phi,varpsi);
    if(!isfinite(A3))
    {
        std::cerr << "A3 not finite! (A3,z,phi,psi,x1,x2,x3)=" << A3 << ", " << z << ", " << phi << ", " << psi << ", " << x1 << ", " << x2 << ", " << x3 << std::endl;
    }
    assert(isfinite(A3));
    std::complex<double> prefactor = sin(2*psi)*exponential(x1,x2,x3,psi,phi,varpsi)*prefactor_phi(psi,phi);
    // std::cout << integrated_bdelta_times_rcubed_J6(z, phi, psi, A3) << std::endl;
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
    if(test_analytical) epsabs = 1.0e-7;
    hcubature_v(2,integrand_2,&params,3,vals_min,vals_max,0,epsabs,1e-4,ERROR_L1,result,error);
    // if(!test_analytical) std::cout << x1 << ", " << x2 << ", " << x3 << ": " << result[0] << " + " << result[1] << " i +/- " << error[0] << " + " << error[1] <<" i" << std::endl;

    return std::complex<double>(result[0],result[1]);
}