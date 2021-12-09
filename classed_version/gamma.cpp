#include "gamma.hpp"
#include "bispectrum.hpp"
#include <cmath>
#include <gsl/gsl_sf_bessel.h>

GammaCalculator::GammaCalculator(BispectrumCalculator *Bispectrum, double prec_h, double prec_k, std::string triangle_center)
{
    initialize_bessel(prec_h, prec_k);
    Bispectrum_ = Bispectrum;
    z_max = Bispectrum_->z_max;

    if (triangle_center == "orthocenter")
        convert_orthocenter_to_centroid_bool = false;
    else if (triangle_center == "centroid")
        convert_orthocenter_to_centroid_bool = true;
    else
    {
        std::cerr << "No valid triangle center given. \n"
                  << "Possible choices: orthocenter,centroid \n"
                  << "Exiting." << std::endl;
        exit(1);
    }
}

/* Functions for actually computing the gamma 3pcf */

std::complex<double> GammaCalculator::gamma0(double x1, double x2, double x3)
{
    std::complex<double> result;

    if (USE_OGATA)
    {
        result = gamma0_from_cubature_and_ogata(x1, x2, x3);
    }
    else
    {
        result = gamma0_from_cubature(x1, x2, x3);
    }

    result *= 27. / 8. * pow(Bispectrum_->cosmo->om, 3) * pow(100. / 299792., 5); //prefactor from limber integration
    result /= 3.;                                                                 //prefactor for bispectrum definition in Eq.(4) of Schneider et al. (2005)
    result /= (2 * pow(2 * M_PI, 3));                                             //prefactor from modified Eq.(15) of Schneider et al. (2005)

    if (convert_orthocenter_to_centroid_bool)
        result = convert_orthocenter_to_centroid(result, x1, x2, x3, false);

    return result;
}

std::complex<double> GammaCalculator::gamma1(double x1, double x2, double x3)
{
    std::complex<double> result;

    if (USE_OGATA)
    {
        result = gamma1_from_cubature_and_ogata(x1, x2, x3);
    }
    else
    {
        result = gamma1_from_cubature(x1, x2, x3);
    }

    result *= 27. / 8. * pow(Bispectrum_->cosmo->om, 3) * pow(100. / 299792., 5); //prefactor from limber integration
    result /= 3.;                                                                 //prefactor for bispectrum definition in Eq.(4) of Schneider et al. (2005)
    result /= (2 * pow(2 * M_PI, 3));                                             //prefactor from modified Eq.(15) of Schneider et al. (2005)

    if (convert_orthocenter_to_centroid_bool)
        result = convert_orthocenter_to_centroid(result, x1, x2, x3, true);

    return result;
}

std::complex<double> GammaCalculator::gamma2(double x1, double x2, double x3)
{
    return gamma1(x2, x3, x1);
}

std::complex<double> GammaCalculator::gamma3(double x1, double x2, double x3)
{
    return gamma1(x3, x1, x2);
}

std::complex<double> GammaCalculator::ggg(double r1, double r2, double r3)
{
    double weights[9] = {0, 1.0e+8, 1.0e+6, 1.0e+4, 1.0e+2, 1.0e+0, 5.0e-3, 1.0e-5, 1.0e-6};
    double a_vals[9] = {3.0e+3, 1.0e+4, 3.0e+4, 1.0e+5, 3.0e+5, 1.0e+6, 3.0e+6, 1.0e+7, 3.0e+7};

    std::complex<double> temp(0, 0);
    for (int i = 0; i < 9; i++)
    {
        temp += weights[i] * ggg_single_a(r1, r2, r3, a_vals[i]);
    }
    return temp;
}

std::complex<double> GammaCalculator::gstargg(double r1, double r2, double r3)
{
    return gggstar(r2, r3, r1);
}

std::complex<double> GammaCalculator::ggstarg(double r1, double r2, double r3)
{
    return gggstar(r3, r1, r2);
}

std::complex<double> GammaCalculator::gggstar(double r1, double r2, double r3)
{
    double weights[9] = {0, 1.0e+8, 1.0e+6, 1.0e+4, 1.0e+2, 1.0e+0, 5.0e-3, 1.0e-5, 1.0e-6};
    double a_vals[9] = {3.0e+3, 1.0e+4, 3.0e+4, 1.0e+5, 3.0e+5, 1.0e+6, 3.0e+6, 1.0e+7, 3.0e+7};

    std::complex<double> temp(0, 0);
    for (int i = 0; i < 9; i++)
    {
        temp += weights[i] * gggstar_single_a(r1, r2, r3, a_vals[i]);
    }
    return temp;
}

/* Geometric functions */

double GammaCalculator::interior_angle(double an1, double an2, double opp)
{
    return acos((pow(an1, 2) + pow(an2, 2) - pow(opp, 2)) / (2.0 * an1 * an2));
}

double GammaCalculator::height_of_triangle(double x1, double x2, double x3)
{
    return 0.5 * sqrt(2 * pow(x1, 2) + 2 * pow(x2, 2) - pow(x3, 2));
}

/*********** Functions for Bessel Integration ********************/

int GammaCalculator::initialize_bessel(double prec_h_arg, double prec_k_arg)
{
    /* This function is used to compute the integral int_0^infty r^3 f(r) J_6(r)*/
    // ##############################################################
    // INITIALIZE BESSEL INTEGRALS
    // ##############################################################
    printf("Initializing bessel arrays...");
    prec_h = prec_h_arg;
    prec_k = uint(prec_k_arg / prec_h);
    std::vector<double> tmp(prec_k);
    bessel_zeros = tmp;
    pi_bessel_zeros = tmp;
    array_psi = tmp;
    array_psi_J2 = tmp;
    array_bessel = tmp;
    array_psip = tmp;
    array_w = tmp;
    array_product = tmp;
    array_product_J2 = tmp;

    for (unsigned int i = 0; i < prec_k; i++)
    {
        bessel_zeros[i] = gsl_sf_bessel_zero_Jnu(6, i);
        pi_bessel_zeros[i] = bessel_zeros[i] / M_PI;
        array_psi[i] = M_PI * psi(pi_bessel_zeros[i] * prec_h) / prec_h;
        array_bessel[i] = gsl_sf_bessel_Jn(6, array_psi[i]);
        array_psip[i] = psip(prec_h * pi_bessel_zeros[i]);
        array_w[i] = 2 / (M_PI * bessel_zeros[i] * pow(gsl_sf_bessel_Jn(7, bessel_zeros[i]), 2));
        array_product[i] = array_w[i] * pow(array_psi[i], 3) * array_bessel[i] * array_psip[i];
    }

    for (unsigned int i = 0; i < prec_k; i++)
    {
        bessel_zeros[i] = gsl_sf_bessel_zero_Jnu(2, i);
        pi_bessel_zeros[i] = bessel_zeros[i] / M_PI;
        array_psi_J2[i] = M_PI * psi(pi_bessel_zeros[i] * prec_h) / prec_h;
        array_bessel[i] = gsl_sf_bessel_Jn(2, array_psi_J2[i]);
        array_psip[i] = psip(prec_h * pi_bessel_zeros[i]);
        array_w[i] = 2 / (M_PI * bessel_zeros[i] * pow(gsl_sf_bessel_Jn(3, bessel_zeros[i]), 2));
        array_product_J2[i] = array_w[i] * pow(array_psi_J2[i], 3) * array_bessel[i] * array_psip[i];
    }
    printf("Done \n");
    return 0;
}

double GammaCalculator::psi(double t)
{
    return t * tanh(M_PI * sinh(t) / 2);
}

double GammaCalculator::psip(double t)
{
    double zahler = sinh(M_PI * sinh(t)) + M_PI * t * cosh(t);
    double nenner = cosh(M_PI * sinh(t)) + 1;
    return zahler / nenner;
}

/*************** Terms that appear in the Schneider et al. Integration ****************/

double GammaCalculator::alpha(double psi, double x1, double x2, double phi, double varpsi)
{
    double zahler = (cos(psi) * x2 - sin(psi) * x1) * sin((phi + varpsi) / 2);
    double nenner = (cos(psi) * x2 + sin(psi) * x1) * cos((phi + varpsi) / 2);
    return atan2(zahler, nenner);
}

double GammaCalculator::betabar(double psi, double phi)
{
    double zahler = cos(2 * psi) * sin(phi);
    double nenner = cos(phi) + sin(2 * psi);
    return 0.5 * atan2(zahler, nenner);
}

std::complex<double> GammaCalculator::exponential_of_betabar(double psi, double phi)
{
    return exp(std::complex<double>(0, betabar(psi, phi) * 2.));
}

std::complex<double> GammaCalculator::exponential_prefactor(double x1, double x2, double x3, double psi, double phi, double varpsi)
{
    return exp(std::complex<double>(0, (interior_angle(x2, x3, x1) - interior_angle(x1, x3, x2) - 6 * alpha(psi, x1, x2, phi, varpsi))));
}

double GammaCalculator::A(double psi, double x1, double x2, double phi, double varpsi)
{
    return sqrt(pow(cos(psi) * x2, 2) + pow(sin(psi) * x1, 2) + sin(2 * psi) * x1 * x2 * cos(phi + varpsi));
}

/************************ Integration Methods *******************************/

std::complex<double> GammaCalculator::integrand_gamma0_r_phi_psi_one_x(double r, double phi, double psi, double x1, double x2, double x3)
{
    double varpsi = interior_angle(x1, x2, x3);
    double A3 = A(psi, x1, x2, phi, varpsi);
    assert(isfinite(A3));
    std::complex<double> prefactor = exponential_prefactor(x1, x2, x3, psi, phi, varpsi) * exponential_of_betabar(psi, phi);
    return sin(2 * psi) * prefactor * gsl_sf_bessel_Jn(6, A3 * r);
}

std::complex<double> GammaCalculator::integrand_gamma0_cubature(double r, double phi, double psi, double z, double x1, double x2, double x3)
{
    double ell1 = r * cos(psi);
    double ell2 = r * sin(psi);
    double ell3 = sqrt(ell1 * ell1 + ell2 * ell2 + 2 * ell1 * ell2 * cos(phi));
    // std::cout << ell1 << ", " << ell2 << ", " << ell3 << std::endl;
    struct ell_params ells = {ell1, ell2, ell3};
    return Bispectrum_->integrand_bkappa(z, ells) * pow(r, 3) *
           (integrand_gamma0_r_phi_psi_one_x(r, phi, psi, x1, x2, x3) + integrand_gamma0_r_phi_psi_one_x(r, phi, psi, x2, x3, x1) + integrand_gamma0_r_phi_psi_one_x(r, phi, psi, x3, x1, x2));
}

int GammaCalculator::integrand_gamma0_cubature(unsigned ndim, size_t npts, const double *vars, void *fdata, unsigned fdim, double *value)
{
    struct GammaCalculatorContainer params = *((GammaCalculatorContainer *)fdata);

    GammaCalculator *gammaCalculator = params.gammaCalculator;
    double x1 = params.x1;
    double x2 = params.x2;
    double x3 = params.x3;

    // std::cout << "Batch-size for evaluation: " << npts << std::endl;

#pragma omp parallel for
    for (unsigned int i = 0; i < npts; i++)
    {
        double r = vars[i * ndim];
        double phi = vars[i * ndim + 1];
        double psi = vars[i * ndim + 2];
        double z = vars[i * ndim + 3];
        std::complex<double> temp = gammaCalculator->integrand_gamma0_cubature(r, phi, psi, z, x1, x2, x3);
        value[fdim * i] = real(temp);
        value[fdim * i + 1] = imag(temp);
    }

    return 0;
}

std::complex<double> GammaCalculator::gamma0_from_cubature(double x1, double x2, double x3)
{
    double vals_min[4] = {0, 0, 0, 0};
    double vals_max[4] = {40000, 2 * M_PI, M_PI / 2, z_max};
    double result[2];
    double error[2];
    struct GammaCalculatorContainer params;
    params.gammaCalculator = this;
    params.x1 = x1;
    params.x2 = x2;
    params.x3 = x3;
    double epsabs = 0;
    if (test_analytical)
        epsabs = 1.0e-7;
    hcubature_v(2, integrand_gamma0_cubature, &params, 4, vals_min, vals_max, 0, epsabs, 1e-4, ERROR_L1, result, error);

    if (!test_analytical)
        std::cout << x1 << ", " << x2 << ", " << x3 << ": " << result[0] << " + " << result[1] << " i +/- " << error[0] << " + " << error[1] << " i" << std::endl;

    return std::complex<double>(result[0], result[1]);
}

double GammaCalculator::integrated_bdelta_times_rcubed_J6(double z, double phi, double psi, double A3)
{ /* this computes int_0^oo dR R^3 b_delta(R cos(psi), R sin(psi), phi) J_6(A_3' R) */
    double bis = 0;
    double cpsi = cos(psi);
    double spsi = sin(psi);
    double ell1, ell2, ell3, temp;

    for (unsigned int k = 1; k < prec_k; k++)
    {
        ell1 = array_psi[k] / A3 * cpsi;
        ell2 = array_psi[k] / A3 * spsi;

        ell3 = ell1 * ell1 + ell2 * ell2 + 2 * ell1 * ell2 * cos(phi);
        if (ell3 <= 0)
            ell3 = 0;
        else
            ell3 = sqrt(ell3);
        // if(isnan(ell3)) ell3 = 0;
        struct ell_params ells = {ell1, ell2, ell3};

        temp = Bispectrum_->integrand_bkappa(z, ells) * array_product[k];
        assert(isfinite(temp));
        bis += temp;
    }

    bis = bis * M_PI / pow(A3, 4);
    return bis;
}

std::complex<double> GammaCalculator::integrand_z_phi_psi_one_x(double z, double phi, double psi, double x1, double x2, double x3)
{
    double varpsi = interior_angle(x1, x2, x3);
    double A3 = A(psi, x1, x2, phi, varpsi);
    if (!isfinite(A3))
    {
        std::cerr << "A3 not finite! (A3,z,phi,psi,x1,x2,x3)=" << A3 << ", " << z << ", " << phi << ", " << psi << ", " << x1 << ", " << x2 << ", " << x3 << std::endl;
    }
    assert(isfinite(A3));
    std::complex<double> prefactor = sin(2 * psi) * exponential_prefactor(x1, x2, x3, psi, phi, varpsi) * exponential_of_betabar(psi, phi);
    return prefactor * integrated_bdelta_times_rcubed_J6(z, phi, psi, A3);
}

std::complex<double> GammaCalculator::integrand_z_phi_psi(double z, double phi, double psi, double x1, double x2, double x3)
{
    return integrand_z_phi_psi_one_x(z, phi, psi, x1, x2, x3) + integrand_z_phi_psi_one_x(z, phi, psi, x2, x3, x1) + integrand_z_phi_psi_one_x(z, phi, psi, x3, x1, x2);
}

int GammaCalculator::integrand_gamma0_cubature_and_ogata(unsigned ndim, size_t npts, const double *vars, void *fdata, unsigned fdim, double *value)
{
    struct GammaCalculatorContainer params = *((GammaCalculatorContainer *)fdata);

    GammaCalculator *gammaCalculator = params.gammaCalculator;
    double x1 = params.x1;
    double x2 = params.x2;
    double x3 = params.x3;

#pragma omp parallel for
    for (unsigned int i = 0; i < npts; i++)
    {
        double z = vars[i * ndim];
        double phi = vars[i * ndim + 1];
        double psi = vars[i * ndim + 2];
        std::complex<double> temp = gammaCalculator->integrand_z_phi_psi(z, phi, psi, x1, x2, x3);
        value[i * fdim] = real(temp);
        value[i * fdim + 1] = imag(temp);
    }

    return 0;
}

std::complex<double> GammaCalculator::gamma0_from_cubature_and_ogata(double x1, double x2, double x3)
{
    double vals_min[3] = {0, 0, 0};
    double vals_max[3] = {z_max, 2 * M_PI, M_PI / 2};
    double result[2];
    double error[2];
    struct GammaCalculatorContainer params;
    params.gammaCalculator = this;
    params.x1 = x1;
    params.x2 = x2;
    params.x3 = x3;
    double epsabs = 0;
    if (test_analytical)
        epsabs = 1.0e-7;
    hcubature_v(2, integrand_gamma0_cubature_and_ogata, &params, 3, vals_min, vals_max, 0, epsabs, 1e-3, ERROR_L1, result, error);
    return std::complex<double>(result[0], result[1]);
}

std::complex<double> GammaCalculator::integrand_r_phi_psi_gamma1(double r, double phi, double psi, double x1, double x2, double x3)
{

    double varbetabar = betabar(psi, phi);

    double varpsi1 = interior_angle(x2, x3, x1);
    double varpsi2 = interior_angle(x3, x1, x2);
    double varpsi3 = interior_angle(x1, x2, x3);

    double A1 = A(psi, x2, x3, phi, varpsi1);
    double A2 = A(psi, x3, x1, phi, varpsi2);
    double A3 = A(psi, x1, x2, phi, varpsi3);

    double alpha1 = alpha(psi, x2, x3, phi, varpsi1);
    double alpha2 = alpha(psi, x3, x1, phi, varpsi2);
    double alpha3 = alpha(psi, x1, x2, phi, varpsi3);

    double ell1, ell2, ell3;
    ell1 = r * cos(psi);
    ell2 = r * sin(psi);
    ell3 = sqrt(ell1 * ell1 + ell2 * ell2 + 2 * ell1 * ell2 * cos(phi));

    std::complex<double> integrand_3(0, varpsi1 - varpsi2 + 2 * varpsi3 + 2 * (varbetabar - phi - alpha3));
    integrand_3 = exp(integrand_3);
    integrand_3 *= gsl_sf_bessel_Jn(2, A3 * r);

    std::complex<double> integrand_1(0, varpsi3 - varpsi2 - 2 * (varbetabar + alpha1));
    integrand_1 = exp(integrand_1);
    integrand_1 *= gsl_sf_bessel_Jn(2, A1 * r);

    std::complex<double> integrand_2(0, varpsi3 - varpsi1 - 2 * varpsi2 + 2 * (varbetabar + phi - alpha2));
    integrand_2 = exp(integrand_2);
    integrand_2 *= gsl_sf_bessel_Jn(2, A2 * r);

    double prefactor = 27. / 8. * pow(Bispectrum_->cosmo->om, 3) * pow(100. / 299792., 5) / (3 * 2 * pow(2 * M_PI, 3));

    return Bispectrum_->bkappa(ell1, ell2, ell3) * pow(r, 3) * sin(2 * psi) * (integrand_1 + integrand_2 + integrand_3) / prefactor;
}

int GammaCalculator::integrand_gamma1_no_ogata_no_limber(unsigned ndim, size_t npts, const double *vars, void *fdata, unsigned fdim, double *value)
{
    struct GammaCalculatorContainer params = *((GammaCalculatorContainer *)fdata);

    GammaCalculator *gammaCalculator = params.gammaCalculator;
    double x1 = params.x1;
    double x2 = params.x2;
    double x3 = params.x3;

    std::cout << "Evaluating " << npts << " points!" << std::endl;

#pragma omp parallel for
    for (unsigned int i = 0; i < npts; i++)
    {
        double r = vars[i * ndim];
        double phi = vars[i * ndim + 1];
        double psi = vars[i * ndim + 2];
        std::complex<double> temp = gammaCalculator->integrand_r_phi_psi_gamma1(r, phi, psi, x1, x2, x3);
        value[i * fdim] = real(temp);
        value[i * fdim + 1] = imag(temp);
    }

    return 0;
}

std::complex<double> GammaCalculator::gamma1_from_cubature(double x1, double x2, double x3)
{
    double vals_min[3] = {0, 0, 0};
    double vals_max[3] = {40000, 2 * M_PI, M_PI / 2};
    double result[2];
    double error[2];
    struct GammaCalculatorContainer params;
    params.gammaCalculator = this;
    params.x1 = x1;
    params.x2 = x2;
    params.x3 = x3;
    double epsabs = 0;
    if (test_analytical)
        epsabs = 1.0e-4;
    hcubature_v(2, integrand_gamma1_no_ogata_no_limber, &params, 3, vals_min, vals_max, 0, epsabs, 1e-4, ERROR_L1, result, error);

    return std::complex<double>(result[0], result[1]);
}

double GammaCalculator::integrated_bdelta_times_rcubed_J2(double z, double phi, double psi, double A3)
{ /* this computes int_0^oo dR R^3 b_delta(R cos(psi), R sin(psi), phi) J_2(A_3' R) */
    double bis = 0;
    double cpsi = cos(psi);
    double spsi = sin(psi);
    double ell1, ell2, ell3, temp;

    for (unsigned int k = 1; k < prec_k; k++)
    {
        ell1 = array_psi_J2[k] / A3 * cpsi;
        ell2 = array_psi_J2[k] / A3 * spsi;
        ell3 = ell1 * ell1 + ell2 * ell2 + 2 * ell1 * ell2 * cos(phi);
        if (ell3 <= 0)
            ell3 = 0;
        else
            ell3 = sqrt(ell3);

        struct ell_params ells = {ell1, ell2, ell3};

        temp = Bispectrum_->integrand_bkappa(z, ells) * array_product_J2[k];
        assert(isfinite(temp));
        bis += temp;
    }

    bis = bis * M_PI / pow(A3, 4);
    return bis;
}

std::complex<double> GammaCalculator::integrand_z_phi_psi_gamma1(double z, double phi, double psi, double x1, double x2, double x3)
{

    double varbetabar = betabar(psi, phi);

    double varpsi1 = interior_angle(x2, x3, x1);
    double varpsi2 = interior_angle(x3, x1, x2);
    double varpsi3 = interior_angle(x1, x2, x3);

    double A1 = A(psi, x2, x3, phi, varpsi1);
    double A2 = A(psi, x3, x1, phi, varpsi2);
    double A3 = A(psi, x1, x2, phi, varpsi3);

    double alpha1 = alpha(psi, x2, x3, phi, varpsi1);
    double alpha2 = alpha(psi, x3, x1, phi, varpsi2);
    double alpha3 = alpha(psi, x1, x2, phi, varpsi3);

    std::complex<double> integrand_3(0, varpsi1 - varpsi2 + 2 * varpsi3 + 2 * (varbetabar - phi - alpha3));
    integrand_3 = exp(integrand_3);
    integrand_3 *= integrated_bdelta_times_rcubed_J2(z, phi, psi, A3);

    std::complex<double> integrand_1(0, varpsi3 - varpsi2 - 2 * (varbetabar + alpha1));
    integrand_1 = exp(integrand_1);
    integrand_1 *= integrated_bdelta_times_rcubed_J2(z, phi, psi, A1);

    std::complex<double> integrand_2(0, varpsi3 - varpsi1 - 2 * varpsi2 + 2 * (varbetabar + phi - alpha2));
    integrand_2 = exp(integrand_2);
    integrand_2 *= integrated_bdelta_times_rcubed_J2(z, phi, psi, A2);

    return sin(2 * psi) * (integrand_1 + integrand_2 + integrand_3);
}

int GammaCalculator::integrand_gamma1_cubature_and_ogata(unsigned ndim, size_t npts, const double *vars, void *fdata, unsigned fdim, double *value)
{
    struct GammaCalculatorContainer params = *((GammaCalculatorContainer *)fdata);

    GammaCalculator *gammaCalculator = params.gammaCalculator;
    double x1 = params.x1;
    double x2 = params.x2;
    double x3 = params.x3;

#pragma omp parallel for
    for (unsigned int i = 0; i < npts; i++)
    {
        double z = vars[i * ndim];
        double phi = vars[i * ndim + 1];
        double psi = vars[i * ndim + 2];
        std::complex<double> temp = gammaCalculator->integrand_z_phi_psi_gamma1(z, phi, psi, x1, x2, x3);
        value[i * fdim] = real(temp);
        value[i * fdim + 1] = imag(temp);
    }

    return 0;
}

std::complex<double> GammaCalculator::gamma1_from_cubature_and_ogata(double x1, double x2, double x3)
{
    double vals_min[3] = {0, 0, 0};
    double vals_max[3] = {z_max, 2 * M_PI, M_PI / 2};
    double result[2];
    double error[2];
    struct GammaCalculatorContainer params;
    params.gammaCalculator = this;
    params.x1 = x1;
    params.x2 = x2;
    params.x3 = x3;
    double epsabs = 0;
    if (test_analytical)
        epsabs = 1.0e-8;
    hcubature_v(2, integrand_gamma1_cubature_and_ogata, &params, 3, vals_min, vals_max, 0, epsabs, 1e-3, ERROR_L1, result, error);
    // if(!test_analytical) std::cout << x1 << ", " << x2 << ", " << x3 << ": " << result[0] << " + " << result[1] << " i +/- " << error[0] << " + " << error[1] <<" i" << std::endl;

    return std::complex<double>(result[0], result[1]);
}

/* Analytic test functions */

std::complex<double> GammaCalculator::ggg_single_a(double r1, double r2, double r3, double a)
{
    // FOR TESTING COMPARISON
    std::complex<double> x(r2, 0);
    double height = (r1 + r2 + r3) * (r2 + r3 - r1) * (r1 - r2 + r3) * (r1 + r2 - r3);
    height = sqrt(height) / (2. * r2);
    double rest_of_r1 = sqrt(r1 * r1 - height * height);
    std::complex<double> y(-rest_of_r1, -height);

    std::complex<double> z = -x - y;
    const std::complex<double> i(0, 1);
    double phi1 = atan2(imag(x), real(x));
    double phi2 = atan2(imag(y), real(y));
    double phi3 = atan2(imag(z), real(z));
    std::complex<double> pref = pow((x + 2. * y) * (x - y) * (y + 2. * x), 2);
    std::complex<double> expon = pow(abs(x), 2) + pow(abs(y), 2) + real(x) * real(y) + imag(x) * imag(y);
    std::complex<double> phase = exp(-2. * i * (phi1 + phi2 + phi3));
    // std::complex<double> temp_result = pref*exp(-2*a*expon)*phase;
    return pref * exp(-2 * a * expon) * phase;
}

std::complex<double> GammaCalculator::gggstar_single_a(double r1, double r2, double r3, long double a)
{
    std::complex<long double> x(r2, 0);
    long double height = (r1 + r2 + r3) * (r2 + r3 - r1) * (r1 - r2 + r3) * (r1 + r2 - r3);
    height = sqrt(height) / (2. * r2);
    long double rest_of_r1 = sqrt(r1 * r1 - height * height);
    std::complex<long double> y(-rest_of_r1, -height);

    long double X1 = real(x);
    long double X2 = imag(x);
    long double Y1 = real(y);
    long double Y2 = imag(y);

    std::complex<long double> z = -y - x;
    long double phi1 = atan2(imag(x), real(x));
    long double phi2 = atan2(imag(y), real(y));
    long double phi3 = atan2(imag(z), real(z));

    const std::complex<long double> i(0, 1);

    std::complex<long double> phase = exp(static_cast<long double>(-2.) * i * (phi1 + phi2 - phi3));

    std::complex<long double> gggstar = (static_cast<long double>(4.) * pow(a, 2) * pow(X1, 6) - static_cast<long double>(4.) * pow(a, 2) * pow(X2, 6) + static_cast<long double>(4) * pow(a, 2) * pow(X2, 5) * (std::complex<long double>(0, 7.) * Y1 - static_cast<long double>(3.) * Y2) + static_cast<long double>(4) * pow(a, 2) * pow(X1, 5) * (std::complex<long double>(0, 2) * X2 + static_cast<long double>(3) * Y1 + std::complex<long double>(0, 7) * Y2) +
                                         a * pow(X1, 4) * (static_cast<long double>(8.) + a * (static_cast<long double>(4.) * pow(X2, 2) - std::complex<long double>(0, 4) * X2 * Y1 - static_cast<long double>(3.) * pow(Y1, 2) - static_cast<long double>(44.) * X2 * Y2 + std::complex<long double>(0, 58.) * Y1 * Y2 - static_cast<long double>(77.) * pow(Y2, 2))) +
                                         std::complex<long double>(0, 4.) * X2 * (Y1 + std::complex<long double>(0, 1) * Y2) * (static_cast<long double>(-6) + pow(a, 2) * (Y1 - std::complex<long double>(0, 1) * Y2) * pow(Y1 + std::complex<long double>(0, 1) * Y2, 2) * (static_cast<long double>(7.) * Y1 - std::complex<long double>(0, 3) * Y2) + a * (static_cast<long double>(5.) * pow(Y1, 2) + std::complex<long double>(0, 4) * Y1 * Y2 + pow(Y2, 2))) +
                                         a * pow(X2, 4) * (static_cast<long double>(-8.) + a * (77. * pow(Y1, 2) + std::complex<long double>(0, 58.) * Y1 * Y2 + static_cast<long double>(3.) * pow(Y2, 2))) + static_cast<long double>(2.) * a * pow(X2, 3) * (std::complex<long double>(0, -53.) * a * pow(Y1, 3) - static_cast<long double>(2.) * Y2 + static_cast<long double>(53.) * a * pow(Y1, 2) * Y2 + static_cast<long double>(13.) * a * pow(Y2, 3) + std::complex<long double>(0, 1) * Y1 * (static_cast<long double>(10.) - static_cast<long double>(13.) * a * pow(Y2, 2))) +
                                         static_cast<long double>(2.) * pow(Y1 + std::complex<long double>(0, 1) * Y2, 2) * (static_cast<long double>(-3) + static_cast<long double>(2) * a * (pow(Y1, 2) + pow(Y2, 2)) * (static_cast<long double>(2.) + a * (pow(Y1, 2) + pow(Y2, 2)))) - pow(X2, 2) * (static_cast<long double>(-6.) + a * (Y1 + std::complex<long double>(0, 1) * Y2) * (std::complex<long double>(0, 24) * Y2 + a * (Y1 + std::complex<long double>(0, 1) * Y2) * (static_cast<long double>(77.) * pow(Y1, 2) - std::complex<long double>(0, 58) * Y1 * Y2 + static_cast<long double>(3.) * pow(Y2, 2)))) +
                                         std::complex<long double>(0, 2) * a * pow(X1, 3) * (static_cast<long double>(8.) * a * pow(X2, 3) - std::complex<long double>(0, 2) * Y1 + static_cast<long double>(10.) * Y2 + static_cast<long double>(4.) * a * pow(X2, 2) * (std::complex<long double>(0, -7) * Y1 + static_cast<long double>(3.) * Y2) + std::complex<long double>(0, 1) * a * (Y1 + std::complex<long double>(0, 1) * Y2) * (static_cast<long double>(13.) * pow(Y1, 2) + static_cast<long double>(53.) * pow(Y2, 2)) - static_cast<long double>(8.) * X2 * (static_cast<long double>(-1.) + a * (static_cast<long double>(4.) * pow(Y1, 2) + std::complex<long double>(0, 2) * Y1 * Y2 + static_cast<long double>(6.) * pow(Y2, 2)))) + pow(X1, 2) * (static_cast<long double>(-6.) - std::complex<long double>(0, 12) * a * (X2 * (Y1 - std::complex<long double>(0, 3) * Y2) + static_cast<long double>(2.) * Y1 * (std::complex<long double>(0, -1) * Y1 + Y2)) + pow(a, 2) * (-4. * pow(X2, 4) + 8. * pow(X2, 3) * (std::complex<long double>(0, 3) * Y1 - 7. * Y2) + 6. * pow(X2, 2) * (7. * pow(Y1, 2) + std::complex<long double>(0, 30) * Y1 * Y2 - 7. * pow(Y2, 2)) + 2. * X2 * (std::complex<long double>(0, -1) * Y1 + Y2) * (13. * pow(Y1, 2) + std::complex<long double>(0, 80) * Y1 * Y2 + 53. * pow(Y2, 2)) - pow(Y1 + std::complex<long double>(0, 1) * Y2, 2) * (3. * pow(Y1, 2) + std::complex<long double>(0, 58) * Y1 * Y2 + 77 * pow(Y2, 2)))) +
                                         2. * X1 * (std::complex<long double>(0, 4) * pow(a, 2) * pow(X2, 5) + 2. * pow(a, 2) * pow(X2, 4) * (11. * Y1 - std::complex<long double>(0, 1) * Y2) + static_cast<long double>(2.) * (static_cast<long double>(-6.) + a * (Y1 + std::complex<long double>(0, 1) * Y2) * (Y1 - std::complex<long double>(0, 5) * Y2) + pow(a, 2) * (Y1 - std::complex<long double>(0, 1) * Y2) * pow(Y1 + std::complex<long double>(0, 1) * Y2, 2) * (3. * Y1 - std::complex<long double>(0, 7) * Y2)) * (Y1 + std::complex<long double>(0, 1) * Y2) + std::complex<long double>(0, 1) * X2 * (static_cast<long double>(-6.) + pow(a, 2) * pow(Y1 + std::complex<long double>(0, 1) * Y2, 2) * (29. * pow(Y1, 2) - std::complex<long double>(0, 74) * Y1 * Y2 - 29. * pow(Y2, 2)) - 12. * a * (pow(Y1, 2) + pow(Y2, 2))) - std::complex<long double>(0, 8) * a * pow(X2, 3) * (static_cast<long double>(-1.) + a * (6 * pow(Y1, 2) - std::complex<long double>(0, 2) * Y1 * Y2 + 4. * pow(Y2, 2))) - a * pow(X2, 2) * (static_cast<long double>(53) * a * pow(Y1, 3) - std::complex<long double>(0, 27) * a * pow(Y1, 2) * Y2 + std::complex<long double>(0, 1) * Y2 * (static_cast<long double>(6) + static_cast<long double>(13) * a * pow(Y2, 2)) + 3. * Y1 * (-6. + 31 * a * pow(Y2, 2))))) /
                                        (pow(a, 2) * exp(2 * a * (pow(X1, 2) + pow(X2, 2) + X1 * Y1 + pow(Y1, 2) + X2 * Y2 + pow(Y2, 2))));

    return std::complex<double>(real(phase * gggstar), imag(phase * gggstar));
}

/* Rotating from orthocenter to centroid */

double GammaCalculator::one_rotation_angle_otc(double x1, double x2, double x3)
{
    double psi3 = interior_angle(x1, x2, x3);
    double h3 = height_of_triangle(x1, x2, x3);
    double cos_2_angle = pow(pow(x2, 2) - pow(x1, 2), 2) - 4 * pow(x1 * x2 * sin(psi3), 2);
    cos_2_angle /= pow(2 * h3 * x3, 2);
    double sin_2_angle = (pow(x2, 2) - pow(x1, 2)) * x1 * x2 * sin(psi3);
    sin_2_angle /= pow(h3 * x3, 2);
    return atan2(sin_2_angle, cos_2_angle) / 2;
}

std::complex<double> GammaCalculator::convert_orthocenter_to_centroid(std::complex<double> &gamma, double x1, double x2, double x3, bool conjugate_phi1)
{
    double angle1 = one_rotation_angle_otc(x2, x3, x1);
    double angle2 = one_rotation_angle_otc(x3, x1, x2);
    double angle3 = one_rotation_angle_otc(x1, x2, x3);
    if (conjugate_phi1)
        angle1 *= -1.;
    return gamma * exp(std::complex<double>(0, -2.) * (angle1 + angle2 + angle3));
}
