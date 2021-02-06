#include "Levin.h"

const double Levin::min_interval = 1.e-2;
const double Levin::tol_abs = 0.0;
const double Levin::min_sv = 1.0e-10;

Levin::Levin(uint type1, uint col1, uint nsub1, double relative_tol1)
{
    type = type1;
    col = col1;
    nsub = nsub1;
    relative_tol = relative_tol1;
    setup(type);
}

Levin::Levin()
{
}

void Levin::initialize(uint type1, uint col1, uint nsub1, double relative_tol1)
{
    type = type1;
    col = col1;
    nsub = nsub1;
    relative_tol = relative_tol1;
    setup(type); 
}

Levin::~Levin()
{
}

void Levin::setup(uint type)
{
    if (type < 2)
    {
        d = 2;
    }
    else
    {
        d = 4;
    }
}

double Levin::w_single(double x, double k, uint ell, uint i)
{
    gsl_sf_result r;
    int status;
    if (type == 0)
    {
        switch (i)
        {
        case 0:
            status = gsl_sf_bessel_Jn_e(ell, x * k, &r);
            break;
        case 1:
            status = gsl_sf_bessel_Jn_e(ell - 1, x * k, &r);
            break;
        default:
            return 0.0;
        }
        if (status != GSL_SUCCESS)
        {
            std::cerr << "Failed to compute cylindrical Bessel function for ell=" << ell << std::endl;
        }
    }
    if (type == 1)
    {
        switch (i)
        {
        case 0:
            status = gsl_sf_bessel_jl_e(ell, x * k, &r);
            break;
        case 1:
            status = gsl_sf_bessel_jl_e(ell - 1, x * k, &r);
            break;
        default:
            return 0.0;
        }
        if (status != GSL_SUCCESS)
        {
            std::cerr << "Failed to compute spherical Bessel function for ell=" << ell << std::endl;
        }
    }
    return r.val;
}

double Levin::w_double(double x, double k, uint ell_1, uint ell_2, uint i)
{
    double result = 0.0;
    gsl_sf_result r;
    int status;
    if (type == 2)
    {
        switch (i)
        {
        case 0:
            status = gsl_sf_bessel_Jn_e(ell_1, x * k, &r);
            result = r.val;
            status = gsl_sf_bessel_Jn_e(ell_2, x * k, &r);
            result *= r.val;
            break;
        case 1:
            status = gsl_sf_bessel_Jn_e(ell_1 + 1, x * k, &r);
            result = r.val;
            status = gsl_sf_bessel_Jn_e(ell_2, x * k, &r);
            result *= r.val;
            break;
        case 2:
            status = gsl_sf_bessel_Jn_e(ell_1, x * k, &r);
            result = r.val;
            status = gsl_sf_bessel_Jn_e(ell_2 + 1, x * k, &r);
            result *= r.val;
            break;
        case 3:
            status = gsl_sf_bessel_Jn_e(ell_1 + 1, x * k, &r);
            result = r.val;
            status = gsl_sf_bessel_Jn_e(ell_2 + 1, x * k, &r);
            result *= r.val;
            break;
        default:
            return 0.0;
        }
        if (status != GSL_SUCCESS)
        {
            std::cerr << "Failed to compute cylindrical Bessel function for ell=" << ell_2 << std::endl;
        }
    }
    if (type == 3)
    {
        switch (i)
        {
        case 0:
            status = gsl_sf_bessel_jl_e(ell_1, x * k, &r);
            result = r.val;
            status = gsl_sf_bessel_jl_e(ell_2, x * k, &r);
            result *= r.val;
            break;
        case 1:
            status = gsl_sf_bessel_jl_e(ell_1 - 1, x * k, &r);
            result = r.val;
            status = gsl_sf_bessel_jl_e(ell_2, x * k, &r);
            result *= r.val;
            break;
        case 2:
            status = gsl_sf_bessel_jl_e(ell_1, x * k, &r);
            result = r.val;
            status = gsl_sf_bessel_jl_e(ell_2 - 1, x * k, &r);
            result *= r.val;
            break;
        case 3:
            status = gsl_sf_bessel_jl_e(ell_1 - 1, x * k, &r);
            result = r.val;
            status = gsl_sf_bessel_jl_e(ell_2 - 1, x * k, &r);
            result *= r.val;
            break;
        default:
            return 0.0;
        }
        if (status != GSL_SUCCESS)
        {
            std::cerr << "Failed to compute spherical Bessel function for ell=" << ell_2 << std::endl;
        }
    }
    return result;
}

double Levin::A_matrix_single(uint i, uint j, double x, double k, uint ell)
{
    switch (type)
    {
    case 0:
        if (i == 0 && j == 0)
        {
            return -static_cast<double>(ell) / x;
        }
        if (i * j == 1)
        {
            return (ell - 1.0) / x;
        }
        if (i < j)
        {
            return k;
        }
        else
        {
            return -k;
        }
    case 1:
        if (i == 0 && j == 0)
        {
            return -(ell + 1.0) / x;
        }
        if (i * j == 1)
        {
            return (ell - 1.0) / x;
        }
        if (i < j)
        {
            return k;
        }
        else
        {
            return -k;
        }
    default:
        return 0.0;
    }
}

double Levin::A_matrix_double(uint i, uint j, double x, double k, uint ell_1, uint ell_2)
{
    switch (type)
    {
    case 2:
        if (i + j == 3)
        {
            return 0.0;
        }
        if (i == 0 && j == 0)
        {
            return static_cast<double>(ell_1 + ell_2) / x;
        }
        if (i == 1 && j == 1)
        {
            return (static_cast<double>(ell_2) - static_cast<double>(ell_1) - 1.0) / x;
        }
        if (i == 2 && j == 2)
        {
            return -(static_cast<double>(ell_1) - static_cast<double>(ell_2) - 1.0) / x;
        }
        if (i == 3 && j == 3)
        {
            return -(ell_1 + ell_2 + 2.0) / x;
        }
        if (i == 0 || j == 3)
        {
            return -k;
        }
        if (j == 0 || i == 3)
        {
            return k;
        }
    case 3:
        if (i + j == 3)
        {
            return 0.0;
        }
        if (i == 0 && j == 0)
        {
            return -(ell_1 + ell_2 + 2.0) / x;
        }
        if (i == 1 && j == 1)
        {
            return (static_cast<double>(ell_1) - static_cast<double>(ell_2) - 2.0) / x;
        }
        if (i == 2 && j == 2)
        {
            return (static_cast<double>(ell_2) - static_cast<double>(ell_1) - 2.0) / x;
        }
        if (i == 3 && j == 3)
        {
            return (ell_1 + ell_2 - 2.0) / x;
        }
        if (i == 0 || j == 3)
        {
            return k;
        }
        if (j == 0 || i == 3)
        {
            return -k;
        }
    default:
        return 0.0;
    }
}

std::vector<double> Levin::setNodes(double A, double B, uint col)
{
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    for (uint j = 0; j < n; j++)
    {
        x_j[j] = A + j * (B - A) / (n - 1);
    }
    return x_j;
}

double Levin::basis_function(double A, double B, double x, uint m)
{
    if (m == 0)
    {
        return 1.0;
    }
    return pow((x - (A + B) / 2) / (B - A), m);
}

double Levin::basis_function_prime(double A, double B, double x, uint m)
{
    if (m == 0)
    {
        return 0.0;
    }
    if (m == 1)
    {
        return 1.0 / (B - A);
    }
    return m / (B - A) * pow((x - (A + B) / 2.) / (B - A), (m - 1));
}

std::vector<double> Levin::solve_LSE_single(double (*function)(double, double), double A, double B, uint col, std::vector<double> x_j, double k, uint ell)
{
    uint n = (col + 1) / 2;
    n *= 2;
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *c = gsl_vector_alloc(d * n);
    if (type >= 2)
    {
        std::cerr << "Please check the type you want to integrate in the constructor (<2 required for this function)" << std::endl;
    }
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            gsl_vector_set(F_stacked, j, (*function)(x_j[j], k));
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_set_zero(matrix_G);
    for (uint i = 0; i < d; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            for (uint q = 0; q < d; q++)
            {
                for (uint m = 0; m < n; m++)
                {
                    double LSE_coeff = A_matrix_single(q, i, x_j[j], k, ell) * basis_function(A, B, x_j[j], m);
                    if (q == i)
                    {
                        LSE_coeff += basis_function_prime(A, B, x_j[j], m);
                    }
                    gsl_matrix_set(matrix_G, i * n + j, q * n + m, LSE_coeff);
                }
            }
        }
    }
    gsl_matrix *U = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(U, matrix_G);
    int s;
    gsl_permutation *P = gsl_permutation_alloc(d * n);
    gsl_linalg_LU_decomp(matrix_G, P, &s);
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    int lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
    if (lu) // in case solution via LU decomposition fails, proceed with SVD
    {
        gsl_matrix *V = gsl_matrix_alloc(d * n, d * n);
        gsl_vector *S = gsl_vector_alloc(d * n);
        gsl_vector *aux = gsl_vector_alloc(d * n);
        gsl_linalg_SV_decomp(U, V, S, aux);
        int i = d * n - 1;
        while (i > 0 && gsl_vector_get(S, i) < min_sv * gsl_vector_get(S, 0))
        {
            gsl_vector_set(S, i, 0.);
            --i;
        }
        gsl_linalg_SV_solve(U, V, S, F_stacked, c);
        gsl_matrix_free(V);
        gsl_vector_free(S);
        gsl_vector_free(aux);
    }
    std::vector<double> result(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        result[j] = gsl_vector_get(c, j);
    }
    gsl_matrix_free(U);
    gsl_vector_free(F_stacked);
    gsl_vector_free(c);
    gsl_permutation_free(P);
    gsl_set_error_handler(old_handler);
    gsl_matrix_free(matrix_G);
    return result;
}

std::vector<double> Levin::solve_LSE_double(double (*function)(double, double), double A, double B, uint col, std::vector<double> x_j, double k, uint ell_1, uint ell_2)
{
    uint n = (col + 1) / 2;
    n *= 2;
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *c = gsl_vector_alloc(d * n);
    if (type <= 1)
    {
        std::cerr << "Please check the type you want to integrate in the constructor (>1 required for this function)" << std::endl;
    }
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            gsl_vector_set(F_stacked, j, (*function)(x_j[j], k));
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_set_zero(matrix_G);
    for (uint i = 0; i < d; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            for (uint q = 0; q < d; q++)
            {
                for (uint m = 0; m < n; m++)
                {
                    double LSE_coeff = A_matrix_double(q, i, x_j[j], k, ell_1, ell_2) * basis_function(A, B, x_j[j], m);
                    if (q == i)
                    {
                        LSE_coeff += basis_function_prime(A, B, x_j[j], m);
                    }
                    gsl_matrix_set(matrix_G, i * n + j, q * n + m, LSE_coeff);
                }
            }
        }
    }
    gsl_matrix *U = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(U, matrix_G);
    int s;
    gsl_permutation *P = gsl_permutation_alloc(d * n);
    gsl_linalg_LU_decomp(matrix_G, P, &s);
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    int lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
    if (lu) // in case solution via LU decomposition fails, proceed with SVD
    {
        gsl_matrix *V = gsl_matrix_alloc(d * n, d * n);
        gsl_vector *S = gsl_vector_alloc(d * n);
        gsl_vector *aux = gsl_vector_alloc(d * n);
        gsl_linalg_SV_decomp(U, V, S, aux);
        int i = d * n - 1;
        while (i > 0 && gsl_vector_get(S, i) < min_sv * gsl_vector_get(S, 0))
        {
            gsl_vector_set(S, i, 0.);
            --i;
        }
        gsl_linalg_SV_solve(U, V, S, F_stacked, c);
        gsl_matrix_free(V);
        gsl_vector_free(S);
        gsl_vector_free(aux);
    }
    std::vector<double> result(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        result[j] = gsl_vector_get(c, j);
    }
    gsl_matrix_free(U);
    gsl_vector_free(F_stacked);
    gsl_vector_free(c);
    gsl_permutation_free(P);
    gsl_set_error_handler(old_handler);
    gsl_matrix_free(matrix_G);
    return result;
}

double Levin::p(double A, double B, uint i, double x, uint col, std::vector<double> c)
{
    uint n = (col + 1) / 2;
    n *= 2;
    double result = 0.0;
    for (uint m = 0; m < n; m++)
    {
        result += c[i * n + m] * basis_function(A, B, x, m);
    }
    return result;
}

double Levin::integrate_single(double (*function)(double, double), double A, double B, uint col, double k, uint ell)
{
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    std::vector<double> c(n);
    x_j = setNodes(A, B, col);
    c = solve_LSE_single(function, A, B, col, x_j, k, ell);
    for (uint i = 0; i < d; i++)
    {
        result += p(A, B, i, B, col, c) * w_single(B, k, ell, i) - p(A, B, i, A, col, c) * w_single(A, k, ell, i);
    }
    return result;
}

double Levin::integrate_double(double (*function)(double, double), double A, double B, uint col, double k, uint ell_1, uint ell_2)
{
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    std::vector<double> c(n);
    x_j = setNodes(A, B, col);
    c = solve_LSE_double(function, A, B, col, x_j, k, ell_1, ell_2);
    for (uint i = 0; i < d; i++)
    {
        result += p(A, B, i, B, col, c) * w_double(B, k, ell_1, ell_2, i) - p(A, B, i, A, col, c) * w_double(A, k, ell_1, ell_2, i);
    }
    return result;
}

double Levin::iterate_single(double (*function)(double, double), double A, double B, uint col, double k, uint ell, uint smax, bool verbose)
{
    std::vector<double> intermediate_results;
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    double I_half = integrate_single(function, A, B, col / 2, k, ell);
    double I_full = integrate_single(function, A, B, col, k, ell);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, fabs(I_full - I_half));
    double result = I_full;
    while (sub <= smax + 1)
    {
        result = 0.0;
        for (uint i = 0; i < approximations.size(); i++)
        {
            result += approximations.at(i);
        }
        /*  if (verbose)
        {
            std::cerr << "estimate: " << std::scientific << result << std::endl
                      << sub << " subintervals: " << std::endl;
            for (uint i = 0; i < approximations.size(); ++i)
            {
                std::cerr << "[" << x_sub[i] << "," << x_sub[i + 1] << "]: " << approximations[i] << " (" << error_estimates[i] << ")" << std::endl;
                std::cerr << std::endl;
            }
        }*/
        intermediate_results.push_back(result);
        if (abs(result - previous) <= GSL_MAX(relative_tol * abs(result), tol_abs))
        {
            if (verbose)
            {
                std::cerr << "converged!" << std::endl;
            }
            return result;
        }
        previous = result;
        sub++;
        uint i = 1;
        while (true)
        {
            i = findMax(error_estimates) + 1;
            if (error_estimates[i - 1] < 0)
            {
                if (verbose)
                {
                    std::cerr << "subintervals too narrow for further bisection!" << std::endl;
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates.at(i - 1) = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub.at(i - 1) + x_sub.at(i)) / 2.);
        I_half = integrate_single(function, x_sub.at(i - 1), x_sub.at(i), col / 2, k, ell);
        I_full = integrate_single(function, x_sub.at(i - 1), x_sub.at(i), col, k, ell);
        approximations.at(i - 1) = I_full;
        error_estimates.at(i - 1) = fabs(I_full - I_half);
        I_half = integrate_single(function, x_sub.at(i), x_sub.at(i + 1), col / 2, k, ell);
        I_full = integrate_single(function, x_sub.at(i), x_sub.at(i + 1), col, k, ell);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, fabs(I_full - I_half));
    }
    if (verbose)
    {
        std::cerr << "maximum number of subintervals reached!" << std::endl;
    }
    return result;
}

double Levin::iterate_double(double (*function)(double, double), double A, double B, uint col, double k, uint ell_1, uint ell_2, uint smax, bool verbose)
{
    std::vector<double> intermediate_results;
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    double I_half = integrate_double(function, A, B, col / 2, k, ell_1, ell_2);
    double I_full = integrate_double(function, A, B, col, k, ell_1, ell_2);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, fabs(I_full - I_half));
    double result = I_full;
    while (sub <= smax + 1)
    {
        result = 0.0;
        for (uint i = 0; i < approximations.size(); i++)
        {
            result += approximations.at(i);
        }
        /*if (verbose)
        {
            std::cerr << "estimate: " << std::scientific << result << std::endl
                      << sub << " subintervals: " << std::endl;
            for (uint i = 0; i < approximations.size(); ++i)
            {
                std::cerr << "[" << x_sub[i] << "," << x_sub[i + 1] << "]: " << approximations[i] << " (" << error_estimates[i] << ")" << std::endl;
                std::cerr << std::endl;
            }
        }*/
        intermediate_results.push_back(result);
        if (abs(result - previous) <= GSL_MAX(relative_tol * abs(result), tol_abs))
        {
            if (verbose)
            {
                std::cerr << "converged!" << std::endl;
            }
            return result;
        }
        previous = result;
        sub++;
        uint i = 1;
        while (true)
        {
            i = findMax(error_estimates) + 1;
            if (error_estimates[i - 1] < 0)
            {
                if (verbose)
                {
                    std::cerr << "subintervals too narrow for further bisection!" << std::endl;
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates.at(i - 1) = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub.at(i - 1) + x_sub.at(i)) / 2.);
        I_half = integrate_double(function, x_sub.at(i - 1), x_sub.at(i), col / 2, k, ell_1, ell_2);
        I_full = integrate_double(function, x_sub.at(i - 1), x_sub.at(i), col, k, ell_1, ell_2);
        approximations.at(i - 1) = I_full;
        error_estimates.at(i - 1) = fabs(I_full - I_half);
        I_half = integrate_double(function, x_sub.at(i), x_sub.at(i + 1), col / 2, k, ell_1, ell_2);
        I_full = integrate_double(function, x_sub.at(i), x_sub.at(i + 1), col, k, ell_1, ell_2);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, fabs(I_full - I_half));
    }
    if (verbose)
    {
        std::cerr << "maximum number of subintervals reached!" << std::endl;
    }
    return result;
}

uint Levin::findMax(const std::vector<double> vec)
{
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

double Levin::levin_integrate_bessel_single(double (*function)(double, double), double k, uint ell, double a, double b)
{
    return iterate_single(function, a, b, col, k, ell, nsub, true);
}

double Levin::levin_integrate_bessel_double(double (*function)(double, double), double k, uint ell_1, uint ell_2, double a, double b)
{
    return iterate_double(function, a, b, col, k, ell_1, ell_2, nsub, true);
}