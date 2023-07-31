#include "bispectrum.cuh"
#include "cuda_helpers.cuh"
#include "cubature.h"

#include <iostream>
#include <vector>
#include <math.h>

// Values that are put in constant memory on device
// These things stay the same for a kernel run
// Warning: They are read-only for the GPU!

bool constant_powerspectrum;
__constant__ bool dev_constant_powerspectrum;

// Cosmological Parameters
__constant__ double dev_h, dev_sigma8, dev_omb, dev_omc, dev_ns, dev_w, dev_om, dev_ow, dev_norm, dev_A_IA;
cosmology cosmo;
double norm_P;

__constant__ double dev_eps = 1.0e-4; //< Integration accuracy
const double eps = 1.0e-4;

__constant__ double dev_f_K_array[n_redshift_bins]; // Array for comoving distance
double f_K_array[n_redshift_bins];


__constant__ bool dev_Pk_given;
__constant__ double dev_Pk[n_kbins];
double Pk[n_kbins];
bool Pk_given;

__constant__ double dev_D1_array[n_redshift_bins];      // Array for growth factor
__constant__ double dev_r_sigma_array[n_redshift_bins]; // Array for r(sigma)
__constant__ double dev_n_eff_array[n_redshift_bins];   // Array for n_eff
__constant__ double dev_ncur_array[n_redshift_bins];    // Array for C in Halofit
double D1_array[n_redshift_bins];
double r_sigma_array[n_redshift_bins];
double n_eff_array[n_redshift_bins];
double ncur_array[n_redshift_bins];

double A96[48] = {/* abscissas for 96-point Gauss quadrature */
                  0.016276744849603, 0.048812985136050, 0.081297495464426, 0.113695850110666,
                  0.145973714654897, 0.178096882367619, 0.210031310460567, 0.241743156163840,
                  0.273198812591049, 0.304364944354496, 0.335208522892625, 0.365696861472314,
                  0.395797649828909, 0.425478988407301, 0.454709422167743, 0.483457973920596,
                  0.511694177154668, 0.539388108324358, 0.566510418561397, 0.593032364777572,
                  0.618925840125469, 0.644163403784967, 0.668718310043916, 0.692564536642172,
                  0.715676812348968, 0.738030643744400, 0.759602341176648, 0.780369043867433,
                  0.800308744139141, 0.819400310737932, 0.837623511228187, 0.854959033434602,
                  0.871388505909297, 0.886894517402421, 0.901460635315852, 0.915071423120898,
                  0.927712456722309, 0.939370339752755, 0.950032717784438, 0.959688291448743,
                  0.968326828463264, 0.975939174585137, 0.982517263563015, 0.988054126329624,
                  0.992543900323763, 0.995981842987209, 0.998364375863182, 0.999689503883231};

double W96[48] = {/* weights for 96-point Gauss quadrature */
                  0.032550614492363, 0.032516118713869, 0.032447163714064, 0.032343822568576,
                  0.032206204794030, 0.032034456231993, 0.031828758894411, 0.031589330770727,
                  0.031316425596861, 0.031010332586314, 0.030671376123669, 0.030299915420828,
                  0.029896344136328, 0.029461089958168, 0.028994614150555, 0.028497411065085,
                  0.027970007616848, 0.027412962726029, 0.026826866725592, 0.026212340735672,
                  0.025570036005349, 0.024900633222484, 0.024204841792365, 0.023483399085926,
                  0.022737069658329, 0.021966644438744, 0.021172939892191, 0.020356797154333,
                  0.019519081140145, 0.018660679627411, 0.017782502316045, 0.016885479864245,
                  0.015970562902562, 0.015038721026995, 0.014090941772315, 0.013128229566962,
                  0.012151604671088, 0.011162102099839, 0.010160770535008, 0.009148671230783,
                  0.008126876925699, 0.007096470791154, 0.006058545504236, 0.005014202742928,
                  0.003964554338445, 0.002910731817935, 0.001853960788947, 0.000796792065552};

__constant__ double dev_A96[48]; // Abscissas for Gauss-Quadrature
__constant__ double dev_W96[48]; // Weights for Gauss-Quadrature

__constant__ double dev_dz, dev_z_max; // Redshift bin and maximal redshift
double dz, z_max;

__constant__ double dev_dk, dev_k_min, dev_k_max;
double dk, k_min, k_max;

__constant__ int dev_n_redshift_bins; // Number of redshift bins
__constant__ int dev_n_kbins;

__constant__ double dev_H0_over_c; // Hubble constant/speed of light [h s/m]
__constant__ double dev_c_over_H0; // Speed of light / Hubble constant [h^-1 m/s]

void copyConstants()
{
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A96, &A96, 48 * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_W96, &W96, 48 * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_H0_over_c, &H0_over_c, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_c_over_H0, &c_over_H0, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_redshift_bins, &n_redshift_bins, sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_kbins, &n_kbins, sizeof(int)));
}

void set_cosmology(cosmology cosmo_arg, double* dev_g_array, double* dev_p_array, std::vector<std::vector<double>> *nz, std::vector<double> *sigma_epsilon_per_bin, std::vector<double> *ngal_per_bin, double * dev_sigma_epsilon, double * dev_ngal, std::vector<double> *P_k, double dk, double kmin, double kmax)
{
#if T17_CORRECTION
  std::cerr << "*****************************************************" << std::endl;
  std::cerr << "WARNING: Applying T+17 corrections to power spectrum!" << std::endl;
  std::cerr << "*****************************************************" << std::endl;
#endif
 
  // set cosmology
  cosmo = cosmo_arg;

  // Copy Cosmological Parameters (constant memory)
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_h, &cosmo.h, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma8, &cosmo.sigma8, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_omb, &cosmo.omb, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_omc, &cosmo.omc, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_ns, &cosmo.ns, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_w, &cosmo.w, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_om, &cosmo.om, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_ow, &cosmo.ow, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A_IA, &cosmo.A_IA, sizeof(double)));

  // Copy P_k and binning
  bool Pk_given = (P_k != NULL);
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_Pk_given, &Pk_given, sizeof(bool)));
  if (Pk_given)
  {
    std::cerr << "Using precomputed linear power spectrum" << std::endl;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_dk, &dk, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_k_min, &kmin, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_k_max, &kmax, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_Pk, (P_k->data()), n_kbins * sizeof(double)));
  }
  else
  {
    std::cerr << "Computing linear power spectrum on the fly (using Eisenstein & Hu)" << std::endl;
  }

  // Calculate Norm and copy
  norm_P = 1; // Initial setting, is overridden in next step
  norm_P = cosmo.sigma8 / sigmam(8., 0);
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_norm, &norm_P, sizeof(double)));
  // Copy redshift binning
  z_max = cosmo.zmax;
  dz = z_max / n_redshift_bins;
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_dz, &dz, sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_z_max, &z_max, sizeof(double)));

  // Calculate f_K(z) and g(z)

  // First: f_K
#pragma omp parallel for
  for (int i = 0; i < n_redshift_bins; i++)
  {
    double z_now = (i+0.5) * dz;
    f_K_array[i] = f_K_at_z(z_now);
  };

    // Second: g and p
  int Ntomo = nz->size();
  double g_array_allTomo[Ntomo*n_redshift_bins];
  double p_array_allTomo[Ntomo*n_redshift_bins];


#pragma omp parallel for
   for (int k = 0; k < Ntomo; k++){
    for (int i = 0; i < n_redshift_bins; i++)
    {
      g_array_allTomo[k*n_redshift_bins+i] = 0;

      p_array_allTomo[k*n_redshift_bins+i] = nz->at(k).at(i);

      // perform trapezoidal integration
      for (int j = i; j < n_redshift_bins; j++)
      {
        double nz_znow;
        nz_znow = nz->at(k).at(j);
        if (j == i || j == n_redshift_bins - 1)
        {

          g_array_allTomo[k*n_redshift_bins+i] += nz_znow * (f_K_array[j] - f_K_array[i]) / f_K_array[j] / 2;

        }
        else
        {

          g_array_allTomo[k*n_redshift_bins+i] += nz_znow * (f_K_array[j] - f_K_array[i]) / f_K_array[j];

        }
      }

      g_array_allTomo[k*n_redshift_bins+i] = g_array_allTomo[k*n_redshift_bins+i] * dz;

      //std::cerr << k << "\t" << i << "\t" <<  g_array[k][i] << std::endl;
    }

    g_array_allTomo[k*n_redshift_bins] = 1.;
  }
  std::cerr << "Finished calculating f_k, p and g" << std::endl;


  // Copy f_k and g to device (constant memory)
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_f_K_array, f_K_array, n_redshift_bins * sizeof(double)));
    
  CUDA_SAFE_CALL(cudaMemcpy(dev_g_array, g_array_allTomo, Ntomo*n_redshift_bins*sizeof(double), cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL(cudaMemcpy(dev_p_array, p_array_allTomo, Ntomo*n_redshift_bins*sizeof(double), cudaMemcpyHostToDevice ));

  if (sigma_epsilon_per_bin!=NULL & ngal_per_bin!=NULL)
  {
    CUDA_SAFE_CALL(cudaMemcpy(dev_sigma_epsilon, sigma_epsilon_per_bin, Ntomo*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_ngal, ngal_per_bin, Ntomo*sizeof(double), cudaMemcpyHostToDevice));
  }
  else
  {
    std::cerr<<"Warning: Shapenoise is not set!"<<std::endl;
  }

  // Calculating Non-linear scales
  double r_sigma_array[n_redshift_bins];
  double n_eff_array[n_redshift_bins];
  double ncur_array[n_redshift_bins];

#pragma omp parallel for
  for (int i = 0; i < n_redshift_bins; i++)
  {
    double z_now = (i+0.5) * dz;
    D1_array[i] = lgr(z_now) / lgr(0.);           // linear growth factor
    r_sigma_array[i] = calc_r_sigma(D1_array[i]); // =1/k_NL [Mpc/h] in Eq.(B1)
    double d1 = -2. * pow(D1_array[i] * sigmam(r_sigma_array[i], 2), 2);
    n_eff_array[i] = -3. + 2. * pow(D1_array[i] * sigmam(r_sigma_array[i], 2), 2); // n_eff in Eq.(B2)
    ncur_array[i] = d1 * d1 + 4. * sigmam(r_sigma_array[i], 3) * pow(D1_array[i], 2);
  }
  std::cerr << "Finished calculating non linear scales" << std::endl;
  // Copy non-linear scales to device
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_D1_array, D1_array, n_redshift_bins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_r_sigma_array, r_sigma_array, n_redshift_bins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_eff_array, n_eff_array, n_redshift_bins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_ncur_array, ncur_array, n_redshift_bins * sizeof(double)));
}

__device__ double bispec(double k1, double k2, double k3, double z, int idx, double didx)
{
  int i, j;
  double q[4], qt, logsigma8z, r1, r2;
  double an, bn, cn, en, fn, gn, hn, mn, nn, pn, alphan, betan, mun, nun, BS1h, BS3h, PSE[4];
  double r_sigma, n_eff, D1, ncur;
  compute_coefficients(idx, didx, &D1, &r_sigma, &n_eff, &ncur);

  if (z > 10.)
    return bispec_tree(k1, k2, k3, z, D1);

  q[1] = k1 * r_sigma, q[2] = k2 * r_sigma, q[3] = k3 * r_sigma; // dimensionless wavenumbers

  // sorting q[i] so that q[1]>=q[2]>=q[3]
  for (i = 1; i <= 3; i++)
  {
    for (j = i + 1; j <= 3; j++)
    {
      if (q[i] < q[j])
      {
        qt = q[j];
        q[j] = q[i];
        q[i] = qt;
      }
    }
  }
  r1 = q[3] / q[1], r2 = (q[2] + q[3] - q[1]) / q[1]; // Eq.(B8)

  q[1] = k1 * r_sigma, q[2] = k2 * r_sigma, q[3] = k3 * r_sigma;
  logsigma8z = log10(D1 * dev_sigma8);

  // 1-halo term parameters in Eq.(B7)
  an = pow(10., -2.167 - 2.944 * logsigma8z - 1.106 * pow(logsigma8z, 2) - 2.865 * pow(logsigma8z, 3) - 0.310 * pow(r1, pow(10., 0.182 + 0.57 * n_eff)));
  bn = pow(10., -3.428 - 2.681 * logsigma8z + 1.624 * pow(logsigma8z, 2) - 0.095 * pow(logsigma8z, 3));
  cn = pow(10., 0.159 - 1.107 * n_eff);
  alphan = pow(10., -4.348 - 3.006 * n_eff - 0.5745 * pow(n_eff, 2) + pow(10., -0.9 + 0.2 * n_eff) * pow(r2, 2));
  if (alphan > 1. - (2. / 3.) * dev_ns)
    alphan = 1. - (2. / 3.) * dev_ns;
  betan = pow(10., -1.731 - 2.845 * n_eff - 1.4995 * pow(n_eff, 2) - 0.2811 * pow(n_eff, 3) + 0.007 * r2);

  // 1-halo term bispectrum in Eq.(B4)
  BS1h = 1.;
  for (i = 1; i <= 3; i++)
  {
    BS1h *= 1. / (an * pow(q[i], alphan) + bn * pow(q[i], betan)) / (1. + 1. / (cn * q[i]));
  }

  // 3-halo term parameters in Eq.(B9)
  fn = pow(10., -10.533 - 16.838 * n_eff - 9.3048 * pow(n_eff, 2) - 1.8263 * pow(n_eff, 3));
  gn = pow(10., 2.787 + 2.405 * n_eff + 0.4577 * pow(n_eff, 2));
  hn = pow(10., -1.118 - 0.394 * n_eff);
  mn = pow(10., -2.605 - 2.434 * logsigma8z + 5.71 * pow(logsigma8z, 2));
  nn = pow(10., -4.468 - 3.08 * logsigma8z + 1.035 * pow(logsigma8z, 2));
  mun = pow(10., 15.312 + 22.977 * n_eff + 10.9579 * pow(n_eff, 2) + 1.6586 * pow(n_eff, 3));
  nun = pow(10., 1.347 + 1.246 * n_eff + 0.4525 * pow(n_eff, 2));
  pn = pow(10., 0.071 - 0.433 * n_eff);
  en = pow(10., -0.632 + 0.646 * n_eff);

  for (i = 1; i <= 3; i++)
  {
    PSE[i] = (1. + fn * pow(q[i], 2)) / (1. + gn * q[i] + hn * pow(q[i], 2)) * pow(D1, 2) * linear_pk(q[i] / r_sigma) + 1. / (mn * pow(q[i], mun) + nn * pow(q[i], nun)) / (1. + pow(pn * q[i], -3)); // enhanced P(k) in Eq.(B6)
  }

  // 3-halo term bispectrum in Eq.(B5)
  BS3h = 2. * (F2(k1, k2, k3, z, D1, r_sigma) * PSE[1] * PSE[2] + F2(k2, k3, k1, z, D1, r_sigma) * PSE[2] * PSE[3] + F2(k3, k1, k2, z, D1, r_sigma) * PSE[3] * PSE[1]);
  for (i = 1; i <= 3; i++)
    BS3h *= 1. / (1. + en * q[i]);
  // if (BS1h+BS3h>=0)
  return BS1h + BS3h;
  // else return 0;
}

__device__ double bkappa(double ell1, double ell2, double ell3, int zbin1, int zbin2, int zbin3, double* dev_g, double* dev_p, int Ntomo)
{
  if (ell1 == 0 || ell2 == 0 || ell3 == 0)
    return 0; // WARNING! THIS MIGHT SCREW WITH THE INTEGRATION ROUTINE!
  return GQ96_of_bdelta(0, dev_z_max, ell1, ell2, ell3, zbin1, zbin2, zbin3, dev_g, dev_p, Ntomo);
}

__device__ double GQ96_of_bdelta(double a, double b, double ell1, double ell2, double ell3, int zbin1, int zbin2, int zbin3, double* dev_g, double* dev_p, int Ntomo)
{
  double cx = (a + b) / 2;
  double dx = (b - a) / 2;
  double q = 0;
  for (int i = 0; i < 48; i++)
    q += dev_W96[i] * (integrand_bkappa(cx - dx * dev_A96[i], ell1, ell2, ell3, zbin1, zbin2, zbin3, dev_g, dev_p, Ntomo) 
    + integrand_bkappa(cx + dx * dev_A96[i], ell1, ell2, ell3, zbin1, zbin2, zbin3, dev_g, dev_p, Ntomo));
  return q * dx;
}

__device__ double integrand_bkappa(double z, double ell1, double ell2, double ell3, int zbin1, int zbin2, int zbin3, double* dev_g, double* dev_p, int Ntomo)
{
  if (z < 1.0e-7)
    return 0.;
  if (ell1 <= 1.0e-10 || ell2 <= 1.0e-10 || ell3 <= 1.0e-10)
  {
    return 0;
  }

  double didx = z / dev_z_max * (dev_n_redshift_bins);
  int idx = didx;
  didx = didx - idx;
   double r_sigma, n_eff, D1, ncur;
  compute_coefficients(idx, didx, &D1, &r_sigma, &n_eff, &ncur);

  double g_value_1 = g_interpolated(idx, didx, zbin1, dev_g, Ntomo);
  double g_value_2 = g_interpolated(idx, didx, zbin2, dev_g, Ntomo);
  double g_value_3 = g_interpolated(idx, didx, zbin3, dev_g, Ntomo);
  double p_value_1 = p_interpolated(idx, didx, zbin1, dev_p, Ntomo);
  double p_value_2 = p_interpolated(idx, didx, zbin2, dev_p, Ntomo);
  double p_value_3 = p_interpolated(idx, didx, zbin3, dev_p, Ntomo);
  double f_K_value = f_K_interpolated(idx, didx);

  double C1_rho_crit = 0.013873073650776856;
  double f_IA = - dev_A_IA * dev_om * C1_rho_crit / D1;
  
  double W_1 = limber_integrand_prefactor_delta(z, g_value_1);
  double W_2 = limber_integrand_prefactor_delta(z, g_value_2);
  double W_3 = limber_integrand_prefactor_delta(z, g_value_3);

  double dz_dchi = E(z)*dev_H0_over_c;

  double bispectrum = bispec(ell1 / f_K_value, ell2 / f_K_value, ell3 / f_K_value, z, idx, didx);

  double B_G_G_G = W_1 * W_2 * W_3 / f_K_value / dz_dchi * bispectrum;
  double B_G_G_I = W_1 * W_2 * p_value_3 / pow(f_K_value,2) * f_IA * bispectrum;
  double B_G_I_G = W_1 * p_value_2 * W_3 / pow(f_K_value,2) * f_IA * bispectrum;
  double B_I_G_G = p_value_1 * W_2 * W_3 / pow(f_K_value,2) * f_IA * bispectrum;
  
  double B_G_I_I = W_1 * p_value_2 * p_value_3 / pow(f_K_value,3) * dz_dchi * f_IA * f_IA * bispectrum;
  double B_I_G_I = p_value_1 * W_2 * p_value_3 / pow(f_K_value,3) * dz_dchi * f_IA * f_IA * bispectrum;
  double B_I_I_G = p_value_1 * p_value_2 * W_3 / pow(f_K_value,3) * dz_dchi * f_IA * f_IA * bispectrum;

  double B_I_I_I =  p_value_1 * p_value_2 * p_value_3 / pow(f_K_value,4) * pow(dz_dchi,2) * f_IA * f_IA * f_IA * bispectrum;

  double result = (B_G_G_G + B_I_I_I + B_G_I_I + B_I_G_I + B_I_I_G + B_G_G_I + B_G_I_G + B_I_G_G);
  
  if (isnan(result))
  {
    printf("nan in bispec!"); // %lf, %lf, %lf, %lf, %.3f, %lf, %lf, %lf \n", f_K_value, ell1, ell2, ell3, z, idx, didx, dev_n_redshift_bins);
    return 0;
  }
  return result;
}

__host__ __device__ double g_interpolated(int idx, double didx, int zbin, double* dev_g, int Ntomo)
{
  int interpolate_idx = idx;
  double weight = 1 - abs(didx - 0.5);
  double interpolate_weight = 1 - weight;
  if (didx > 0.5)
  {
  #ifdef __CUDA_ARCH__
    if (idx == dev_n_redshift_bins - 1)
  #else
    if (idx == n_redshift_bins - 1)
#endif
    {
      interpolate_idx = idx;
      interpolate_weight = 0.0;
    }
    else
    {
      interpolate_idx = idx + 1;
    }
  }
  else if (didx < 0.5)
  {
    if(idx==0)
    {
      interpolate_idx=idx;
    interpolate_weight=0.0;
    }
    else
    {
      interpolate_idx=idx-1;
    }
  }
#ifdef __CUDA_ARCH__
  return dev_g[zbin*n_redshift_bins+idx] * weight + dev_g[zbin*n_redshift_bins+interpolate_idx] * interpolate_weight;
#else
  return dev_g[zbin*n_redshift_bins+idx] * weight + dev_g[zbin*n_redshift_bins+interpolate_idx] * interpolate_weight;
#endif

}


__host__ __device__ double p_interpolated(int idx, double didx, int zbin, double* dev_p, int Ntomo)
{
  int interpolate_idx = idx;
  double weight = 1 - abs(didx - 0.5);
  double interpolate_weight = 1 - weight;
  if (didx > 0.5)
  {
  #ifdef __CUDA_ARCH__
    if (idx == dev_n_redshift_bins - 1)
  #else
    if (idx == n_redshift_bins - 1)
#endif
    {
      interpolate_idx = idx;
      interpolate_weight = 0.0;
    }
    else
    {
      interpolate_idx = idx + 1;
    }
  }
  else if (didx < 0.5)
  {
    if(idx==0)
    {
      interpolate_idx=idx;
    interpolate_weight=0.0;
    }
    else
    {
      interpolate_idx=idx-1;
    }
  }
#ifdef __CUDA_ARCH__
  return dev_p[zbin*n_redshift_bins+idx] * weight + dev_p[zbin*n_redshift_bins+interpolate_idx] * interpolate_weight;
#else
  return dev_p[zbin*n_redshift_bins+idx] * weight + dev_p[zbin*n_redshift_bins+interpolate_idx] * interpolate_weight;
#endif

}

__host__ __device__ double f_K_interpolated(int idx, double didx)
{
  int interpolate_idx = idx;
  double weight = 1 - abs(didx - 0.5);
  double interpolate_weight = 1 - weight;
  if (didx > 0.5)
  {
  #ifdef __CUDA_ARCH__
    if (idx == dev_n_redshift_bins - 1)
  #else
    if (idx == n_redshift_bins - 1)
#endif
    {
      interpolate_idx = idx;
      interpolate_weight = 0.0;
    }
    else
    {
      interpolate_idx = idx + 1;
    }
  }
  else if (didx < 0.5)
  {
    if(idx==0)
    {
      interpolate_idx=idx;
    interpolate_weight=0.0;
    }
    else
    {
      interpolate_idx=idx-1;
    }
  }
  #ifdef __CUDA_ARCH__
  return dev_f_K_array[idx]* weight + dev_f_K_array[interpolate_idx]* interpolate_weight;
#else
  return f_K_array[idx] * weight + f_K_array[interpolate_idx]* interpolate_weight;
#endif

}

__device__ void compute_coefficients(int idx, double didx, double *D1, double *r_sigma, double *n_eff, double *ncur)
{
  int interpolate_idx = idx;
  double weight = 1 - abs(didx - 0.5);
  double interpolate_weight = 1 - weight;
  if (didx > 0.5)
  {

  #ifdef __CUDA_ARCH__
    if (idx == dev_n_redshift_bins - 1)
  #else
    if (idx == n_redshift_bins - 1)
#endif
    {
      interpolate_idx = idx;
      interpolate_weight = 0.0;
    }
    else
    {
      interpolate_idx = idx + 1;
    }
  }
  else if (didx < 0.5)
  {
    if(idx==0){
      interpolate_idx=idx;
      interpolate_weight=0.0;
    }
    else{
      interpolate_idx=idx-1;
    }
  }
#ifdef __CUDA_ARCH__
  *D1 = dev_D1_array[idx] * weight + dev_D1_array[interpolate_idx] * interpolate_weight;
  *r_sigma = dev_r_sigma_array[idx] * weight + dev_r_sigma_array[interpolate_idx] * interpolate_weight;
  *n_eff = dev_n_eff_array[idx] * weight + dev_n_eff_array[interpolate_idx] * interpolate_weight;
  *ncur = dev_ncur_array[idx] * weight + dev_ncur_array[interpolate_idx] * interpolate_weight;
#else
  *D1 = D1_array[idx] * weight + D1_array[interpolate_idx] * interpolate_weight;
  *r_sigma = r_sigma_array[idx] * weight + r_sigma_array[interpolate_idx] * interpolate_weight;
  *n_eff = n_eff_array[idx] * weight + n_eff_array[interpolate_idx] * interpolate_weight;
  *ncur = ncur_array[idx] * weight + ncur_array[interpolate_idx] * interpolate_weight;
#endif

}

__host__ __device__ double om_m_of_z(double z)
{
  double aa = 1. / (1 + z);
#ifndef __CUDA_ARCH__
  return cosmo.om / (cosmo.om + aa * (aa * aa * cosmo.ow + (1. - cosmo.om - cosmo.ow)));
#else
  return dev_om / (dev_om + aa * (aa * aa * dev_ow + (1. - dev_om - dev_ow)));
#endif
}

__host__ __device__ double om_v_of_z(double z)
{
  double aa = 1. / (1 + z);
#ifndef __CUDA_ARCH__
  return cosmo.ow * aa * aa * aa / (cosmo.om + aa * (aa * aa * cosmo.ow + (1. - cosmo.om - cosmo.ow)));
#else
  return dev_ow * aa * aa * aa / (dev_om + aa * (aa * aa * dev_ow + (1. - dev_om - dev_ow)));
#endif
}

__global__ void limber_integrand_wrapper(const double *vars, unsigned ndim, size_t npts, int zbin1, int zbin2, double ell, double* dev_g, double* dev_p, int Ntomo, double *value)
{
  // index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-Stride loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double z = vars[i * ndim];
    value[i] = limber_integrand_power_spectrum(ell, z, zbin1, zbin2, dev_g, dev_p, Ntomo);
  }
}

int limber_integrand_wrapper(unsigned ndim, size_t npts, const double *vars, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1)
  {
    std::cerr << "integrand: Wrong number of function dimensions" << std::endl;
    exit(1);
  };
  if (ndim != 1)
  {
    std::cerr << "integrand: Wrong number of variable dimensions" << std::endl;
    exit(1);
  };
  SpectraContainer *container = (SpectraContainer *)thisPtr;

  int zbin1 = container->zbins.at(0);
  int zbin2 = container->zbins.at(1);

  // Read data for integration
  double ell = container->ell;

  double * dev_g = container->dev_g;
  double * dev_p = container->dev_p;

  int Ntomo = container->Ntomo;

  // Allocate memory on device for integrand values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

  // Copy integration variables to device
  double *dev_vars;
  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // allocate memory
  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying

  // Calculate values
  limber_integrand_wrapper<<<BLOCKS, THREADS>>>(dev_vars, ndim, npts, zbin1, zbin2, ell, dev_g, dev_p, Ntomo, dev_value);


  cudaFree(dev_vars); // Free variables

  // Copy results to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Free values

  return 0; // Success :)
}

double Pell(double ell, int zbin1, int zbin2, double *dev_g, double * dev_p, int Ntomo, std::vector<double>* sigma_epsilon,  std::vector<double>* ngal)
{
    SpectraContainer container;
    container.zbins = {zbin1, zbin2};
    container.ell = ell;
    container.dev_g = dev_g;
    container.dev_p = dev_p;
    container.Ntomo = Ntomo;

    double vals_min[1] = {0};
    double vals_max[1] = {z_max};
    double result, error;

    double Pshapenoise=0;
    if (zbin1==zbin2)
    {
      Pshapenoise=0.5*pow(sigma_epsilon->at(zbin1),2)/ngal->at(zbin1);
    };

    hcubature_v(1, limber_integrand_wrapper, &container, 1, vals_min, vals_max, 0, 0, 1e-6, ERROR_L1, &result, &error);

    //std::cerr<<ell<<" "<<result<<" "<<error<<" "<<P_shapenoise<<std::endl;
    return result+Pshapenoise;
}

__host__ __device__ double P_k_nonlinear(double k, double z)
{
/* get the interpolation coefficients */
#ifdef __CUDA_ARCH__
  double didx = z / dev_z_max * (dev_n_redshift_bins );
#else
  double didx = z / z_max * (n_redshift_bins);
#endif
  int idx = didx;
  didx = didx - idx;

  double r_sigma, n_eff, D1, ncur;
  compute_coefficients(idx, didx, &D1, &r_sigma, &n_eff, &ncur);

  double a, b, c, gam, alpha, beta, xnu, y, ysqr, ph, pq, f1, f2, f3;
  double f1a, f2a, f3a, f1b, f2b, f3b, frac;
  double plin, delta_nl;
  double om_m, om_v;
  double nsqr = n_eff * n_eff;

#ifdef __CUDA_ARCH__
  if (abs(dev_om + dev_ow - 1) > 1e-4)
  {
    printf("Warning: omw as a function of redshift only implemented for flat universes yet!");
  }
  double w = dev_w;
#else
  if (abs(cosmo.om + cosmo.ow - 1) > 1e-4)
  {
    std::cerr << "Warning: omw as a function of redshift only implemented for flat universes yet!" << std::endl;
  }
  double w = cosmo.w;
#endif

  om_m = om_m_of_z(z);
  om_v = om_v_of_z(z);

  f1a = pow(om_m, (-0.0732));
  f2a = pow(om_m, (-0.1423));
  f3a = pow(om_m, (0.0725));
  f1b = pow(om_m, (-0.0307));
  f2b = pow(om_m, (-0.0585));
  f3b = pow(om_m, (0.0743));
  frac = om_v / (1. - om_m);
  f1 = frac * f1b + (1 - frac) * f1a;
  f2 = frac * f2b + (1 - frac) * f2a;
  f3 = frac * f3b + (1 - frac) * f3a;
  a = 1.5222 + 2.8553 * n_eff + 2.3706 * nsqr + 0.9903 * n_eff * nsqr + 0.2250 * nsqr * nsqr - 0.6038 * ncur + 0.1749 * om_v * (1.0 + w);
  a = pow(10.0, a);
  b = pow(10.0, -0.5642 + 0.5864 * n_eff + 0.5716 * nsqr - 1.5474 * ncur + 0.2279 * om_v * (1.0 + w));
  c = pow(10.0, 0.3698 + 2.0404 * n_eff + 0.8161 * nsqr + 0.5869 * ncur);
  gam = 0.1971 - 0.0843 * n_eff + 0.8460 * ncur;
  alpha = fabs(6.0835 + 1.3373 * n_eff - 0.1959 * nsqr - 5.5274 * ncur);
  beta = 2.0379 - 0.7354 * n_eff + 0.3157 * nsqr + 1.2490 * n_eff * nsqr + 0.3980 * nsqr * nsqr - 0.1682 * ncur;
  xnu = pow(10.0, 5.2105 + 3.6902 * n_eff);


  plin = linear_pk(k) * D1 * D1 * k * k * k / (2 * M_PI * M_PI);


  y = k * r_sigma;
  ysqr = y * y;
  ph = a * pow(y, f1 * 3) / (1 + b * pow(y, f2) + pow(f3 * c * y, 3 - gam));
  ph = ph / (1 + xnu / ysqr);
  pq = plin * pow(1 + plin, beta) / (1 + plin * alpha) * exp(-y / 4.0 - ysqr / 8.0);

  delta_nl = pq + ph;

  return (2 * M_PI * M_PI * delta_nl / (k * k * k));
}

__host__ __device__ double linear_pk(double k)
{
#ifdef __CUDA_ARCH__
  if (dev_Pk_given)
  {
    if (k >= dev_k_max)
      return dev_Pk[dev_n_kbins - 1];

    if (k <= dev_k_min)
      return dev_Pk[0];

    double didx = (log(k / dev_k_min) / dev_dk); // lower index (Warning: Pk is logarithmically binned!)
    int idx = didx;
    didx = didx - idx;

    if (idx == dev_n_kbins - 1)
      return dev_Pk[dev_n_kbins - 1];

    return dev_Pk[idx] * (1 - didx) + dev_Pk[idx + 1] * didx;
  };
#else
  if (Pk_given)
  {
    if (k >= k_max)
      return Pk[n_kbins - 1];

    if (k <= k_min)
      return Pk[0];

    double didx = (log(k / k_min) / dk); // lower index (Warning: Pk is logarithmically binned!)
    int idx = didx;
    didx = didx - idx;

    if (idx == n_kbins - 1)
      return Pk[n_kbins - 1];

    return Pk[idx] * (1 - didx) + Pk[idx + 1] * didx;
  };
#endif

  // Use Eisenstein & Hu if the linear P(k) is not given
  double pk, delk, alnu, geff, qeff, L, C;
#ifdef __CUDA_ARCH__
  k *= dev_h; // unit conversion from [h/Mpc] to [1/Mpc]
#else
  k*=cosmo.h;
#endif

#ifdef __CUDA_ARCH__
  double fc = dev_omc / dev_om;
  double fb = dev_omb / dev_om;
#else
  double fc = cosmo.omc / cosmo.om;
  double fb = cosmo.omb / cosmo.om;
#endif
  double theta = 2.728 / 2.7;
  double pc = 0.25 * (5.0 - sqrt(1.0 + 24.0 * fc));
#ifdef __CUDA_ARCH__
  double omh2 = dev_om * dev_h * dev_h;
  double ombh2 = dev_omb * dev_h * dev_h;
#else
  double omh2 = cosmo.om * cosmo.h * cosmo.h;
  double ombh2 = cosmo.omb * cosmo.h * cosmo.h;
#endif
  double zeq = 2.5e+4 * omh2 / pow(theta, 4);
  double b1 = 0.313 * pow(omh2, -0.419) * (1.0 + 0.607 * pow(omh2, 0.674));
  double b2 = 0.238 * pow(omh2, 0.223);
  double zd = 1291.0 * pow(omh2, 0.251) / (1.0 + 0.659 * pow(omh2, 0.828)) * (1.0 + b1 * pow(ombh2, b2));
  double yd = (1.0 + zeq) / (1.0 + zd);
  double sh = 44.5 * log(9.83 / (omh2)) / sqrt(1.0 + 10.0 * pow(ombh2, 0.75));

  alnu = fc * (5.0 - 2.0 * pc) / 5.0 * (1.0 - 0.553 * fb + 0.126 * fb * fb * fb) * pow(1.0 + yd, -pc) * (1.0 + 0.5 * pc * (1.0 + 1.0 / (7.0 * (3.0 - 4.0 * pc))) / (1.0 + yd));

  geff = omh2 * (sqrt(alnu) + (1.0 - sqrt(alnu)) / (1.0 + pow(0.43 * k * sh, 4)));
  qeff = k / geff * theta * theta;

  L = log(2.718281828 + 1.84 * sqrt(alnu) * qeff / (1.0 - 0.949 * fb));
  C = 14.4 + 325.0 / (1.0 + 60.5 * pow(qeff, 1.11));

#ifdef __CUDA_ARCH__
  delk = pow(dev_norm, 2) * pow(k * 2997.9 / dev_h, 3. + dev_ns) * pow(L / (L + C * qeff * qeff), 2);
#else
  delk = pow(norm_P, 2) * pow(k * 2997.9 / cosmo.h, 3. + cosmo.ns) * pow(L / (L + C * qeff * qeff), 2);
#endif
  pk = 2.0 * M_PI * M_PI / (k * k * k) * delk;

#ifdef __CUDA_ARCH__
  return dev_h * dev_h * dev_h * pk;
#else
  return cosmo.h * cosmo.h * cosmo.h * pk;
#endif
}

__device__ double bispec_tree(double k1, double k2, double k3, double z, double D1) // tree-level BS [(Mpc/h)^6]
{
  return pow(D1, 4) * 2. * (F2_tree(k1, k2, k3) * linear_pk(k1) * linear_pk(k2) + F2_tree(k2, k3, k1) * linear_pk(k2) * linear_pk(k3) + F2_tree(k3, k1, k2) * linear_pk(k3) * linear_pk(k1));
}

__device__ double F2(double k1, double k2, double k3, double z, double D1, double r_sigma)
{
  double a, q[4], dn, omz, logsigma8z;

  q[3] = k3 * r_sigma;

  logsigma8z = log10(D1 * dev_sigma8);
  a = 1. / (1. + z);
  omz = dev_om / (dev_om + dev_ow * pow(a, -3. * dev_w)); // Omega matter at z

  dn = pow(10., -0.483 + 0.892 * logsigma8z - 0.086 * omz);

  return F2_tree(k1, k2, k3) + dn * q[3];
}

__device__ double F2_tree(double k1, double k2, double k3) // F2 kernel in tree level
{
  double costheta12 = 0.5 * (k3 * k3 - k1 * k1 - k2 * k2) / (k1 * k2);
  return (5. / 7.) + 0.5 * costheta12 * (k1 / k2 + k2 / k1) + (2. / 7.) * costheta12 * costheta12;
}

double f_K_at_z(double z)
{
  return c_over_H0 * GQ96_of_Einv(0, z);
}

double lgr(double z) // linear growth factor at z (not normalized at z=0)
{
  int i, j, n;
  double a, a0, x, h, yp;
  double k1[2], k2[2], k3[2], k4[2], y[2], y2[2], y3[2], y4[2];

  a = 1. / (1. + z);
  a0 = 1. / 1100.;

  yp = -1.;
  n = 10;

  for (;;)
  {
    n *= 2;
    h = (log(a) - log(a0)) / n;

    x = log(a0);
    y[0] = 1., y[1] = 0.;
    for (i = 0; i < n; i++)
    {
      for (j = 0; j < 2; j++)
        k1[j] = h * lgr_func(j, x, y);

      for (j = 0; j < 2; j++)
        y2[j] = y[j] + 0.5 * k1[j];
      for (j = 0; j < 2; j++)
        k2[j] = h * lgr_func(j, x + 0.5 * h, y2);

      for (j = 0; j < 2; j++)
        y3[j] = y[j] + 0.5 * k2[j];
      for (j = 0; j < 2; j++)
        k3[j] = h * lgr_func(j, x + 0.5 * h, y3);

      for (j = 0; j < 2; j++)
        y4[j] = y[j] + k3[j];
      for (j = 0; j < 2; j++)
        k4[j] = h * lgr_func(j, x + h, y4);

      for (j = 0; j < 2; j++)
        y[j] += (k1[j] + k4[j]) / 6. + (k2[j] + k3[j]) / 3.;
      x += h;
    }

    if (fabs(y[0] / yp - 1.) < 0.1 * eps)
      break;
    yp = y[0];
  }

  return a * y[0];
}

double lgr_func(int j, double la, double y[2])
{
  if (j == 0)
    return y[1];
  if (j == 1)
  {
    double g, a;
    a = exp(la);
    g = -0.5 * (5. * cosmo.om + (5. - 3 * cosmo.w) * cosmo.ow * pow(a, -3. * cosmo.w)) * y[1] - 1.5 * (1. - cosmo.w) * cosmo.ow * pow(a, -3. * cosmo.w) * y[0];
    g = g / (cosmo.om + cosmo.ow * pow(a, -3. * cosmo.w));
    return g;
  };

  // This is only reached, if j is not a valid value
  std::cerr << "lgr_func: j not a valid value. Exiting \n";
  exit(1);
}

double sigmam(double r, int j) // r[Mpc/h]
{
  if (cosmo.sigma8 < 1e-8)
    return 0;

  int n, i;
  double k1, k2, xx, xxp, xxpp, k, a, b, hh;

  k1 = 2. * M_PI / r;
  k2 = 2. * M_PI / r;

  xxpp = -1.0;
  for (;;)
  {
    k1 = k1 / 10.0;
    k2 = k2 * 2.0;

    a = log(k1), b = log(k2);

    xxp = -1.0;
    n = 2;
    for (;;)
    {
      n = n * 2;
      hh = (b - a) / (double)n;

      xx = 0.;
      for (i = 1; i < n; i++)
      {
        k = exp(a + hh * i);
        if (j < 3)
          xx += k * k * k * linear_pk(k) * pow(window(k * r, j), 2);
        else
          xx += k * k * k * linear_pk(k) * window(k * r, j);
      }
      if (j < 3)
        xx += 0.5 * (k1 * k1 * k1 * linear_pk(k1) * pow(window(k1 * r, j), 2) + k2 * k2 * k2 * linear_pk(k2) * pow(window(k2 * r, j), 2));
      else
        xx += 0.5 * (k1 * k1 * k1 * linear_pk(k1) * window(k1 * r, j) + k2 * k2 * k2 * linear_pk(k2) * window(k2 * r, j));

      xx *= hh;

      if (fabs((xx - xxp) / xx) < eps)
        break;
      xxp = xx;
    }

    if (fabs((xx - xxpp) / xx) < eps)
      break;
    xxpp = xx;
  }

  if (j < 3)
    return sqrt(xx / (2.0 * M_PI * M_PI));
  else
    return xx / (2.0 * M_PI * M_PI);
}

double window(double x, int i)
{
  if (i == 0)
    return 3.0 / pow(x, 3) * (sin(x) - x * cos(x)); // top hat
  if (i == 1)
    return exp(-0.5 * x * x); // gaussian
  if (i == 2)
    return x * exp(-0.5 * x * x); // 1st derivative gaussian
  if (i == 3)
    return x * x * (1 - x * x) * exp(-x * x);
  if (i == 4)
    return (3 * (x * x - 3) * sin(x) + 9 * x * cos(x)) / x / x / x / x; // 1st derivative top hat
  printf("window ran out \n");
  return -1;
}

double calc_r_sigma(double D1) // return r_sigma[Mpc/h] (=1/k_sigma)
{
  if (cosmo.sigma8 < 1e-8)
    return 0;

  double k, k1, k2;

  k1 = k2 = 1.;
  for (;;)
  {
    if (D1 * sigmam(1. / k1, 1) < 1.)
      break;
    k1 *= 0.5;
  }
  for (;;)
  {
    if (D1 * sigmam(1. / k2, 1) > 1.)
      break;
    k2 *= 2.;
  }

  for (;;)
  {
    k = 0.5 * (k1 + k2);
    if (D1 * sigmam(1. / k, 1) < 1.)
      k1 = k;
    else if (D1 * sigmam(1. / k, 1) > 1.)
      k2 = k;
    if (D1 * sigmam(1. / k, 1) == 1. || fabs(k2 / k1 - 1.) < eps * 0.1)
      break;
  }

  return 1. / k;
}

__host__ __device__ double GQ96_of_Einv(double a, double b)
{ /* 96-pt Gauss qaudrature integrates E^-1(x) from a to b */
  double cx = (a + b) / 2;
  double dx = (b - a) / 2;
  double q = 0;
  for (int i = 0; i < 48; i++)
  {
  #ifndef __CUDA_ARCH__
    q += W96[i] * (E_inv(cx - dx * A96[i]) + E_inv(cx + dx * A96[i]));
  #else
    q += dev_W96[i] * (E_inv(cx - dx * dev_A96[i]) + E_inv(cx + dx * dev_A96[i]));
  #endif
  };
  return (q * dx);
}

__host__ __device__ double E(double z)
{ // assuming flat universe
#ifndef __CUDA_ARCH__
  return sqrt(cosmo.om * pow(1 + z, 3) + cosmo.ow * pow(1 + z, 3 * (1.0 + cosmo.w)));
#else
  return sqrt(dev_om * pow(1 + z, 3) + dev_ow * pow(1 + z, 3 * (1.0 + dev_w)));
#endif
}

__host__ __device__ double E_inv(double z)
{
  return 1. / E(z);
}

__host__ __device__ double GQ96_of_Pk(double a, double b, double ell, double * dev_g, double * dev_p, int Ntomo)
{ /* 96-pt Gauss qaudrature integrates bdelta(x,ells) from x=a to b */
  int i;
  double cx, dx, q;
  cx = (a + b) / 2;
  dx = (b - a) / 2;
  q = 0;
  for (i = 0; i < 48; i++)
#ifndef __CUDA_ARCH__
    q += W96[i] * (limber_integrand_power_spectrum(ell, cx - dx * A96[i], 0, 0, dev_g, dev_p, Ntomo) + limber_integrand_power_spectrum(ell, cx + dx * A96[i], 0, 0, dev_g, dev_p, Ntomo));
#else
    q += dev_W96[i] * (limber_integrand_power_spectrum(ell, cx - dx * dev_A96[i], 0, 0, dev_g, dev_p, Ntomo) + limber_integrand_power_spectrum(ell, cx + dx * dev_A96[i], 0, 0, dev_g, dev_p, Ntomo));
#endif
  return (q * dx);
}

__host__ __device__ double limber_integrand_power_spectrum(double ell, double z, double zbin1, double zbin2, double * dev_g, double * dev_p, int Ntomo)
{
  if (z < 1e-5)
    return 0;
    #ifndef __CUDA_ARCH__
  double didx = z / z_max * (n_redshift_bins);
    #else
  double didx = z / dev_z_max * (dev_n_redshift_bins);
  #endif
  int idx = didx;
  didx = didx - idx;

  double r_sigma, n_eff, D1, ncur;
  compute_coefficients(idx, didx, &D1, &r_sigma, &n_eff, &ncur);

  double p_value_1 = p_interpolated(idx,didx,zbin1, dev_p, Ntomo);
  double p_value_2 = p_interpolated(idx,didx,zbin2, dev_p, Ntomo);
  double g_value_1 = g_interpolated(idx,didx,zbin1, dev_g, Ntomo);
  double g_value_2 = g_interpolated(idx,didx,zbin2, dev_g, Ntomo);
  double f_K_value = f_K_interpolated(idx,didx);

  double C1_rho_crit = 0.013873073650776856;//0.0134;
#ifdef __CUDA_ARCH__
  double A_IA = dev_A_IA;
  double f_IA = - A_IA * dev_om * C1_rho_crit / D1;
  double H_c=dev_H0_over_c;

#else
  double A_IA=cosmo.A_IA;
  double f_IA = - A_IA * cosmo.om * C1_rho_crit / D1;
  double H_c=H0_over_c;


#endif

  
  double k = ell/f_K_value;
  double P_k = P_k_nonlinear(k, z);

  double Pell_G_G = limber_integrand_prefactor_delta(z, g_value_1) * limber_integrand_prefactor_delta(z, g_value_2) * P_k / (E(z)*H_c);
  
  double Pell_G_I = limber_integrand_prefactor_delta(z, g_value_1) * p_value_2 * P_k * f_IA / f_K_value;

  double Pell_I_G = limber_integrand_prefactor_delta(z, g_value_2) * p_value_1 * P_k * f_IA / f_K_value;

  double Pell_I_I = p_value_1 * p_value_2 * P_k * f_IA * f_IA *(E(z)*H_c) / (f_K_value*f_K_value); 


  double correction=1.0;

#if T17_CORRECTION // Correction for T17 simulations
  double c1 = 9.5171e-4;
  double c2 = 5.1543e-3;
  double a1 = 1.3063;
  double a2 = 1.1475;
  double a3 = 0.62793;
  correction = pow(1 + c1 * pow(k, -1. * a1), a1) / pow(1 + c2 * pow(k, -1. * a2), a3);
#endif

  return (Pell_G_G+Pell_G_I+Pell_I_G+Pell_I_I)*correction;

}

__host__ __device__ double limber_integrand_prefactor(double z, double g_value)
{
#ifndef __CUDA_ARCH__
  return 9. / 4. * H0_over_c * H0_over_c * H0_over_c * cosmo.om * cosmo.om * (1. + z) * (1. + z) * g_value * g_value / E(z);
#else
  return 9. / 4. * dev_H0_over_c * dev_H0_over_c * dev_H0_over_c * dev_om * dev_om * (1. + z) * (1. + z) * g_value * g_value / E(z);
#endif
}

__host__ __device__ double limber_integrand_prefactor_delta(double z, double g_value)
{
#ifndef __CUDA_ARCH__
  return 3. / 2. * H0_over_c * H0_over_c * cosmo.om * (1. + z) * g_value;
#else
  return 3. / 2. * dev_H0_over_c * dev_H0_over_c * dev_om * (1. + z) *  g_value;
#endif
}