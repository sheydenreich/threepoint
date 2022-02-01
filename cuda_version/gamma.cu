#include "gamma.cuh"
#include "bispectrum.cuh"
#include "cuda_helpers.cuh"



__constant__ double dev_array_psi[prec_k];
__constant__ double dev_array_product[prec_k];
__constant__ double dev_array_psi_J2[prec_k];
__constant__ double dev_array_product_J2[prec_k];
__constant__ int dev_prec_k;

void compute_weights_bessel()
{
  printf("Warning!, AtomicAdd disabled right now until CUDA is updated!");
  // Compute weights
  // Allocate memory for bessel integration on CPU
    double bessel_zeros[prec_k];
    double pi_bessel_zeros[prec_k];
    double array_psi[prec_k];
    double array_bessel[prec_k];
    double array_psip[prec_k];
    double array_w[prec_k];
    double array_product[prec_k];
    double array_psi_J2[prec_k];
    double array_product_J2[prec_k];

    // Compute the weights
    for(int i=0;i<prec_k;i++)
      {
        bessel_zeros[i] = gsl_sf_bessel_zero_Jnu(6,i);
        pi_bessel_zeros[i] = bessel_zeros[i]/M_PI;
        array_psi[i] = M_PI*psi(pi_bessel_zeros[i]*prec_h)/prec_h;
        array_bessel[i] = gsl_sf_bessel_Jn(6,array_psi[i]);
        array_psip[i] = psip(prec_h*pi_bessel_zeros[i]);
        array_w[i] = 2/(M_PI*bessel_zeros[i]*pow(gsl_sf_bessel_Jn(7,bessel_zeros[i]),2));
        array_product[i] = array_w[i]*pow(array_psi[i],3)*array_bessel[i]*array_psip[i];
      };

    for(int i=0;i<prec_k;i++)
      {
        bessel_zeros[i] = gsl_sf_bessel_zero_Jnu(2,i);
        pi_bessel_zeros[i] = bessel_zeros[i]/M_PI;
        array_psi_J2[i] = M_PI*psi(pi_bessel_zeros[i]*prec_h)/prec_h;
        array_bessel[i] = gsl_sf_bessel_Jn(2,array_psi_J2[i]);
        array_psip[i] = psip(prec_h*pi_bessel_zeros[i]);
        array_w[i] = 2/(M_PI*bessel_zeros[i]*pow(gsl_sf_bessel_Jn(3,bessel_zeros[i]),2));
        array_product_J2[i] = array_w[i]*pow(array_psi_J2[i],3)*array_bessel[i]*array_psip[i];
      };

    // Copy the weights to the GPU
    CudaSafeCall(cudaMemcpyToSymbol(dev_array_psi,array_psi,prec_k*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(dev_array_product,array_product,prec_k*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(dev_array_psi_J2,array_psi_J2,prec_k*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(dev_array_product_J2,array_product_J2,prec_k*sizeof(double)));
    CudaSafeCall(cudaMemcpyToSymbol(dev_prec_k,&prec_k,sizeof(int)));
    
}

double inline psi(double t)
{
    return t*tanh(M_PI*sinh(t)/2);
}

double inline psip(double t)
{
    double zahler = sinh(M_PI*sinh(t))+M_PI*t*cosh(t);
    double nenner = cosh(M_PI*sinh(t))+1;
    return zahler/nenner;
}

/* Rotating from orthocenter to centroid */

double one_rotation_angle_otc(double x1, double x2, double x3)
{
    double psi3 = interior_angle(x1,x2,x3);
    double h3 = height_of_triangle(x1,x2,x3);
    double cos_2_angle = pow(pow(x2,2)-pow(x1,2),2)-4*pow(x1*x2*sin(psi3),2);
    cos_2_angle /= pow(2*h3*x3,2);
    double sin_2_angle = (pow(x2,2)-pow(x1,2))*x1*x2*sin(psi3);
    sin_2_angle /= pow(h3*x3,2);
    return atan2(sin_2_angle,cos_2_angle)/2;
}

std::complex<double> convert_orthocenter_to_centroid(std::complex<double> gamma, double x1, double x2, double x3, bool conjugate_phi1)
{
    double angle1 = one_rotation_angle_otc(x2, x3, x1);
    double angle2 = one_rotation_angle_otc(x3, x1, x2);
    double angle3 = one_rotation_angle_otc(x1, x2, x3);
    if(conjugate_phi1) angle1 *= -1.;
    return gamma*exp(std::complex<double>(0,-2.)*(angle1+angle2+angle3));
}

double height_of_triangle(double x1, double x2, double x3)
{
    return 0.5*sqrt(2*pow(x1,2)+2*pow(x2,2)-pow(x3,2));
}




std::complex<double> gamma0(double x1, double x2, double x3, double z_max)
{
  double vals_min[3] = {0,0,0};
  double vals_max[3] = {z_max,2*M_PI,M_PI/2};
  double result[2];
  double error[2];
  struct GammaCudaContainer params;
  std::complex<double> complexResult;
  params.x1 = x1;
  params.x2 = x2;
  params.x3 = x3;
  double epsabs = 0;
  hcubature_v(2,integrand_gamma0,&params,3,vals_min,vals_max,0,epsabs,1e-3,ERROR_L1,result,error);
  complexResult = std::complex<double>(result[0],result[1]);
  #ifdef CONVERT_TO_CENTROID 
  complexResult = convert_orthocenter_to_centroid(complexResult,x1,x2,x3,false);
  #endif //CONVERT_TO_CENTROID
  double om = cosmo.om;
  return complexResult*27./8.*pow(om,3)*pow(100./299792.,5)/3./(2*pow(2*M_PI,3));
}

std::complex<double> gamma1(double x1, double x2, double x3, double z_max)
{
  double vals_min[3] = {0,0,0};
  double vals_max[3] = {z_max,2*M_PI,M_PI/2};
  double result[2];
  double error[2];
  struct GammaCudaContainer params;
  std::complex<double> complexResult;
  params.x1 = x1;
  params.x2 = x2;
  params.x3 = x3;
  double epsabs = 0;
  hcubature_v(2,integrand_gamma1,&params,3,vals_min,vals_max,0,epsabs,1e-3,ERROR_L1,result,error);
  complexResult = std::complex<double>(result[0],result[1]);
  #ifdef CONVERT_TO_CENTROID 
  complexResult = convert_orthocenter_to_centroid(complexResult,x1,x2,x3,true);
  #endif //CONVERT_TO_CENTROID
  double om = cosmo.om;
  return complexResult*27./8.*pow(om,3)*pow(100./299792.,5)/3./(2*pow(2*M_PI,3));
}

std::complex<double> gamma2(double x1, double x2, double x3, double z_max)
{
  return gamma1(x2,x3,x1,z_max);
}

std::complex<double> gamma3(double x1, double x2, double x3, double z_max)
{
  return gamma1(x3,x1,x2,z_max);
}




int integrand_gamma0(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value)
{
    struct GammaCudaContainer params = *((GammaCudaContainer*) fdata);

    // std::cout << npts << std::endl;

    double x1 = params.x1;
    double x2 = params.x2;
    double x3 = params.x3;

    double* d_vars;
    double* d_value;
    
    // allocate memory
    CudaSafeCall(cudaMalloc(&d_value,fdim*npts*sizeof(double)));
    CudaSafeCall(cudaMalloc(&d_vars,ndim*npts*sizeof(double)));


    // copy the parameters
    CudaSafeCall(cudaMemcpy(d_vars,vars,ndim*npts*sizeof(double),cudaMemcpyHostToDevice));

    #ifdef PERFORM_SUM_REDUCTION
      dim3 threadsPerBlock(prec_k-1,THREADS/(prec_k-1));
      dim3 blocksPerGrid(1,BLOCKS);

    #else

      // set array to zero
      CudaSafeCall(cudaMemset(d_value,0,fdim*npts*sizeof(double)));

      int threadsPerBlock_k = 4;
      dim3 threadsPerBlock(threadsPerBlock_k,THREADS/threadsPerBlock_k);

      int blocksPerGrid_k = static_cast<int>(ceil(prec_k*1./threadsPerBlock_k));
      dim3 blocksPerGrid(blocksPerGrid_k,BLOCKS/blocksPerGrid_k);

    #endif //PERFORM_SUM_REDUCTION

    #ifdef DEBUG_OUTPUT
    std::cout << "Computing Gamma0 at (" << x1 << "," << x2 << "," << x3 
    << ") with " << npts << "evaluations, (" << blocksPerGrid.x << "," << blocksPerGrid.y 
    << ") = " << blocksPerGrid.x*blocksPerGrid.y << " blocks and (" << threadsPerBlock.x 
    << "," << threadsPerBlock.y << ") = " << threadsPerBlock.x*threadsPerBlock.y << " threads." 
    << std::endl;
    #endif // DEBUG_OUTPUT

    compute_integrand_gamma0<<<blocksPerGrid,threadsPerBlock>>>(d_vars,d_value,npts,x1,x2,x3);

    CudaCheckError();


    // copy the result
    CudaSafeCall(cudaMemcpy(value,d_value,fdim*npts*sizeof(double),cudaMemcpyDeviceToHost));

    // free allocated memory
    CudaSafeCall(cudaFree(d_vars));
    CudaSafeCall(cudaFree(d_value));

    return 0;
}

int integrand_gamma1(unsigned ndim, size_t npts, const double* vars, void* fdata, unsigned fdim, double* value)
{
    struct GammaCudaContainer params = *((GammaCudaContainer*) fdata);

    // std::cout << npts << std::endl;

    double x1 = params.x1;
    double x2 = params.x2;
    double x3 = params.x3;

    double* d_vars;
    double* d_value;
    
    // allocate memory
    CudaSafeCall(cudaMalloc(&d_value,fdim*npts*sizeof(double)));
    CudaSafeCall(cudaMalloc(&d_vars,ndim*npts*sizeof(double)));


    // copy the parameters
    CudaSafeCall(cudaMemcpy(d_vars,vars,ndim*npts*sizeof(double),cudaMemcpyHostToDevice));

    #ifdef PERFORM_SUM_REDUCTION
      dim3 threadsPerBlock(prec_k-1,THREADS/(prec_k-1));
      dim3 blocksPerGrid(1,BLOCKS);

    #else

      // set array to zero
      CudaSafeCall(cudaMemset(d_value,0,fdim*npts*sizeof(double)));

      int threadsPerBlock_k = 4;
      dim3 threadsPerBlock(threadsPerBlock_k,THREADS/threadsPerBlock_k);

      int blocksPerGrid_k = static_cast<int>(ceil(prec_k*1./threadsPerBlock_k));
      dim3 blocksPerGrid(blocksPerGrid_k,BLOCKS/blocksPerGrid_k);

    #endif //PERFORM_SUM_REDUCTION

    #ifdef DEBUG_OUTPUT
    std::cout << "Computing Gamma1 at (" << x1 << "," << x2 << "," << x3 
    << ") with " << npts << "evaluations, (" << blocksPerGrid.x << "," << blocksPerGrid.y 
    << ") = " << blocksPerGrid.x*blocksPerGrid.y << " blocks and (" << threadsPerBlock.x 
    << "," << threadsPerBlock.y << ") = " << threadsPerBlock.x*threadsPerBlock.y << " threads." 
    << std::endl;
    #endif // DEBUG_OUTPUT

    compute_integrand_gamma1<<<blocksPerGrid,threadsPerBlock>>>(d_vars,d_value,npts,x1,x2,x3);

    CudaCheckError();


    // copy the result
    CudaSafeCall(cudaMemcpy(value,d_value,fdim*npts*sizeof(double),cudaMemcpyDeviceToHost));

    // free allocated memory
    CudaSafeCall(cudaFree(d_vars));
    CudaSafeCall(cudaFree(d_value));

    return 0;
}



#ifdef PERFORM_SUM_REDUCTION

__global__ void compute_integrand_gamma0(double* dev_vars, double* dev_result_array, unsigned int npts, double x1, double x2, double x3)
{
    unsigned int idx = blockDim.y*blockIdx.y+threadIdx.y;
    unsigned int k = blockDim.x*blockIdx.x+threadIdx.x+1;
    int tidx = threadIdx.y;
    int idk = threadIdx.x; //idk = k-1
  
    // if(k==1 && idx > 70) printf("%d \n",idx);
    if(k>=dev_prec_k) return;
    if(idx>=npts) return;

    __shared__ double r[BLOCKS*(prec_k-1)];
    __shared__ double r2[BLOCKS*(prec_k-1)];

    for(int i=idx;i<npts;i+=blockDim.y*gridDim.y)
    {
      double z=dev_vars[i*3];
      double phi=dev_vars[i*3+1];
      double psi=dev_vars[i*3+2];

      cuDoubleComplex result = full_integrand_gamma0(phi,psi,z,k,x1,x2,x3);

      //perform sum reduction

      r[tidx*(dev_prec_k-1)+idk] = cuCreal(result);
      r2[tidx*(dev_prec_k-1)+idk] = cuCimag(result);
    
  
      __syncthreads();
      for (int size = (dev_prec_k-1)/2; size>0; size/=2) { //uniform
          if (idk<size)
          {
            r[tidx*(dev_prec_k-1)+idk] += r[tidx*(dev_prec_k-1)+idk+size];
            r2[tidx*(dev_prec_k-1)+idk] += r2[tidx*(dev_prec_k-1)+idk+size];
          }
          __syncthreads();
      }
      if (idk == 0)
      {
        dev_result_array[i*2] = r[tidx*(dev_prec_k-1)];
        dev_result_array[i*2+1] = r2[tidx*(dev_prec_k-1)];
      }    
    }

    return;
}

__global__ void compute_integrand_gamma1(double* dev_vars, double* dev_result_array, unsigned int npts, double x1, double x2, double x3)
{
    unsigned int idx = blockDim.y*blockIdx.y+threadIdx.y;
    unsigned int k = blockDim.x*blockIdx.x+threadIdx.x+1;
    int tidx = threadIdx.y;
    int idk = threadIdx.x; //idk = k-1
  
    // if(k==1 && idx > 70) printf("%d \n",idx);
    if(k>=dev_prec_k) return;
    if(idx>=npts) return;

    __shared__ double r[BLOCKS*(prec_k-1)];
    __shared__ double r2[BLOCKS*(prec_k-1)];

    for(int i=idx;i<npts;i+=blockDim.y*gridDim.y)
    {
      double z=dev_vars[i*3];
      double phi=dev_vars[i*3+1];
      double psi=dev_vars[i*3+2];

      cuDoubleComplex result = full_integrand_gamma1(phi,psi,z,k,x1,x2,x3);

      //perform sum reduction

      r[tidx*(dev_prec_k-1)+idk] = cuCreal(result);
      r2[tidx*(dev_prec_k-1)+idk] = cuCimag(result);
    
  
      __syncthreads();
      for (int size = (dev_prec_k-1)/2; size>0; size/=2) { //uniform
          if (idk<size)
          {
            r[tidx*(dev_prec_k-1)+idk] += r[tidx*(dev_prec_k-1)+idk+size];
            r2[tidx*(dev_prec_k-1)+idk] += r2[tidx*(dev_prec_k-1)+idk+size];
          }
          __syncthreads();
      }
      if (idk == 0)
      {
        dev_result_array[i*2] = r[tidx*(dev_prec_k-1)];
        dev_result_array[i*2+1] = r2[tidx*(dev_prec_k-1)];
      }    
    }

    return;
}



#else

__global__ void compute_integrand_gamma0(double* dev_vars, double* dev_result_array, unsigned int npts, double x1, double x2, double x3)
{
    unsigned int idx = blockDim.y*blockIdx.y+threadIdx.y;
    unsigned int k = blockDim.x*blockIdx.x+threadIdx.x+1;

    if(k>=dev_prec_k) return;
    if(idx>=npts) return;

    for(int i=idx;i<npts;i+=blockDim.y*gridDim.y)
    {
      double z=dev_vars[i*3];
      double phi=dev_vars[i*3+1];
      double psi=dev_vars[i*3+2];

      cuDoubleComplex result = full_integrand_gamma0(phi,psi,z,k,x1,x2,x3);

      
      // atomicAdd(&dev_result_array[i*2],cuCreal(result));
      // atomicAdd(&dev_result_array[i*2+1],cuCimag(result));
    }

    return;
}

__global__ void compute_integrand_gamma1(double* dev_vars, double* dev_result_array, unsigned int npts, double x1, double x2, double x3)
{
    unsigned int idx = blockDim.y*blockIdx.y+threadIdx.y;
    unsigned int k = blockDim.x*blockIdx.x+threadIdx.x+1;

    if(k>=dev_prec_k) return;
    if(idx>=npts) return;

    for(int i=idx;i<npts;i+=blockDim.y*gridDim.y)
    {
      double z=dev_vars[i*3];
      double phi=dev_vars[i*3+1];
      double psi=dev_vars[i*3+2];

      cuDoubleComplex result = full_integrand_gamma1(phi,psi,z,k,x1,x2,x3);

      
      // atomicAdd(&dev_result_array[i*2],cuCreal(result));
      // atomicAdd(&dev_result_array[i*2+1],cuCimag(result));
    }

    return;
}

#endif // PERFORM_SUM_REDUCTION


__device__ cuDoubleComplex full_integrand_gamma1(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3)
{

    double varbetabar = betabar(psi,phi);

    double varpsi1 = interior_angle(x2,x3,x1);
    double varpsi2 = interior_angle(x3,x1,x2);
    double varpsi3 = interior_angle(x1,x2,x3);

    double A1 = A(psi,x2,x3,phi,varpsi1);
    double A2 = A(psi,x3,x1,phi,varpsi2);
    double A3 = A(psi,x1,x2,phi,varpsi3);

    double alpha1 = alpha(psi, x2, x3, phi, varpsi1);
    double alpha2 = alpha(psi, x3, x1, phi, varpsi2);
    double alpha3 = alpha(psi, x1, x2, phi, varpsi3);

    double ell1,ell2,ell3;

    ell1 = dev_array_psi_J2[k]*cos(psi);
    ell2 = dev_array_psi_J2[k]*sin(psi);
    ell3 = ell1*ell1+ell2*ell2+2*ell1*ell2*cos(phi);
    if(ell3 <= 0) ell3 = 0;
    else ell3 = sqrt(ell3);

    
    cuDoubleComplex integrand_3 = exp_of_imag(varpsi1-varpsi2+2*varpsi3+2*(varbetabar-phi-alpha3));
    integrand_3 = cuCmul(integrand_3,integrand_bkappa(z, ell1/A3, ell2/A3, ell3/A3)*M_PI/pow(A3,4));

    cuDoubleComplex integrand_1 = exp_of_imag(varpsi3-varpsi2-2*(varbetabar+alpha1));
    integrand_1 = cuCmul(integrand_1,integrand_bkappa(z, ell1/A1, ell2/A1, ell3/A1)*M_PI/pow(A1,4));

    cuDoubleComplex integrand_2 = exp_of_imag(varpsi3-varpsi1-2*varpsi2+2*(varbetabar+phi-alpha2));
    integrand_2 = cuCmul(integrand_2,integrand_bkappa(z, ell1/A2, ell2/A2, ell3/A2)*M_PI/pow(A2,4));

    return cuCmul(cuCadd(integrand_1,cuCadd(integrand_2,integrand_3)),sin(2*psi)*dev_array_product_J2[k]);
}


__device__ __inline__ cuDoubleComplex full_integrand_gamma0(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3)
{
    return cuCadd(cuCadd(one_integrand_gamma0(phi,psi,z,k,x1,x2,x3),one_integrand_gamma0(phi,psi,z,k,x2,x3,x1)),one_integrand_gamma0(phi,psi,z,k,x3,x1,x2));
}


__device__ cuDoubleComplex one_integrand_gamma0(double phi, double psi, double z, unsigned int k, double x1, double x2, double x3)
{
    double varpsi = interior_angle(x1,x2,x3);
    double A3 = A(psi,x1,x2,phi,varpsi);
    cuDoubleComplex prefac = prefactor(x1,x2,x3,phi,psi);
    double ell1 = dev_array_psi[k]/A3*cos(psi);
    double ell2 = dev_array_psi[k]/A3*sin(psi);
    double ell3 = ell1*ell1+ell2*ell2+2*ell1*ell2*cos(phi);
    if(ell3 <= 0) ell3 = 0;
    else ell3 = sqrt(ell3);

    double bis = integrand_bkappa(z,ell1,ell2,ell3)*dev_array_product[k]*M_PI/pow(A3,4);

    if(isnan(cuCreal(prefac)) || isnan(cuCimag(prefac)) || isnan(bis))
      printf("%.3e, %.3e, %.3e, %d, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e \n",phi,psi,z,k,ell1,ell2,ell3,integrand_bkappa(z,ell1,ell2,ell3),dev_array_product[k],
      cuCreal(prefac),cuCimag(prefac));
    return cuCmul(prefac,bis);
}

__host__ __device__ static __inline__ cuDoubleComplex cuCmul(cuDoubleComplex x,double y)
{
    cuDoubleComplex prod;
    prod = make_cuDoubleComplex ((cuCreal(x) * y),(cuCimag(x) * y));
    return prod;
}

__device__ __host__ double interior_angle(double an1, double an2, double opp)
{
    return acos((pow(an1,2)+pow(an2,2)-pow(opp,2))/(2.0*an1*an2));
}


__device__ inline double A(double psi, double x1, double x2, double phi, double varpsi)
{
    return sqrt(pow(cos(psi)*x2,2)+pow(sin(psi)*x1,2)+sin(2*psi)*x1*x2*cos(phi+varpsi));
}


__device__ cuDoubleComplex prefactor(double x1, double x2, double x3, double phi, double psi)
{
    double varpsi = interior_angle(x1,x2,x3);
    cuDoubleComplex exponential = exp_of_imag(interior_angle(x2,x3,x1)-interior_angle(x1,x3,x2)-6*alpha(psi,x1,x2,phi,varpsi));
    cuDoubleComplex prefactor_phi = exp_of_imag(2.*betabar(psi,phi));
    double prefactor_psi = sin(2*psi);
    return cuCmul(cuCmul(exponential,prefactor_phi),prefactor_psi);
}


__host__ __device__ static __inline__ cuDoubleComplex exp_of_imag(double imag_part)
{
    cuDoubleComplex result;
    result = make_cuDoubleComplex(cos(imag_part), sin(imag_part));
    return result;
}


__device__ inline double alpha(double psi, double x1, double x2, double phi, double varpsi)
{
    double zahler = (cos(psi)*x2-sin(psi)*x1)*sin((phi+varpsi)/2);
    double nenner = (cos(psi)*x2+sin(psi)*x1)*cos((phi+varpsi)/2);
    return atan2(zahler,nenner);
}

__device__ inline double betabar(double psi, double phi)
{
    double zahler = cos(2*psi)*sin(phi);
    double nenner = cos(phi)+sin(2*psi);
    return 0.5*atan2(zahler,nenner);
}
