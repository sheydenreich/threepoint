#include "gamma.hpp"
#include "helper.hpp"
#include <omp.h>


// gsl_complex test_function(double x, void* params)
// {
//     gsl_complex z,xc;
//     GSL_SET_COMPLEX(&z,1,1);
//     GSL_SET_COMPLEX(&xc,xc,0);
//     return gsl_complex_add(z,xc);
// }


int main()
{
    
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
    
  BispectrumCalculator test_class(cosmo,200,2,false);
  // std::cout << test_class.bkappa(1000.,1000.,1000.)*pow(1000.,3) << "\n";
  // std::cout << test_class.bkappa(2.055e+03, 5.472e+02, 2.602e+03)*pow(1000.,3) << "\n";

  /*
    ###############################################################################################
    ######## THIS COMPUTES 3pcf for all triangles #################################################
    ###############################################################################################
  */
  //     int steps = 10;
  //     int usteps = 10;
  //     int vsteps = 11;

  //     std::string infile;
  //     // Reading the triangle configurations
  //     if(slics)
  //     {
  //         infile = "../necessary_files/triangles_slics.dat";
  //     }
  //     else
  //     {
  //         infile = "../necessary_files/triangles_millennium.dat";
  //     }

  //     GammaCalculator class_gamma(cosmo, 0.2, 3.5, false, 200, 2);

  //     treecorr_bin* triangle_configurations;
  //     triangle_configurations = new treecorr_bin [steps*usteps*vsteps];
  //     read_triangle_configurations(infile, triangle_configurations, steps, usteps, vsteps);

  //     std::complex<double> result[steps*usteps*vsteps]; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
  //     printf("Computing 3pcf. This might take a day or so ...\n");

  //     #pragma omp parallel for collapse(3)
  //     for(int i=0;i<steps;i++){
  //         for(int j=0;j<usteps;j++){
  //         	for(int k=0;k<vsteps;k++){
  // //                 printf("\b\b\b\b\b\b[%3d%%]",static_cast<int>(100*i*j*k/(steps*usteps*vsteps)));
  //         		fflush(stdout);

  //     //                  printf("%3f\n",i*100./steps);
  //                     // TODO: Do the triangle-averaging-steps in u,v,r instead of r1,r2,r3
  //                     double r,u,v;
  //                     int id_x = i*usteps*vsteps+j*vsteps+k;
  //                     r = triangle_configurations[id_x].r;
  //                     u = triangle_configurations[id_x].u;
  //                     v = triangle_configurations[id_x].v;


  //                     double r2 = r; //THIS IS THE BINNING BY JARVIS. FROM THE WEBSITE, NOT THE PAPER.
  //                     double r3 = r2*u;
  //                     double r1 = v*r3+r2;
  //                     // std::complex<double> res_temp;
  //                     std::complex<double> res_temp = class_gamma.gamma0(M_PI/180./60.*r1, M_PI/180./60.*r2, M_PI/180./60.*r3);
  //                     if(isnan(real(res_temp)) || isnan(imag(res_temp))){
  //                         printf("%lf %lf %lf %lf %lf \n",r1,r2,r3,real(res_temp),imag(res_temp));
  //                         res_temp = (0,0);
  //                     }
  //                     result[id_x] = res_temp;
  //              }

  //                  // printf("%lf %lf %lf \n",r1,r2,r3);
  //                  // double i = 30.;
  //                  // std::complex<double>result(0,0);
  //                  // printf("[%e, %e, %e, %e, %e], \n",r_array[i],u_array[j],v_array[k],real(result),imag(result));
  //          }
  //      }

  //      printf("[100%%]...Done. Writing results ...");
  //      fflush(stdout);

  //      FILE *fp;
  //      if(slics)
  //      {
  //          fp = fopen("../results/results_slics.dat","w");
  //      }
  //      else
  //      {
  //          fp = fopen("../results/results_millennium_short_classed.dat","w");
  //      }
  //      for(int i=0;i<steps;i++){
  //          for(int j=0;j<usteps;j++){
  //              for(int k=0;k<vsteps;k++){
  //                  int id_x;
  //                  id_x = i*usteps*vsteps+j*vsteps+k;

  //                  fprintf(fp,"%d %d %d %e %e \n",i ,j ,k ,real(result[id_x]),imag(result[id_x]));
  //              }
  //          }
  //      }
  //      fclose(fp);
  //      printf("Done.\n");


  // ###############################################################################################
  // ######## THIS IS FOR TESTING THE KAPPA - GAMMA INTEGRATION ####################################
  // ###############################################################################################
  int steps = 10;
  int usteps = 10;
  int vsteps = 11;
  double r_array[steps] = {0.65763, 1.1376, 1.968, 3.4044, 5.8893, 10.188, 17.624, 30.488, 52.741, 91.237};
  double u_array[usteps] = {0.05  ,  0.15 ,   0.25  ,  0.39229 , 0.45  ,  0.55  ,  0.63257,  0.79057,  0.89457,  0.95};
  double v_array[vsteps] = {0.045455 ,  0.13636 ,  0.22727 ,  0.31818 ,  0.40909  , 0.5     ,  0.59091 , 0.68182  , 0.77273 ,  0.86364 , 0.95455};


  GammaCalculator class_gamma(cosmo, 0.05, 3.5, false, 200, 2);


  for(int i=0;i<steps;i++){
    for(int j=0;j<usteps;j++){
      //            	 printf("[%3d%%]\n",static_cast<int>(10*i+j));
      for(int k=0;k<vsteps;k++){
	//                 printf("\b\b\b\b\b\b[%3d%%]",static_cast<int>(100*i*j*k/(steps*usteps*vsteps)));

	//                  printf("%3f\n",i*100./steps);
	// TODO: Do the triangle-averaging-steps in u,v,r instead of r1,r2,r3
	double r,u,v;

	r = r_array[i]; //THIS IS THE BINNING BY JARVIS. FROM THE WEBSITE, NOT THE PAPER.
	u = u_array[j];
	v = v_array[k];

	double r2 = r*M_PI/(60*180.); //THIS IS THE BINNING BY JARVIS. FROM THE WEBSITE, NOT THE PAPER.
	double r3 = r2*u;
	double r1 = v*r3+r2;

	std::complex<double> res_temp = class_gamma.gamma0(r1, r2, r3);
	if(isnan(real(res_temp)) || isnan(imag(res_temp))){
	  printf("%lf %lf %lf %lf %lf \n",r1,r2,r3,real(res_temp),imag(res_temp));
	  res_temp = std::complex<double>(0,0);
	}
	std::complex<double> X(r1,0);
	double height = (r1+r2+r3)*(r2+r3-r1)*(r1-r2+r3)*(r1+r2-r3);
	height = sqrt(height)/(2.*r1);
	double rest_of_r1 = sqrt(r2*r2-height*height);
	std::complex<double> Y(rest_of_r1,height);
	std::complex<double> comp_temp;
	comp_temp = class_gamma.ggg(Y,X);

	printf("[%d, %d, %d, %.4e, %.4e, %.4e, %.4e],",i,j,k,
	       real(res_temp),imag(res_temp),real(comp_temp),imag(comp_temp));

      }
    }
  }
  printf("\n Done. \n");
}
