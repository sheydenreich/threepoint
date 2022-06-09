#include "apertureStatisticsCovariance.cuh"
#include "helpers.cuh"

#include <iostream>
#include <vector>
#include <chrono>
#include <string>

int main()
{
    double thetaMax=1.87;
    thetaMax*=M_PI/180;
    double A= thetaMax*thetaMax;
    std::string outfn="T4_testBispec_analytical.dat";

    std::vector<double> thetas{2,4,8,16};
    int N=thetas.size();
    std::vector<std::vector<double>> theta_combis;
    for (int i=0; i<N; i++)
    {
      double theta1 = convert_angle_to_rad(thetas.at(i)); // Conversion to rad
      for (int j = i; j < N; j++)
      {
        double theta2 = convert_angle_to_rad(thetas.at(j));
        for (int k = j; k < N; k++)
        {
          double theta3 = convert_angle_to_rad(thetas.at(k));
          std::vector<double> thetas_123 = {theta1, theta2, theta3};
  
          theta_combis.push_back(thetas_123);
        }
      }
    }

    int N_ind = theta_combis.size(); // Number of independent theta-combinations
    int N_total = N_ind * (N_ind+1) / 2;
  

    int completed_steps = 0;

    auto begin = std::chrono::high_resolution_clock::now(); // Begin time measurement
    for (int i = 0; i < N_ind; i++)
    {
      for (int j=0; j<N_ind; j++)
      {
      //  std::cerr<<i<<" "<<j<<std::endl;
        try
        {
          double res=T4_testBispec_analytical(theta_combis.at(i).at(0), theta_combis.at(i).at(1), theta_combis.at(i).at(2), theta_combis.at(j).at(0), theta_combis.at(j).at(1), theta_combis.at(j).at(2))/A;
          std::cout<<res<<std::endl;
        }
        catch (const std::exception &e)
        {
          std::cerr << e.what() << '\n';
          return -1;
        }



              // Progress for the impatient user
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
      completed_steps++;
      // double progress = (completed_steps * 1.) / (N_total);
      //   fprintf(stderr, "\r [%3d%%] in %.2f h. Est. remaining: %.2f h. Average: %.2f s per step. Last thetas: (%.2f, %.2f, %.2f, %.2f, %.2f, %.2f) [%s]",
      //   static_cast<int>(progress * 100),
      //   elapsed.count() * 1e-9 / 3600,
      //   (N_total - completed_steps) * elapsed.count() * 1e-9 / 3600 / completed_steps,
      //   elapsed.count() * 1e-9 / completed_steps,
      //   convert_rad_to_angle(theta_combis.at(i).at(0)), convert_rad_to_angle(theta_combis.at(i).at(1)), convert_rad_to_angle(theta_combis.at(i).at(2)),
      //   convert_rad_to_angle(theta_combis.at(j).at(0)), convert_rad_to_angle(theta_combis.at(j).at(1)), convert_rad_to_angle(theta_combis.at(j).at(2)), "arcmin");
      }
      
    };

    return 0;
}