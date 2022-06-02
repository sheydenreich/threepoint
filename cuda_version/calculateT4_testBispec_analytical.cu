#include "apertureStatisticsCovariance.cuh"

#include <iostream>

int main()
{
    double thetaMax=1.87;
    thetaMax*=M_PI/180;
    double A= thetaMax*thetaMax;
    outfn="T4_testBispec_analytical.dat"

    std::vector<double> thetas([2,4,8,16]);

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

    for (int i = 0; i < N_ind; i++)
    {
      for (int j=i; j<N_ind; j++)
      {
        double res=T4_testBispec_analytical(theta_combis.at(i), theta_combis.at(j));
        std::cout<<res<<std::endl;
      }
    };

    return 0;
}