#include "helper.hpp"
#include <sstream>
#include <numeric>

int read_triangle_configurations(std::string& infile, std::vector<treecorr_bin>& triangle_config)
{
  std::ifstream fin(infile);
  std::cout << "Reading " << infile << "\n";
  if(!fin.is_open()) //checking if file can be opened
    {
      std::cerr << "Could not open input file. Exiting. \n";
      exit(1);
    }
  while(!fin.eof())
    {
      int rind,uind,vind;
      double r,u,v;
      fin >> rind >> uind >> vind >> r >> u >> v;
      treecorr_bin bin;
      bin.r=r;
      bin.u=u;
      bin.v=v;
      triangle_config.push_back(bin);
    }

  fin.close();

  return 1;

}

void read_n_of_z(const std::string& fn, const int& n_bins, const double& zMax, std::vector<double>& nz)
{
    //Open file
    std::ifstream input(fn.c_str());
    if(input.fail())
    {
        std::cout<<"read_n_of_z: Could not open "<<fn<<std::endl;
        exit(1);
    };
    std::vector<double> zs;
    std::vector<double> n_of_zs;

    // Read in file
    if(input.is_open())
    {
        std::string line;
        while(std::getline(input, line))
        {
	        if(line[0]=='#' || line.empty()) continue;
	        double z, n_of_z;
	        std::istringstream iss(line);
	        iss>>z>>n_of_z;
	        zs.push_back(z);
	        n_of_zs.push_back(n_of_z);
	    };
    };


    // Casting in our used bins
    int n_bins_file=zs.size(); //Number of z bins in file
    double zmin_file=zs.front(); //Minimal z in file
    double zmax_file=zs.back(); //Maximal z in file

    double dz_file=(zmax_file-zmin_file)/(n_bins_file-1);
    double dz=zMax/n_bins;
    for(int i=0; i<n_bins; i++)
    {
        double z=i*dz;
        int ix_file=int((z-zmin_file)/dz_file);
        double dix_file=(z-zmin_file)/dz_file - ix_file;
        double n_of_z=0;

     
        if(ix_file>=0 && ix_file<n_bins_file-1)//Interpolate between closest bins
	    {
	        n_of_z=n_of_zs.at(ix_file+1)*dix_file+n_of_zs.at(ix_file)*(1-dix_file); 
	    };
        if(ix_file==n_bins_file-1)//If at end of vector, don't interpolate
	    {
	        n_of_z=n_of_zs.at(n_bins_file-1);
	    };
        nz.push_back(n_of_z);
    }

    // Normalization
    double norm=std::accumulate(nz.begin(), nz.end(), 0.0);
    if(norm==0)
    {
        std::cerr<<"sum of n(z) is zero! Check "<<fn<<"! Exiting."<<std::endl;
        exit(1);
    };
    norm*=dz;
    for(int i=0; i<n_bins; i++)
    {
       nz.at(i)/=norm;
    };
}

void read_measurement(const std::string& fn, std::vector<std::vector<double>>& thetas, 
                      std::vector<double>& measurements, const std::string& unit)

{
    // Open file
    std::ifstream input(fn.c_str());
    if(input.fail())
    {
        std::cout<<"read_thetas: Could not open "<<fn<<std::endl;
        return;
    };

  
    // Read in file
    if(input.is_open())
    {
        std::string line;
        while(std::getline(input, line))
	    {
	        if(line[0]=='#' || line.empty()) continue;
	        double theta1, theta2, theta3;
            double Map3;
	        std::istringstream iss(line);
	        iss>>theta1>>theta2>>theta3>>Map3;
            theta1=convert_angle_to_rad(theta1, unit);
            theta2=convert_angle_to_rad(theta2, unit);
            theta3=convert_angle_to_rad(theta3, unit);
            std::vector<double> theta={theta1, theta2, theta3};
	        thetas.push_back(theta);
            measurements.push_back(Map3);
	    };
    };
}


double convert_angle_to_rad(const double& value, const std::string& unit)
{
    //Conversion factor
    double conversion;

    if(unit=="arcmin")
    {
        conversion = 2.9088820866e-4;
    }
    else if(unit=="deg")
    {
        conversion = 0.017453;
    }
    else if(unit=="rad")
    {
        conversion = 1;
    }
    else
    {
        std::cerr<<"Unit not correctly specified. Needs to be arcmin, deg, or rad. Exiting.";
        exit(1);
    };
    return conversion*value;
}

double convert_rad_to_angle(const double& value, const std::string& unit)
{
    //Conversion factor
    double conversion;

    if(unit=="arcmin")
    {
        conversion = 3437.74677;
    }
    else if(unit=="deg")
    {
        conversion = 57.3;
    }
    else
    {
        std::cerr<<"Unit not correctly specified. Needs to be arcmin or deg. Exiting.";
        exit(1);
    };
    return conversion*value;
}