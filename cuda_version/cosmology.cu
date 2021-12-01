#include "cosmology.cuh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>

cosmology::cosmology(const std::string& fn_parameters)
{
  std::ifstream input(fn_parameters.c_str());
  if(input.fail())
    {
      std::cout<<"cosmology: Could not open "<<fn_parameters<<std::endl;
      exit(1);
    };
  std::vector<std::string> parameterNames;
  std::vector<double> parameterValues;
  
  if(input.is_open())
    {
      std::string line;
      while(std::getline(input, line))
	{
	  if(line[0]=='#' || line.empty()) continue;
	  std::string name;
	  double value;
	  std::istringstream iss(line);
	  iss>>name>>value;
	  parameterNames.push_back(name);
	  parameterValues.push_back(value);
	};
    };

  for(unsigned int i=0; i<parameterNames.size(); i++)
    {
      if(parameterNames.at(i)=="h")
	{
	  h=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="sigma_8")
	{
	  sigma8=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="Omega_b")
	{
	  omb=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="n_s")
	{
	  ns=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="w")
	{
	  w=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="Omega_m")
	{
	  om=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="z_max")
	{
	  zmax=parameterValues.at(i);
	}
      else
	{
	  std::cout<<"Cosmology::Parameter file is not in the right format"
		   <<std::endl;
	  return;
	};
    };

  ow=1-om;
  omc=om-omb;
}

std::ostream& operator<<(std::ostream& out, const cosmology& cosmo)
{
  out<<"h:"<<cosmo.h<<std::endl;
  out<<"sigma_8:"<<cosmo.sigma8<<std::endl;
  out<<"Omega_b:"<<cosmo.omb<<std::endl;
  out<<"n_s:"<<cosmo.ns<<std::endl;
  out<<"w:"<<cosmo.w<<std::endl;
  out<<"Omega_m:"<<cosmo.om<<std::endl;
  out<<"z_max:"<<cosmo.zmax<<std::endl;
  return out;
}



void read_n_of_z(const std::string& fn, const double& dz, const int& n_bins, std::vector<double>& nz)
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
  norm*=dz;
  for(int i=0; i<n_bins; i++)
    {
      nz.at(i)/=norm;
    };
  
}


void read_thetas(const std::string& fn, std::vector<double>& thetas)
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
	  double theta;
	  std::istringstream iss(line);
	  iss>>theta;
	  thetas.push_back(theta);
	};
    };
}

void read_covariance_param(const std::string& fn, covarianceParameters& covPar)
{
  
  // Open file
  std::ifstream input(fn.c_str());
  if(input.fail())
    {
      std::cout<<"read_covariance_param: Could not open "<<fn<<std::endl;
      return;
    };
  
  
  // Read in file
  std::vector<std::string> parameterNames;
  std::vector<double> parameterValues;
  
  if(input.is_open())
    {
      std::string line;
      while(std::getline(input, line))
	{
	  if(line[0]=='#' || line.empty()) continue;
	  std::string name;
	  double value;
	  std::istringstream iss(line);
	  iss>>name>>value;
	  parameterNames.push_back(name);
	  parameterValues.push_back(value);
	};
  };

  for(unsigned int i=0; i<parameterNames.size(); i++)
  {
    if(parameterNames.at(i)=="survey_area")
    {
      covPar.survey_area=parameterValues.at(i);
    }
    else if(parameterNames.at(i)=="shapenoise_sigma")
    {
      covPar.shapenoise_sigma=parameterValues.at(i);
    }
    else if(parameterNames.at(i)=="galaxy_density")
    {
      covPar.galaxy_density=parameterValues.at(i);
    }
    else
    {
      std::cout<<"Cosmology::Parameter file is not in the right format"
        <<std::endl;
      return;
    };
  };
  // double survey_area_radsq = covPar.survey_area*pow(M_PI/180.,2);
  // double galaxy_number_density = covPar.galaxy_density;
  double power_spectrum_term = pow(covPar.shapenoise_sigma,2)/2./covPar.galaxy_density;
  covPar.power_spectrum_contribution = power_spectrum_term;
}

std::ostream& operator<<(std::ostream& out, const covarianceParameters& covPar)
{
  out<<"survey area [deg^2]: "<<covPar.survey_area<<std::endl;
  // out<<"number of galaxies: "<<covPar.n_gal<<std::endl;
  out<<"galaxy density [arcmin^-2]: "<<covPar.galaxy_density/pow(180*60./M_PI,2)<<std::endl;
  out<<"shapenoise: "<<covPar.shapenoise_sigma<<std::endl;
  out<<"power spectrum: "<<covPar.power_spectrum_contribution<<std::endl;
  return out;
}
