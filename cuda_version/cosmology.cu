#include "cosmology.cuh"
#include "helpers.cuh"

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
  std::vector<std::string> parameterValues;
  
  if(input.is_open())
    {
      std::string line;
      while(std::getline(input, line))
	{
	  if(line[0]=='#' || line.empty()) continue;
	  std::string name;
	  std::string value;
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
	  h=std::stod(parameterValues.at(i));
	}
      else if(parameterNames.at(i)=="sigma_8")
	{
	  sigma8=std::stod(parameterValues.at(i));
	}
      else if(parameterNames.at(i)=="Omega_b")
	{
	  omb=std::stod(parameterValues.at(i));
	}
      else if(parameterNames.at(i)=="n_s")
	{
	  ns=std::stod(parameterValues.at(i));
	}
      else if(parameterNames.at(i)=="w")
	{
	  w=std::stod(parameterValues.at(i));
	}
      else if(parameterNames.at(i)=="Omega_m")
	{
	  om=std::stod(parameterValues.at(i));
	}
      else if(parameterNames.at(i)=="z_max")
	{
	  zmax=std::stod(parameterValues.at(i));
	}
      else if(parameterNames.at(i)=="baryons")
  {
    std::istringstream(parameterValues.at(i)) >> baryons;
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
  out<<"h "<<cosmo.h<<std::endl;
  out<<"sigma_8 "<<cosmo.sigma8<<std::endl;
  out<<"Omega_b "<<cosmo.omb<<std::endl;
  out<<"n_s "<<cosmo.ns<<std::endl;
  out<<"w "<<cosmo.w<<std::endl;
  out<<"Omega_m "<<cosmo.om<<std::endl;
  out<<"z_max "<<cosmo.zmax<<std::endl;
  if(cosmo.baryons)
  {
    out << "Computing bispectrum WITH baryons" << std::endl;
  }
  else
  {
    out << "Computing bispectrum WITHOUT baryons" << std::endl;
  }
  return out;
}



covarianceParameters::covarianceParameters(const std::string& fn)
{
  
  // Open file
  std::ifstream input(fn.c_str());
  if(input.fail())
    {
      std::cout<<"covariance_param: Could not open "<<fn<<std::endl;
      return;
    };
  
  
  // Read in file
  std::vector<std::string> parameterNames;
  std::vector<std::string> parameterValues;
  
  if(input.is_open())
    {
      std::string line;
      while(std::getline(input, line))
	{
	  if(line[0]=='#' || line.empty()) continue;
	  std::string name;
	  std::string value;
	  std::istringstream iss(line);
	  iss>>name>>value;
	  parameterNames.push_back(name);
	  parameterValues.push_back(value);
	};
  };
  std::string unit;

  for(unsigned int i=0; i<parameterNames.size(); i++)
  {
    if(parameterNames.at(i)=="thetaMax")
    {
      thetaMax=std::stod(parameterValues.at(i));
    }
    else if(parameterNames.at(i)=="shapenoise_sigma")
    {
      shapenoise_sigma=std::stod(parameterValues.at(i));
    }
    else if(parameterNames.at(i)=="galaxy_density")
    {
      galaxy_density=std::stod(parameterValues.at(i));
    }
    else if(parameterNames.at(i)=="unit")
    {
      unit=parameterValues.at(i);
    }
    else if (parameterNames.at(i)=="shapenoiseOnly")
    {
      shapenoiseOnly=std::stoi(parameterValues.at(i));
    }
    else
    {
      std::cout<<"covarianceParameters::Parameter file is not in the right format"
        <<std::endl;
      return;
    };
  };

  // Convert thetaMax and galaxy density to radians
  thetaMax=convert_angle_to_rad(thetaMax, unit);
  galaxy_density=galaxy_density/ convert_angle_to_rad(1, unit) / convert_angle_to_rad(1, unit);
}

std::ostream& operator<<(std::ostream& out, const covarianceParameters& covPar)
{
  out<<"# thetaMax and galaxy density in rad and rad^-2"<<std::endl;
  out<<"thetaMax "<<covPar.thetaMax<<std::endl;
  out<<"galaxy density "<<covPar.galaxy_density<<std::endl;
  out<<"shapenoise "<<covPar.shapenoise_sigma<<std::endl;
  return out;
}

void read_gamma_config(const std::string& fn, configGamma& config)
{

  // Open file
  std::ifstream input(fn.c_str());
  if(input.fail())
    {
      std::cout<<"read_gamma_config: Could not open "<<fn<<std::endl;
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
      }
    }

  for(unsigned int i=0; i<parameterNames.size(); i++)
  {
    if(parameterNames.at(i)=="rmin")
    {
      config.rmin=parameterValues.at(i);
    }
    else if(parameterNames.at(i)=="rmax")
    {
      config.rmax=parameterValues.at(i);
    }
    else if(parameterNames.at(i)=="umin")
    {
      config.umin=parameterValues.at(i);
    }
    else if(parameterNames.at(i)=="umax")
    {
      config.umax=parameterValues.at(i);
    }
    else if(parameterNames.at(i)=="vmin")
    {
      config.vmin=parameterValues.at(i);
    }
    else if(parameterNames.at(i)=="vmax")
    {
      config.vmax=parameterValues.at(i);
    }
    else if(parameterNames.at(i)=="rsteps")
    {
      config.rsteps=std::round(parameterValues.at(i));
    }
    else if(parameterNames.at(i)=="usteps")
    {
      config.usteps=std::round(parameterValues.at(i));
    }
    else if(parameterNames.at(i)=="vsteps")
    {
      config.vsteps=std::round(parameterValues.at(i));
    }
    else
    {
      std::cout<<"Cosmology::Parameter file is not in the right format"
        <<std::endl;
      return;
    }
  }
}

std::ostream& operator<<(std::ostream& out, const configGamma& config)
{
  out<<"Binning r in "<<config.rsteps << " bins from " << config.rmin << " to " << config.rmax <<std::endl;
  out<<"Binning u in "<<config.usteps << " bins from " << config.umin << " to " << config.umax <<std::endl;
  out<<"Binning v in "<<config.vsteps << " bins from " << config.vmin << " to " << config.vmax <<std::endl;
  return out;
}
