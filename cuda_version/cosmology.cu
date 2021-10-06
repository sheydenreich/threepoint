#include "cosmology.cuh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

cosmology::cosmology(const std::string& fn_parameters)
{
  std::ifstream input(fn_parameters.c_str());
  if(input.fail())
    {
      std::cout<<"cosmology: Could not open "<<fn_parameters<<std::endl;
      return;
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
