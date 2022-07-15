#include "helpers.cuh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <math.h>

void read_n_of_z(const std::string &fn, const int &n_bins, const double &zMax, std::vector<double> &nz)
{
    // Open file
    std::ifstream input(fn.c_str());
    if (input.fail())
    {
        std::cout << "read_n_of_z: Could not open " << fn << std::endl;
        exit(1);
    };
    std::vector<double> zs;
    std::vector<double> n_of_zs;

    // Read in file
    if (input.is_open())
    {
        std::string line;
        while (std::getline(input, line))
        {
            if (line[0] == '#' || line.empty())
                continue;
            double z, n_of_z;
            std::istringstream iss(line);
            iss >> z >> n_of_z;
            zs.push_back(z);
            n_of_zs.push_back(n_of_z);
        };
    };

    // Casting in our used bins
    int n_bins_file = zs.size();   // Number of z bins in file
    double zmin_file = zs.front(); // Minimal z in file
    double zmax_file = zs.back();  // Maximal z in file

    double dz_file = (zmax_file - zmin_file) / (n_bins_file - 1);
    double dz = zMax / n_bins;
    for (int i = 0; i < n_bins; i++)
    {
        double z = i * dz;
        int ix_file = int((z - zmin_file) / dz_file);
        double dix_file = (z - zmin_file) / dz_file - ix_file;
        double n_of_z = 0;

        if (ix_file >= 0 && ix_file < n_bins_file - 1) // Interpolate between closest bins
        {
            n_of_z = n_of_zs.at(ix_file + 1) * dix_file + n_of_zs.at(ix_file) * (1 - dix_file);
        };
        if (ix_file == n_bins_file - 1) // If at end of vector, don't interpolate
        {
            n_of_z = n_of_zs.at(n_bins_file - 1);
        };
        nz.push_back(n_of_z);
    }

    // Normalization
    double norm = std::accumulate(nz.begin(), nz.end(), 0.0);
    if (norm == 0)
    {
        std::cerr << "sum of n(z) is zero! Check " << fn << "! Exiting." << std::endl;
        exit(1);
    };
    norm *= dz;
    for (int i = 0; i < n_bins; i++)
    {
        nz.at(i) /= norm;
    };
}

double convert_angle_to_rad(const double &value, const std::string &unit)
{
    // Conversion factor
    double conversion;

    if (unit == "arcmin")
    {
        conversion = 2.9088820866e-4;
    }
    else if (unit == "deg")
    {
        conversion = 0.017453;
    }
    else if (unit == "rad")
    {
        conversion = 1;
    }
    else
    {
        std::cerr << "Unit not correctly specified. Needs to be arcmin, deg, or rad. Exiting.";
        exit(1);
    };
    return conversion * value;
}

double convert_rad_to_angle(const double &value, const std::string &unit)
{
    // Conversion factor
    double conversion;

    if (unit == "arcmin")
    {
        conversion = 3437.74677;
    }
    else if (unit == "deg")
    {
        conversion = 57.3;
    }
    else
    {
        std::cerr << "Unit not correctly specified. Needs to be arcmin or deg. Exiting.";
        exit(1);
    };
    return conversion * value;
}

void read_thetas(const std::string &fn, std::vector<double> &thetas)
{
    // Open file
    std::ifstream input(fn.c_str());
    if (input.fail())
    {
        std::cout << "read_thetas: Could not open " << fn << std::endl;
        return;
    };

    // Read in file
    if (input.is_open())
    {
        std::string line;
        while (std::getline(input, line))
        {
            if (line[0] == '#' || line.empty())
                continue;
            double theta;
            std::istringstream iss(line);
            iss >> theta;
            thetas.push_back(theta);
        };
    };
}

void convert_Pk(const std::map<double, double> &Pk_given, const int &n_bins, double &kMin, double &kMax, double &dk, std::vector<double> &Pk)
{
    kMin = Pk_given.begin()->first;
    kMax = Pk_given.rbegin()->first;

    dk = log(kMax / kMin) / n_bins;

    for (int i = 0; i < n_bins; i++)
    {
        double k = exp(log(kMin) + dk * i);
        Pk.push_back(valueMap(Pk_given, k));
    };
}

double valueMap(const std::map<double, double> &map, double value)
{
    auto ix = map.upper_bound(value); // Upper index limit
    if (ix == map.end())
    {
        double p = (--ix)->second;
        return p;
    };

    if (ix == map.begin())
    {
        double p = (ix)->second;
        return p;
    };
    auto ix_lower = ix;
    --ix_lower;

    double diff = (value - ix_lower->first) / (ix->first - ix_lower->first);
    return diff * ix->second + (1 - diff) * ix_lower->second;
}

int factorial(int n)
{
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

void read_gamma_config(const std::string &fn, configGamma &config)
{

  // Open file
  std::ifstream input(fn.c_str());
  if (input.fail())
  {
    std::cout << "read_gamma_config: Could not open " << fn << std::endl;
    return;
  };

  // Read in file
  std::vector<std::string> parameterNames;
  std::vector<double> parameterValues;

  if (input.is_open())
  {
    std::string line;
    while (std::getline(input, line))
    {
      if (line[0] == '#' || line.empty())
        continue;
      std::string name;
      double value;
      std::istringstream iss(line);
      iss >> name >> value;
      parameterNames.push_back(name);
      parameterValues.push_back(value);
    }
  }

  for (unsigned int i = 0; i < parameterNames.size(); i++)
  {
    if (parameterNames.at(i) == "rmin")
    {
      config.rmin = parameterValues.at(i);
    }
    else if (parameterNames.at(i) == "rmax")
    {
      config.rmax = parameterValues.at(i);
    }
    else if (parameterNames.at(i) == "umin")
    {
      config.umin = parameterValues.at(i);
    }
    else if (parameterNames.at(i) == "umax")
    {
      config.umax = parameterValues.at(i);
    }
    else if (parameterNames.at(i) == "vmin")
    {
      config.vmin = parameterValues.at(i);
    }
    else if (parameterNames.at(i) == "vmax")
    {
      config.vmax = parameterValues.at(i);
    }
    else if (parameterNames.at(i) == "rsteps")
    {
      config.rsteps = std::round(parameterValues.at(i));
    }
    else if (parameterNames.at(i) == "usteps")
    {
      config.usteps = std::round(parameterValues.at(i));
    }
    else if (parameterNames.at(i) == "vsteps")
    {
      config.vsteps = std::round(parameterValues.at(i));
    }
    else
    {
      std::cout << "Cosmology::Parameter file is not in the right format"
                << std::endl;
      return;
    }
  }
}

std::ostream &operator<<(std::ostream &out, const configGamma &config)
{
  out << "Binning r in " << config.rsteps << " bins from " << config.rmin << " to " << config.rmax << std::endl;
  out << "Binning u in " << config.usteps << " bins from " << config.umin << " to " << config.umax << std::endl;
  out << "Binning v in " << config.vsteps << " bins from " << config.vmin << " to " << config.vmax << std::endl;
  #ifdef CONVERT_TO_CENTROID
    out << "Setting the triangle center as Orthocenter" << std::endl;
  #else
    out << "Setting the triangle center as Centroid" << std::endl;
  #endif //CONVERT_TO_CENTROID
  return out;
}