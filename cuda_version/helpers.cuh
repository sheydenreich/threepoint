#ifndef HELPERS_CUH
#define HELPERS_CUH

#include <string>
#include <vector>
#include <map>

/**
 * Reads in n(z) from ASCII file, normalizes it and casts it into given bins (using linear interpolation)
 * @warning Assumes linear binning in z
 * @param fn Filename of n(z), format: z n(z), lines starting with # are ignored
 * @param n_bins Number of bins of final n(z)
 * @param zMax Maximal redshift of final n(z)
 * @param nz array, which will contain the final n(z) values
 */
 void read_n_of_z(const std::string& fn, const int& n_bins, const double& zMax, std::vector<double>& nz);


/**
 * Converts angle to radians
 * @param value Value that is to be converted
 * @param unit Unit of the value, possible values are "rad", "deg", and "arcmin", default is arcmin
 */
 double convert_angle_to_rad(const double& value, const std::string& unit="arcmin");

 /**
  * Converts radians to angular unit
  * @param value Value that is to be converted
  * @param unit Unit value should have, possible values are "deg" and "arcmin", default is arcmin
 */
 double convert_rad_to_angle(const double& value, const std::string& unit="arcmin");

 void convert_Pk(const std::map<double, double>& Pk_given, const int& n_bins, double& kMin, double& kMax, double& dk, std::vector<double>& Pk);


// Read in of thetas (in arcmin!)
void read_thetas(const std::string& fn, std::vector<double>& thetas);


double valueMap(const std::map<double, double>& map, double value);

int factorial(int n);
 #endif //HELPERS_CUH