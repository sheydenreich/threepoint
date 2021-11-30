#ifndef HELPER_HPP
#define HELPER_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

struct treecorr_bin
{
  double r,u,v;
};

int read_triangle_configurations(std::string& infile, std::vector<treecorr_bin>& triangle_config);

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
 * Reads in <Map^3> measurement from ASCII file
 * @param fn Filename of <Map^3>, format: theta1 theta2 theta3 Map3, lines starting with # are ignored
 * @param thetas Array which will contain the theta1, theta2, theta3 values
 * @param measurements Array which will contain the measurements <Map³>
 * @param unit Unit of the thetas, possible values are "rad", "deg", and "arcmin", default is "arcmin"
*/
void read_measurement(const std::string& fn, std::vector<std::vector<double>>& thetas, 
                      std::vector<double>& measurements, const std::string& unit="arcmin");



/**
 * Converts angle to radians
 * @param value Value that is to be converted
 * @param unit Unit of the value, possible values are "rad", "deg", and "arcmin", default is arcmin
 */
double convert_angle_to_rad(const double& value, const std::string& unit="arcmin");

#endif //HELPER_HPP
