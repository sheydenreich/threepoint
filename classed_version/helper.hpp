#ifndef HELPER_HPP
#define HELPER_HPP

#include <string>
#include <iostream>
#include <fstream>

struct treecorr_bin
{
	double r,u,v;
};

int read_triangle_configurations(std::string& infile, treecorr_bin* triangle_config, int rbins, int ubins, int vbins)
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
		int idx = rind*ubins*vbins + uind*vbins + vind-vbins;
		triangle_config[idx].r = r;
		triangle_config[idx].u = u;
		triangle_config[idx].v = v;
	}

	fin.close();

	return 1;

}


#endif //HELPER_HPP
