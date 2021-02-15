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


#endif //HELPER_HPP
