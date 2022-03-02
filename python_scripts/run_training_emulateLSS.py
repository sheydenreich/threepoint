import os
import sys
import subprocess
from subprocess import Popen
from tqdm import tqdm,trange

from time import time

import numpy as np

os.chdir("../cuda_version/")

class cosmology:
    def __init__(self,h=0.6898,sigma_8=0.826,Omega_b=0.0473,n_s=0.969,w=-1.,Omega_m=0.2905,z_max=3.):
        self.h = h
        self.sigma_8 = sigma_8
        self.Omega_b = Omega_b
        self.n_s = n_s
        self.w = w
        self.Omega_m = Omega_m
        self.z_max = z_max

def write_cosmo_file(fname,cosmo):
    fil = open(fname,"w")
    fil.write("# TEMPORARY cosmology for calculation of aperture statistics in latin hypercube \n")
    fil.write("h {}\n".format(cosmo.h))
    fil.write("sigma_8 {}\n".format(cosmo.sigma_8))
    fil.write("Omega_b {}\n".format(cosmo.Omega_b))
    fil.write("n_s {}\n".format(cosmo.n_s))
    fil.write("w {}\n".format(cosmo.w))
    fil.write("Omega_m {}\n".format(cosmo.Omega_m))
    fil.write("z_max {}\n".format(cosmo.z_max))
    fil.close()

def run_map23(cosmo,logname="../logs/log.dat",logname_err="../logs/log_err.dat",
            out_fname="../../results/temp_maps_emulateLSS",
            nz_fname = "../necessary_files/nz_SLICS_euclidlike.dat", theta_fname = "../necessary_files/Our_thetas.dat",
            cosmo_fname = "../necessary_files/temp_cosmo.dat",cleanup=True,cov_fname = "../necessary_files/Covariance_SLICS.dat",
            debug=False):
    command_map3 = ["./calculateApertureStatistics.x",cosmo_fname,theta_fname,out_fname+"_map3.dat","1",nz_fname]
    command_map2 = ["./calculateSecondOrderApertureStatistics.x",cosmo_fname,cov_fname,theta_fname,out_fname+"_map2.dat","1",nz_fname]

    write_cosmo_file(cosmo_fname,cosmo)

    if(debug):
        proc = Popen(command_map2)
        proc.communicate()

        print("Map2 done. Running Map3.")
        proc = Popen(command_map3)
        proc.communicate()
        print("Map3 done.")
    else:
        proc = Popen(command_map2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out,err) = proc.communicate()
        if err is not None and len(err)>0:
            print("WARNING: Error in Map2!")
            print(err)
            logfile = open(logname_err,"w")
            logfile.write(err.decode("utf-8"))
            logfile.close()
        logfile = open(logname,"w")
        logfile.write(out.decode("utf-8"))
        logfile.close()

        proc = Popen(command_map3, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out,err) = proc.communicate()
        if err is not None and len(err)>0:
            print("WARNING: Error in Map3!")
            print(err)
            logfile = open(logname_err,"w+")
            logfile.write(err.decode("utf-8"))
            logfile.close()
        logfile = open(logname,"w+")
        logfile.write(out.decode("utf-8"))
        logfile.close()

    map2s = np.loadtxt(out_fname+"_map2.dat")[:,-1]
    map3s = np.loadtxt(out_fname+"_map3.dat")[:,-1]
    result = np.concatenate((map2s,map3s))

    if(cleanup):
        os.remove(cosmo_fname)
        os.remove(out_fname+"_map2.dat")
        os.remove(out_fname+"_map3.dat")

    return result


if(__name__=="__main__"):
    training_data = np.loadtxt("../../EmulateLSS/training_data/latin_hypercube_params.txt")
    results = np.zeros((training_data.shape[0],24))
    for i,line in tqdm(enumerate(training_data),total = training_data.shape[0]):
        om = line[0]
        sig8 = line[1]
        h = line[2]
        w = line[3]

        cosmo = cosmology(h=h,sigma_8=sig8,Omega_m=om,w=w)

        map23 = run_map23(cosmo,logname="../logs/log_{}.dat".format(i),
        logname_err = "../logs/log_err_{}.dat".format(i))

        results[i] = map23
        np.savetxt("../../EmulateLSS/training_data/single_results/result_{}.dat".format(i),map23)

    np.savetxt("../../EmulateLSS/training_data/map23_from_latin_hypercube.dat",results)

    
