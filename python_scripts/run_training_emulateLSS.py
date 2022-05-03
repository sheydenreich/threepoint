import os
import sys
import subprocess
from subprocess import Popen
from tqdm import tqdm,trange
import argparse

from time import time

import numpy as np

os.chdir("../cuda_version/")

parser = argparse.ArgumentParser(
    description='Script for preparing training data for the LSS emulator.')

# parser.add_argument(
#     '--npix', default=4096, metavar='INT', type=int,
#     help='Number of pixels in the aperture mass map. default: %(default)s'
# )

parser.add_argument(
    '--statistic', default='map23', metavar='STR',
    help='Emulated statistic. Available: [map23,gamma]. default: %(default)s'
)

parser.add_argument(
    '--gpu_num', default=0, metavar='INT', type=int,
    help='Which GPU to use. default: %(default)s'
)

parser.add_argument(
    '--skip', default=0, metavar='INT', type=int,
    help='How many entries in the parameter space to skip'
)

parser.add_argument(
    '--output_name', default="", metavar='STR',
    help='Name of the output'
)

parser.add_argument(
    '--thetas', default="../necessary_files/Our_thetas.dat", metavar='STR',
    help='which file to use for computation of aperture masses. default: %(default)s'
)

parser.add_argument(
    "--debug", action="store_true",
    help="turns on debug mode. default: %(default)s"
)


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
            debug=False, gpu_num=0):
    
    command_map3 = ["./calculateApertureStatistics.x",cosmo_fname,theta_fname,out_fname+"_map3.dat","1",nz_fname]
    command_map2 = ["./calculateSecondOrderApertureStatistics.x",cosmo_fname,cov_fname,theta_fname,out_fname+"_map2.dat","1",nz_fname]

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)

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

    map2s = np.loadtxt(out_fname+"_map2.dat")#[:,-1]
    map3s = np.loadtxt(out_fname+"_map3.dat")#[:,-1]
    if(debug):
        print(map2s)
        print(map3s)
        print(map2s.shape)
        print(map3s.shape)
    map2s = map2s[:,-1]
    map3s = map3s[:,-1]
    result = np.concatenate((map2s,map3s))

    if(cleanup):
        os.remove(cosmo_fname)
        os.remove(out_fname+"_map2.dat")
        os.remove(out_fname+"_map3.dat")

    return result


def run_gamma(cosmo,logname="../logs/log.dat",logname_err="../logs/log_err.dat",
            out_fname="../../results/temp_gamma_emulateLSS",
            nz_fname = "../necessary_files/nz_SLICS_euclidlike.dat", config_fname = "../necessary_files/config_gamma_0p1_to_100_10_bins.dat",
            cosmo_fname = "../necessary_files/temp_cosmo.dat",cleanup=True,
            debug=False,gpu_num=None):
    if(gpu_num is None):
        command = ["./calculateGamma.x",cosmo_fname,config_fname,out_fname+"_gamma.dat","1",nz_fname]
    else:
        command = ["./calculateGamma.x",cosmo_fname,config_fname,out_fname+"_gamma.dat","1",nz_fname,str(gpu_num)]

    write_cosmo_file(cosmo_fname,cosmo)

    if(debug):
        proc = Popen(command)
        proc.communicate()
        print("Done.")
    else:
        proc = Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out,err) = proc.communicate()
        if err is not None and len(err)>0:
            print("WARNING: Error in calculation!")
            print(err)
            logfile = open(logname_err,"w")
            logfile.write(err.decode("utf-8"))
            logfile.close()
        logfile = open(logname,"w")
        logfile.write(out.decode("utf-8"))
        logfile.close()

    result = np.loadtxt(out_fname+"_gamma.dat")

    if(cleanup):
        os.remove(cosmo_fname)
        os.remove(out_fname+"_gamma.dat")

    return result


###### CALCULATION OF MAP23
# if(__name__=="__main__"):
#     training_data = np.loadtxt("../../EmulateLSS/training_data/latin_hypercube_params.txt")
#     results = np.zeros((training_data.shape[0],24))
#     for i,line in tqdm(enumerate(training_data),total = training_data.shape[0]):
#         om = line[0]
#         sig8 = line[1]
#         h = line[2]
#         w = line[3]

#         cosmo = cosmology(h=h,sigma_8=sig8,Omega_m=om,w=w)

#         map23 = run_map23(cosmo,logname="../logs/log_{}.dat".format(i),
#         logname_err = "../logs/log_err_{}.dat".format(i))

#         results[i] = map23
#         np.savetxt("../../EmulateLSS/training_data/single_results/result_{}.dat".format(i),map23)

#     np.savetxt("../../EmulateLSS/training_data/map23_from_latin_hypercube.dat",results)


###### CALCULATION OF GAMMA
if(__name__=="__main__"):
    args = parser.parse_args()
    training_data = np.loadtxt("../../EmulateLSS/training_data/latin_hypercube_params.txt")
    gpu_num = args.gpu_num

    for i,line in tqdm(enumerate(training_data),total = training_data.shape[0]):
        om = line[0]
        sig8 = line[1]
        h = line[2]
        w = line[3]

        if(i<args.skip):
            continue

        cosmo = cosmology(h=h,sigma_8=sig8,Omega_m=om,w=w)

        if(args.statistic=="gamma"):
            savename = "../../EmulateLSS/training_data/single_results/result_gamma_{}{}.dat".format(i,args.output_name)

            if(os.path.exists(savename)):
                print(savename," already exists. Moving on.")
            else:
                result = run_gamma(cosmo,logname="../logs/log_gamma_{}{}.dat".format(i,args.output_name),
                logname_err = "../logs/log_err_gamma_{}{}.dat".format(i,args.output_name),gpu_num=gpu_num,
                out_fname="../../results/temp_gamma_emulateLSS_{}".format(gpu_num),
                cosmo_fname="../necessary_files/temp_cosmo_{}.dat".format(gpu_num),
                debug=args.debug)

                np.savetxt(savename,result)

        elif(args.statistic=="map23"):
            savename = "../../EmulateLSS/training_data/single_results/result_map23_{}{}.dat".format(i,args.output_name)

            if(os.path.exists(savename)):
                print(savename," already exists. Moving on.")

            else:
                map23 = run_map23(cosmo,logname="../logs/log_{}{}.dat".format(i,args.output_name),
                logname_err = "../logs/log_err_{}{}.dat".format(i,args.output_name),gpu_num=gpu_num,
                out_fname="../../results/temp_gamma_emulateLSS_{}".format(gpu_num),
                cosmo_fname="../necessary_files/temp_cosmo_{}.dat".format(gpu_num),
                theta_fname=args.thetas, debug=args.debug)
                np.savetxt(savename,map23)
