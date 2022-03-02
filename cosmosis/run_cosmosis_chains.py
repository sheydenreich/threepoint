from __future__ import print_function
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0import sys
import os
import sys
import subprocess
from subprocess import Popen

from time import time


def run_chain(mode,datavec,cosmology='SLICS',sampler='multinest',
    cosmosis_path = "/users/sven/Documents/code/threepoint/cosmosis/",
    logname = None, n_cores = 32, ini_filename = "temp_ini.ini",
    debug=False,only_om_s8=False,covariance='SLICS'):
    config = ConfigParser()
    config.read("example_config.ini")
    if(cosmology=='SLICS'):
        config.set("threepoint_emulator","zMax","3.")
        config.set("threepoint_emulator","survet_area","360000.0")
        config.set("threepoint_emulator","galaxy_shapenoise","0.3654")
        config.set("threepoint_emulator","galaxy_density","30.")
        config.set("threepoint_emulator","nz_file",cosmosis_path+"inputs/nz_SLICS_euclidlike.dat")
        config.set("pipeline","values",cosmosis_path+"values.ini")
    elif(cosmology=='MR'):
        config.set("threepoint_emulator","zMax","1.1")
        config.set("threepoint_emulator","survet_area","360000.0")
        config.set("threepoint_emulator","galaxy_shapenoise","0.02")
        config.set("threepoint_emulator","galaxy_density","291.2711111")
        config.set("threepoint_emulator","nz_file",cosmosis_path+"inputs/nz_MR.dat")
        config.set("pipeline","values",cosmosis_path+"values_MR.ini")
        # config.set("pipeline","values",cosmosis_path+"values.ini")

    else:
        print("Invalid cosmology! Accepted: SLICS,MR")
        sys.exit()

    if(only_om_s8):
        config.set("pipeline","values",cosmosis_path+"values_om_s8.ini")


    config.set("runtime","sampler",sampler)

    if(sampler=="multinest"):
        dirname = cosmosis_path+"outputs/multinest/"+datavec+"/"+mode
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        config.set("multinest","multinest_outfile_root",cosmosis_path+"outputs/multinest/"+datavec+"/"+mode+"/")
        dirname = cosmosis_path+"outputs/results/"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        config.set("output","filename",
        cosmosis_path+"outputs/results/output_"+datavec+"_"+cosmology+"_"+mode+"_"+covariance+".txt")
    
    if(only_om_s8):
        config.set("output","filename",
        cosmosis_path+"outputs/results/output_"+datavec+"_"+cosmology+"_"+mode+"_"+covariance+"_om_s8.txt")

    elif(sampler=="test"):
        dirname = cosmosis_path+"outputs/test/"+cosmology+"/"+mode
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        config.set("test","save_dir",cosmosis_path+"outputs/test/"+cosmology+"/"+mode+"/")



    config.set("threepoint_likelihood","cov_mat",cosmosis_path+"inputs/{}_cov_{}.dat".format(covariance,mode))
    config.set("threepoint_likelihood","data",cosmosis_path+"inputs/{}_{}.dat".format(datavec,mode))
    config.set("threepoint_likelihood","cov_inv",cosmosis_path+"outputs/{}_cov_inv_{}.dat".format(covariance,mode))

    if(covariance=="SLICS"):
        config.set("threepoint_likelihood","n_sim","927")
    elif("MR" in covariance):
        config.set("threepoint_likelihood","n_sim","64")
    else:
        raise ValueError("Invalid covariance mode!")


    if(mode=="map2"):
        config.set("threepoint_emulator","calculate_Map2","T")
        config.set("threepoint_emulator","calculate_Map3","F")
        config.set("threepoint_emulator","diag_only","F")
        config.set("threepoint_likelihood","likelihoods","Map2")
    elif(mode=="map3_diag"):
        config.set("threepoint_emulator","calculate_Map2","F")
        config.set("threepoint_emulator","calculate_Map3","T")
        config.set("threepoint_emulator","diag_only","T")
        config.set("threepoint_likelihood","likelihoods","Map3")
    elif(mode=="map3"):
        config.set("threepoint_emulator","calculate_Map2","F")
        config.set("threepoint_emulator","calculate_Map3","T")
        config.set("threepoint_emulator","diag_only","F")
        config.set("threepoint_likelihood","likelihoods","Map3")
    elif(mode=="joint"):
        config.set("threepoint_emulator","calculate_Map2","T")
        config.set("threepoint_emulator","calculate_Map3","T")
        config.set("threepoint_emulator","diag_only","F")
        config.set("threepoint_likelihood","likelihoods","Map2 Map3")
    else:
        print("Invalid mode! Available: map2,map3_diag,map3,joint")
        sys.exit()
        
    if logname is None:
        logname = "logs/{}_{}_{}.log".format(datavec,cosmology,mode)
    logname_err = logname.split(".")[0]+"_err."+logname.split(".")[1]

    print("Running cosmosis on {} cores. \n Using {} cosmology, {} mode and {} datavector".format(n_cores,cosmology,mode,datavec))

    if(n_cores == 1):
        command = ["cosmosis",ini_filename]
    else:
        command = ["mpirun","-n",str(n_cores),"cosmosis","--mpi",ini_filename]

    with open(ini_filename,'w') as configfile:
        config.write(configfile)

    startt = time()

    if(debug):
        proc = Popen(command)
        proc.communicate()
    else:
        proc = Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out,err) = proc.communicate()
        # print(out,err)
        if err is not None and len(err)>0:
            logfile = open(logname_err,"w")
            logfile.write(err)
            logfile.close()
            print("WARNING: Error in run!")
        logfile = open(logname,"w")
        logfile.write("Run {} {} {} on {} cores \n".format(cosmology,mode,datavec,n_cores))
        taken_time = time()-startt
        hourtime = int(taken_time/3600)
        mintime = int(taken_time/60)%60
        sectime = int(taken_time)%60
        logfile.write("Finished in {} h, {} min, {} s \n".format(hourtime,mintime,sectime))
        logfile.write("Log: \n\n")
        logfile.write(out)
        logfile.close()


if(__name__=="__main__"):
    import numpy as np
    # all_modes = ["joint"]
    all_modes = ["map2","map3_diag","map3","joint"]
    cosmosis_path = "/users/sven/Documents/code/threepoint/cosmosis/"
    cosmology = "SLICS"
    # for mode in all_modes:
    #     run_chain(mode,"slics_mean",sampler='test',n_cores=1)
    #     testpath = cosmosis_path+"outputs/test/{}/{}/threepoint/".format(cosmology,mode)
    #     test_data = np.zeros(0)
    #     for data in ["map2s.txt","map3s.txt"]:
    #         test_data = np.append(test_data,np.loadtxt(testpath+data))
    #     np.savetxt(cosmosis_path+"inputs/analytic_{}_{}.dat".format(cosmology,mode),test_data)
        # print(mode,test_data.shape)

    for mode in all_modes:
        # run_chain(mode,"analytic_SLICS",n_cores=128,debug=False,only_om_s8=True)
        # run_chain(mode,"analytic_SLICS",n_cores=128,debug=False)
        run_chain(mode,"analytic_SLICS",n_cores=128,debug=False,only_om_s8=True,covariance="MR_div_35")
        run_chain(mode,"analytic_SLICS",n_cores=128,debug=False,covariance="MR_div_35")

        # run_chain(mode,"SLICS",n_cores=128,debug=False,only_om_s8=True)
        # run_chain(mode,"SLICS",n_cores=128,debug=False)
        run_chain(mode,"SLICS",n_cores=128,debug=False,only_om_s8=True,covariance="MR_div_35")
        run_chain(mode,"SLICS",n_cores=128,debug=False,covariance="MR_div_35")