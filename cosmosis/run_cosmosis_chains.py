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
    debug=False,only_om_s8=False,covariance='SLICS',name="",account_for_emulator_uncertainty=False,
    emulator_name = None, inputname="", theta_bins = 4, theta_max = 16):
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
        cosmosis_path+"outputs/results/output_"+datavec+"_"+cosmology+"_"+mode+"_"+covariance+name+".txt")
    
    if(only_om_s8):
        config.set("output","filename",
        cosmosis_path+"outputs/results/output_"+datavec+"_"+cosmology+"_"+mode+"_"+covariance+name+"_om_s8.txt")

    elif(sampler=="test"):
        dirname = cosmosis_path+"outputs/test/"+cosmology+"/"+mode
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        config.set("test","save_dir",cosmosis_path+"outputs/test/"+cosmology+"/"+mode+"/")



    config.set("threepoint_likelihood","cov_mat",cosmosis_path+"inputs/{}_cov_{}{}.dat".format(covariance,mode,inputname))
    config.set("threepoint_likelihood","data",cosmosis_path+"inputs/{}_{}{}.dat".format(datavec,mode,inputname))
    config.set("threepoint_likelihood","cov_inv",cosmosis_path+"outputs/{}_cov_inv_{}{}.dat".format(covariance,mode,inputname))

    if(covariance=="SLICS"):
        if(mode=="gamma"):
            config.set("threepoint_likelihood","n_sim","{}".format(np.loadtxt("inputs/SLICS_gamma{}_number_of_sims.dat".format(inputname))))
        else:
            config.set("threepoint_likelihood","n_sim","927")
    elif("MR" in covariance):
        config.set("threepoint_likelihood","n_sim","64")
    else:
        print("WARNING: Assuming analytic covariance!")
        config.set("threepoint_likelihood","n_sim","-1")


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
    elif(mode=="gamma"):
        config.set("threepoint_emulator","calculate_Map2","F")
        config.set("threepoint_emulator","calculate_Map3","F")
        config.set("threepoint_emulator","diag_only","F")
        config.set("threepoint_emulator","file","emulator_gamma.py")
        config.set("threepoint_emulator","mode","basic_gamma_emu_10_10_10_bins_0p1_to_100")
        config.set("threepoint_likelihood","likelihoods","gamma")
        if(account_for_emulator_uncertainty):
            config.set("threepoint_likelihood","account_for_emulator_error","T")
        else:
            config.set("threepoint_likelihood","account_for_emulator_error","F")
            
    else:
        print("Invalid mode! Available: map2,map3_diag,map3,joint,gamma")
        sys.exit()

    if mode in ["map2","map3","map3_diag","joint"]:
        config.set("threepoint_emulator","file","emulator_map.py")
        config.set("threepoint_emulator","mode","basic_map23_emu_thetas_2_to_16")
        config.set("threepoint_emulator","theta_bins",str(theta_bins))
        config.set("threepoint_likelihood","theta_max",str(theta_max))
        if("MR" in covariance):
            config.set("threepoint_likelihood","fieldsize",str(4*60))
        else:
            config.set("threepoint_likelihood","fieldsize",str(600))
        
        
    if(emulator_name is not None):
        config.set("threepoint_emulator","mode",emulator_name)

        
    if logname is None:
        logname = "logs/{}_{}_{}{}.log".format(datavec,cosmology,mode,name)
    logname_err = logname.split(".")[0]+"_err."+logname.split(".")[1]

    print("Running cosmosis on {} cores. \n Using {} cosmology, {} mode, {} datavector and {} covariance".format(n_cores,cosmology,mode,datavec,covariance))

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
    # run_chain("gamma","SLICS",n_cores=1,debug=True,account_for_emulator_uncertainty=True,name="_with_emulator_uncertainty")
    
    
    # run_chain("gamma","SLICS",name="_no_small_scales",inputname="_no_small_scales",
    #           emulator_name="basic_gamma_emu_10_10_10_bins_0p1_to_100_no_small_scales",n_cores=1,debug=True)
    # run_chain("gamma","SLICS",name="_no_large_scales",inputname="_no_large_scales",
    #           emulator_name="basic_gamma_emu_10_10_10_bins_0p1_to_100_no_large_scales",n_cores=1,debug=True)
    
    # run_chain("gamma","analytic_SLICS",name="_no_small_scales",inputname="_no_small_scales",
    #           emulator_name="basic_gamma_emu_10_10_10_bins_0p1_to_100_no_small_scales",n_cores=1,debug=True)
    # run_chain("gamma","analytic_SLICS",name="_no_large_scales",inputname="_no_large_scales",
    #           emulator_name="basic_gamma_emu_10_10_10_bins_0p1_to_100_no_large_scales",n_cores=1,debug=True)
    
    # run_chain("gamma","analytic_SLICS",n_cores=1,debug=True)
    
    for mode in all_modes:
        run_chain(mode,"analytic_SLICS",n_cores=1,name="_0p5_to_32",inputname="_0p5_to_32",
                  emulator_name="basic_map3_emu_0p5_to_32",debug=True,theta_bins=7, theta_max=32)

        run_chain(mode,"SLICS",n_cores=1,name="_0p5_to_32",inputname="_0p5_to_32",
                  emulator_name="basic_map3_emu_0p5_to_32",debug=False,theta_bins=7, theta_max=32)
        
        run_chain(mode,"analytic_SLICS",n_cores=1,debug=True)
        run_chain(mode,"SLICS",n_cores=1,debug=True)
        

    # all_covs_laila = ["analytic_SLICS_t1","analytic_SLICS_t1_inf","analytic_SLICS_t1_t2","analytic_SLICS_t1_inf_t2"]
    # for mode in ["map3_diag","map3"]:
    #     for cov in [all_covs_laila[2]]:
    #         # run_chain(mode, "analytic_SLICS", n_cores=128,debug=False,covariance=cov)
    #         run_chain(mode, "analytic_SLICS", n_cores=128,debug=False,covariance=cov,only_om_s8=True)

        # run_chain(mode,"analytic_SLICS",n_cores=128,debug=False,only_om_s8=True)
        # run_chain(mode,"analytic_SLICS",n_cores=128,debug=False)
        # run_chain(mode,"analytic_SLICS",n_cores=128,debug=False,only_om_s8=True,covariance="MR_div_35")
        # run_chain(mode,"analytic_SLICS",n_cores=128,debug=False,covariance="MR_div_35")

        # run_chain(mode,"SLICS",n_cores=128,debug=False,only_om_s8=True)
        # run_chain(mode,"SLICS",n_cores=128,debug=False)
        # run_chain(mode,"SLICS",n_cores=128,debug=False,only_om_s8=True,covariance="MR_div_35")
        # run_chain(mode,"SLICS",n_cores=128,debug=False,covariance="MR_div_35")