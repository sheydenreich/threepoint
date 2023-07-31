import os
import sys
import subprocess
from subprocess import Popen
from tqdm import tqdm,trange
import argparse
import functions as fct
from time import time
from pathlib import Path
import numpy as np

os.chdir("/vol/euclid6/euclid6_1/pburger/software/threepoint-1.0.1/cuda_version_tomo_KiDS_IA/")
store_path = "/vol/euclidraid4/data/pburger/postdoc_Bonn/3pt_model/"

os.environ["CUDA_VISIBLE_DEVICES"]=str(1)

parser = argparse.ArgumentParser(
    description="Script for preparing training data for the LSS emulator.")

# parser.add_argument(
#     "--npix", default=4096, metavar="INT", type=int,
#     help="Number of pixels in the aperture mass map. default: %(default)s"
# )

parser.add_argument(
    "--statistic", default="map23", metavar="STR",
    help="Emulated statistic. Available: [map23,gamma]. default: %(default)s"
)

parser.add_argument(
    "--gpu_num", default=1, metavar="INT", type=int,
    help="Which GPU to use. default: %(default)s"
)

parser.add_argument(
    "--skip", default=0, metavar="INT", type=int,
    help="How many entries in the parameter space to skip"
)

parser.add_argument(
    "--output_name", default="", metavar="STR",
    help="Name of the output"
)

parser.add_argument(
    "--thetas", default="../necessary_files/Our_thetas.dat", metavar="STR",
    help="which file to use for computation of aperture masses. default: %(default)s"
)

parser.add_argument(
    "--debug", action="store_true",
    help="turns on debug mode. default: %(default)s"
)

parser.add_argument(
    "--nofz", default="../necessary_files/nz_SLICS_euclidlike.dat", metavar = "STR",
    help="which n(z) file to use for computation of aperture masses. default: %(default)s"
)

parser.add_argument(
    "--nodes", default=1500, metavar="INT", type=int,
    help="Number of training nodes. default: %(default)s"
)

parser.add_argument(
    "--cov_file", default = "../necessary_files/Covariance_SLICS.dat", metavar="STR",
    help="Filename for covariance parameters. default: %(default)s"
)


class cosmology:
    def __init__(self,h=0.6898,sigma_8=0.826,Omega_b=0.046,n_s=0.97,w=-1.,Omega_m=0.2905,z_max=3.2,A_IA=0.0):
        self.h = h
        self.sigma_8 = sigma_8
        self.Omega_b = Omega_b
        self.n_s = n_s
        self.w = w
        self.Omega_m = Omega_m
        self.z_max = z_max
        self.A_IA = A_IA

def write_cosmo_file(fname,cosmo):
    fil = open(fname,'w')
    fil.write('# TEMPORARY cosmology for calculation of aperture statistics in latin hypercube \n')
    fil.write('h {}\n'.format(cosmo.h))
    fil.write('sigma_8 {}\n'.format(cosmo.sigma_8))
    fil.write('Omega_b {}\n'.format(cosmo.Omega_b))
    fil.write('n_s {}\n'.format(cosmo.n_s))
    fil.write('w {}\n'.format(cosmo.w))
    fil.write('Omega_m {}\n'.format(cosmo.Omega_m))
    fil.write('z_max {}\n'.format(cosmo.z_max))
    fil.write('A_IA {}\n'.format(cosmo.A_IA))
    fil.close()

def save_single_z_and_theta_combis(combi,z_combi_file,theta_combi_file):
    z_combis = []
    theta_combis = []
    
    if(len(combi[0])==2):
        z_combis.append([np.int16(combi[0][0]),np.int16(combi[0][1]),999])
        theta_combis.append([np.float32(combi[1][0]),np.float32(combi[1][1]),999])
    else:
        z_combis.append([np.int16(combi[0][0]),np.int16(combi[0][1]),np.int16(combi[0][2])])
        theta_combis.append([np.float32(combi[1][0]),np.float32(combi[1][1]),np.float32(combi[1][2])])

    np.savetxt(z_combi_file,z_combis, fmt='%i')
    np.savetxt(theta_combi_file,theta_combis, fmt='%1.4e')

def save_z_and_theta_combis(combis,z_combi_file,theta_combi_file):
    z_combis = []
    theta_combis = []
    for combi in combis:
        if(len(combi[0])==2):
            z_combis.append([np.int16(combi[0][0]),np.int16(combi[0][1]),999])
            theta_combis.append([np.float32(combi[1][0]),np.float32(combi[1][1]),999])
        else:
            z_combis.append([np.int16(combi[0][0]),np.int16(combi[0][1]),np.int16(combi[0][2])])
            theta_combis.append([np.float32(combi[1][0]),np.float32(combi[1][1]),np.float32(combi[1][2])])

    np.savetxt(z_combi_file,z_combis, fmt='%i')
    np.savetxt(theta_combi_file,theta_combis, fmt='%1.4e')


nz_files = [store_path+'config_files/nofz/nofz_KiDS1000_temp_tomobin1_takashi_low.dat',
            store_path+'config_files/nofz/nofz_KiDS1000_temp_tomobin2_takashi_low.dat',
            store_path+'config_files/nofz/nofz_KiDS1000_temp_tomobin3_takashi_low.dat',
            store_path+'config_files/nofz/nofz_KiDS1000_temp_tomobin4_takashi_low.dat',
            store_path+'config_files/nofz/nofz_KiDS1000_temp_tomobin5_takashi_low.dat']

nz_files_name = nz_files[0]+' '+nz_files[1]+' '+nz_files[2]+' '+nz_files[3]+' '+nz_files[4]

sigma_epsilon = np.array([0.270,0.258,0.273,0.254,0.270])*np.sqrt(2)*0
ngal_per_arcmin = [0.62,1.18,1.85,1.26,1.31] 
shapenoise_file = store_path+'config_files/shapenoise.dat'
np.savetxt(shapenoise_file,np.array([sigma_epsilon,ngal_per_arcmin]).T,fmt='%.4f')


Number_of_train_nodes = 3000
Map23 = []
indices_available=[]
for i in np.arange(Number_of_train_nodes):
    my_file = Path('/vol/euclidraid4/data/pburger/postdoc_Bonn/3pt_model/results/train_data_full_model_IA_corrected/Map23_KiDS1000_T17_full_train_'+str(i)+'.dat')
    if my_file.is_file():
        data=np.loadtxt(my_file)
        Map23.append(data)
        indices_available.append(i)
indices_available=np.array(indices_available)
Map23 = np.array(Map23)
print(Map23.shape,indices_available.shape)

Map23_log10 = np.log10(np.abs(Map23))
Map23_diff = Map23_log10[:,1:]-Map23_log10[:,:-1]
mean_Map23 = np.mean(Map23_diff,axis=0)
std_Map23 = np.std(Map23_diff,axis=0)

cosmo_indices = []
combi_indices = []
for i in range(len(indices_available)):
    if(all(Map23[i]>0)):
        if(any((mean_Map23-Map23_diff[i])/std_Map23>5)):
            cosmo_indices.append(indices_available[i])
            combi_indices.append(np.where((mean_Map23-Map23_diff[i])/std_Map23>5)[0])
cosmo_indices = np.array(cosmo_indices)
combi_indices = np.array(combi_indices)


print(cosmo_indices,combi_indices)

# cosmo_indices = []
# combi_indices = []
# for i in range(Number_of_train_nodes):
#     if(any(Map23[i]<0)):
#         cosmo_indices.append(indices_available[i])
#         combi_indices.append(np.where(Map23[i]<0)[0])
# cosmo_indices= np.array(cosmo_indices)
# combi_indices = np.array(combi_indices)


# combis=fct.get_all_combis(all_theta_ap=['4','6','8','10','14','18','22','26','32','36'])
# equi_indices = fct.get_equi_tri_indices(combis)
# combis=combis[equi_indices]
combis=fct.get_all_combis()

z_combi_file=store_path+'config_files/z_combis.dat'
theta_combi_file=store_path+'config_files/theta_combis.dat'

cosmo_fname = store_path+'config_files/cosmo.dat'

param = np.load(store_path+'outputs_4_emulation_cp/paramters_for_training_IA_concatenated.npz')
for k in range(len(cosmo_indices)):
    cosmo_index = cosmo_indices[k] 
    combi_index = [*set(list(combi_indices[k])+list(combi_indices[k]+1))]
    if(cosmo_index<1500):
        continue

    print(cosmo_index,combi_index)
    save_z_and_theta_combis(combis[combi_index],z_combi_file,theta_combi_file)

    cosmo = cosmology(h=0.7,n_s=0.97,sigma_8=param['S8'][cosmo_index]*np.sqrt(0.3/param['Om'][cosmo_index]),Omega_m=param['Om'][cosmo_index],w=param['w'][cosmo_index],A_IA=param['A_IA'][cosmo_index])
    write_cosmo_file(fname=cosmo_fname,cosmo=cosmo)

    for sbin in np.arange(1,6):
        nz=np.loadtxt(store_path+'config_files/nofz/nofz_KiDS1000_tomobin'+str(sbin)+'_takashi_low.dat')
        nz[:,0]=nz[:,0]+param['dz'+str(sbin)][cosmo_index]
        np.savetxt(store_path+'config_files/nofz/nofz_KiDS1000_temp_tomobin'+str(sbin)+'_takashi_low.dat',nz[np.where(nz[:,0]>=0)])


    output_file = store_path+'results/train_data_full_model_IA_corrected/Map23_KiDS1000_temp.dat'
    os.system('./calculate_Map_n_KiDS10000.x '+cosmo_fname+' '+z_combi_file+' '+theta_combi_file+' '+output_file+' '+nz_files_name+' '+shapenoise_file)

    corrected_Map23 = np.loadtxt(store_path+'results/train_data_full_model_IA_corrected/Map23_KiDS1000_T17_full_train_'+str(cosmo_index)+'.dat')
    correction = np.loadtxt(store_path+'results/train_data_full_model_IA_corrected/Map23_KiDS1000_temp.dat')
    corrected_Map23[combi_index]=correction
    corrected_Map23 = np.savetxt(store_path+'results/train_data_full_model_IA_corrected/Map23_KiDS1000_T17_full_train_'+str(cosmo_index)+'.dat',corrected_Map23)




# Number_of_test_nodes = 500
# Map23 = []
# indices_available=[]
# for i in np.arange(Number_of_test_nodes):
#     my_file = Path('/vol/euclidraid4/data/pburger/postdoc_Bonn/3pt_model/results/test_data_full_model_IA_equi_corrected/Map23_KiDS1000_T17_full_test_'+str(i)+'.dat')
#     if my_file.is_file():
#         data=np.loadtxt(my_file)
#         Map23.append(data)
#         indices_available.append(i)
# indices_available=np.array(indices_available)
# Map23 = np.array(Map23)
# print(Map23.shape,indices_available.shape)

# Map23_log10 = np.log10(np.abs(Map23))
# Map23_diff = Map23_log10[:,1:]-Map23_log10[:,:-1]
# mean_Map23 = np.mean(Map23_diff,axis=0)
# std_Map23 = np.std(Map23_diff,axis=0)

# cosmo_indices = []
# combi_indices = []
# for i in range(len(indices_available)):
#     if(all(Map23[i]>0)):
#         if(any((mean_Map23-Map23_diff[i])/std_Map23>5)):
#             cosmo_indices.append(indices_available[i])
#             combi_indices.append(np.where((mean_Map23-Map23_diff[i])/std_Map23>5)[0])
# cosmo_indices = np.array(cosmo_indices)
# combi_indices = np.array(combi_indices)

# combis=fct.get_all_combis(all_theta_ap=['4','6','8','10','14','18','22','26','32','36'])
# equi_indices = fct.get_equi_tri_indices(combis)
# combis=combis[equi_indices]
# # combis=fct.get_all_combis()

# z_combi_file=store_path+'config_files/z_combis.dat'
# theta_combi_file=store_path+'config_files/theta_combis.dat'

# cosmo_fname = store_path+'config_files/cosmo.dat'

# param = np.load(store_path+'outputs_4_emulation_cp/paramters_for_testing_IA.npz')
# for k in range(len(cosmo_indices)):

#     cosmo_index = cosmo_indices[k] 
#     combi_index = [*set(list(combi_indices[k])+list(combi_indices[k]+1))]
#     print(cosmo_index,combi_index)
#     save_z_and_theta_combis(combis[combi_index],z_combi_file,theta_combi_file)

#     cosmo = cosmology(h=0.7,n_s=0.97,sigma_8=param['S8'][cosmo_index]*np.sqrt(0.3/param['Om'][cosmo_index]),Omega_m=param['Om'][cosmo_index],w=param['w'][cosmo_index],A_IA=param['A_IA'][cosmo_index])
#     write_cosmo_file(fname=cosmo_fname,cosmo=cosmo)

#     for sbin in np.arange(1,6):
#         nz=np.loadtxt(store_path+'config_files/nofz/nofz_KiDS1000_tomobin'+str(sbin)+'_takashi_low.dat')
#         nz[:,0]=nz[:,0]+param['dz'+str(sbin)][cosmo_index]
#         np.savetxt(store_path+'config_files/nofz/nofz_KiDS1000_temp_tomobin'+str(sbin)+'_takashi_low.dat',nz[np.where(nz[:,0]>=0)])


#     output_file = store_path+'results/test_data_full_model_IA_equi_corrected/Map23_KiDS1000_temp.dat'
#     os.system('./calculate_Map_n_KiDS10000.x '+cosmo_fname+' '+z_combi_file+' '+theta_combi_file+' '+output_file+' '+nz_files_name+' '+shapenoise_file)

#     corrected_Map23 = np.loadtxt(store_path+'results/test_data_full_model_IA_equi_corrected/Map23_KiDS1000_T17_full_test_'+str(cosmo_index)+'.dat')
#     correction = np.loadtxt(store_path+'results/test_data_full_model_IA_equi_corrected/Map23_KiDS1000_temp.dat')
#     corrected_Map23[combi_index]=correction
#     corrected_Map23 = np.savetxt(store_path+'results/test_data_full_model_IA_equi_corrected/Map23_KiDS1000_T17_full_test_'+str(cosmo_index)+'.dat',corrected_Map23)


    


