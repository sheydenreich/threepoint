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

os.chdir('/vol/euclid6/euclid6_1/pburger/software/threepoint-1.0.1/cuda_version_tomo_KiDS_IA/')
store_path = '/vol/euclidraid4/data/pburger/postdoc_Bonn/3pt_model/'

even_or_odd = 'odd' # even odd
GPU_number=1
if(even_or_odd=='odd'):
    GPU_number=0

os.environ['CUDA_VISIBLE_DEVICES']=str(GPU_number)

cosmology_indices = np.arange(500,1000)[::2]
if(even_or_odd=='odd'):
    cosmology_indices = np.arange(500,1000)[::2]+1

print(GPU_number,cosmology_indices)

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
    '--gpu_num', default=GPU_number, metavar='INT', type=int,
    help='Which GPU to use. default: %(default)s'
)

parser.add_argument(
    '--skip', default=0, metavar='INT', type=int,
    help='How many entries in the parameter space to skip'
)

parser.add_argument(
    '--output_name', default='', metavar='STR',
    help='Name of the output'
)

parser.add_argument(
    '--thetas', default='../necessary_files/Our_thetas.dat', metavar='STR',
    help='which file to use for computation of aperture masses. default: %(default)s'
)

parser.add_argument(
    '--debug', action='store_true',
    help='turns on debug mode. default: %(default)s'
)

parser.add_argument(
    '--nofz', default='../necessary_files/nz_SLICS_euclidlike.dat', metavar = 'STR',
    help='which n(z) file to use for computation of aperture masses. default: %(default)s'
)

parser.add_argument(
    '--nodes', default=1500, metavar='INT', type=int,
    help='Number of training nodes. default: %(default)s'
)

parser.add_argument(
    '--cov_file', default = '../necessary_files/Covariance_SLICS.dat', metavar='STR',
    help='Filename for covariance parameters. default: %(default)s'
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


# combis=fct.get_all_combis(all_theta_ap=['4','6','8','10','14','18','22','26','32','36'])
# equi_indices = fct.get_equi_tri_indices(combis)
# combis=combis[equi_indices]
combis=fct.get_all_combis()
print(combis) 

z_combi_file=store_path+'config_files/z_combis.dat'
theta_combi_file=store_path+'config_files/theta_combis.dat'
save_z_and_theta_combis(combis,z_combi_file,theta_combi_file)

nz_files = [store_path+'config_files/nofz/nofz_KiDS1000_temp_'+even_or_odd+'_tomobin1_takashi_low.dat',
            store_path+'config_files/nofz/nofz_KiDS1000_temp_'+even_or_odd+'_tomobin2_takashi_low.dat',
            store_path+'config_files/nofz/nofz_KiDS1000_temp_'+even_or_odd+'_tomobin3_takashi_low.dat',
            store_path+'config_files/nofz/nofz_KiDS1000_temp_'+even_or_odd+'_tomobin4_takashi_low.dat',
            store_path+'config_files/nofz/nofz_KiDS1000_temp_'+even_or_odd+'_tomobin5_takashi_low.dat']
nz_files_name = nz_files[0]+' '+nz_files[1]+' '+nz_files[2]+' '+nz_files[3]+' '+nz_files[4]

cosmo_fname = store_path+'config_files/cosmo_'+even_or_odd+'.dat'

sigma_epsilon = np.array([0.270,0.258,0.273,0.254,0.270])*np.sqrt(2)*0
ngal_per_arcmin = [0.62,1.18,1.85,1.26,1.31] 
shapenoise_file = store_path+'config_files/shapenoise.dat'
np.savetxt(shapenoise_file,np.array([sigma_epsilon,ngal_per_arcmin]).T,fmt='%.4f')

param = np.load(store_path+'outputs_4_emulation_cp/paramters_for_training_IA_concatenated.npz')
for i in cosmology_indices:

    my_file = Path(store_path+'results/train_data_full_model_IA/Map23_KiDS1000_T17_full_train_'+str(i)+'.dat')
    if my_file.is_file():
        continue

    cosmo = cosmology(h=0.7,n_s=0.97,sigma_8=param['S8'][i]*np.sqrt(0.3/param['Om'][i]),Omega_m=param['Om'][i],w=param['w'][i],A_IA=param['A_IA'][i])
    write_cosmo_file(fname=cosmo_fname,cosmo=cosmo)
    print(i)
    for sbin in np.arange(1,6):
        nz=np.loadtxt(store_path+'config_files/nofz/nofz_KiDS1000_tomobin'+str(sbin)+'_takashi_low.dat')
        nz[:,0]=nz[:,0]+param['dz'+str(sbin)][i]
        np.savetxt(store_path+'config_files/nofz/nofz_KiDS1000_temp_'+even_or_odd+'_tomobin'+str(sbin)+'_takashi_low.dat',nz[np.where(nz[:,0]>=0)])

    output_file = store_path+'results/train_data_full_model_IA/Map23_KiDS1000_T17_full_train_'+str(i)+'.dat'
    os.system('./calculate_Map_n_KiDS10000.x '+cosmo_fname+' '+z_combi_file+' '+theta_combi_file+' '+output_file+' '+nz_files_name+' '+shapenoise_file)#+'  >> '+store_path+'results/Map23_temp_model_opt.dat')
    output_file_4_correction = store_path+'results/train_data_full_model_IA_corrected/Map23_KiDS1000_T17_full_train_'+str(i)+'.dat'
    os.system('cp '+output_file+' '+output_file_4_correction)


param = np.load(store_path+'outputs_4_emulation_cp/paramters_for_testing_IA_concatenated.npz')
for i in cosmology_indices:

    my_file = Path(store_path+'results/test_data_full_model_IA/Map23_KiDS1000_T17_full_test_'+str(i)+'.dat')
    if my_file.is_file():
        continue

    cosmo = cosmology(h=0.7,n_s=0.97,sigma_8=param['S8'][i]*np.sqrt(0.3/param['Om'][i]),Omega_m=param['Om'][i],w=param['w'][i],A_IA=param['A_IA'][i])
    write_cosmo_file(fname=cosmo_fname,cosmo=cosmo)
    print(i)
    for sbin in np.arange(1,6):
        nz=np.loadtxt(store_path+'config_files/nofz/nofz_KiDS1000_tomobin'+str(sbin)+'_takashi_low.dat')
        nz[:,0]=nz[:,0]+param['dz'+str(sbin)][i]
        np.savetxt(store_path+'config_files/nofz/nofz_KiDS1000_temp_'+even_or_odd+'_tomobin'+str(sbin)+'_takashi_low.dat',nz[np.where(nz[:,0]>=0)])

    output_file = store_path+'results/test_data_full_model_IA/Map23_KiDS1000_T17_full_test_'+str(i)+'.dat'
    os.system('./calculate_Map_n_KiDS10000.x '+cosmo_fname+' '+z_combi_file+' '+theta_combi_file+' '+output_file+' '+nz_files_name+' '+shapenoise_file)#+'  >> '+store_path+'results/Map23_temp_model_opt.dat')
    output_file_4_correction = store_path+'results/test_data_full_model_IA_corrected/Map23_KiDS1000_T17_full_test_'+str(i)+'.dat'
    os.system('cp '+output_file+' '+output_file_4_correction)









    


