from file_loader import get_kappa_millennium
from compute_aperture_mass import aperture_mass_computer
import numpy as np
import sys
import multiprocessing as mp
from itertools import permutations
import os
# from astropy.convolution import convolve_fft
from scipy import ndimage
from scipy.signal import fftconvolve,correlate
from astropy.io import fits
import warnings
from time import sleep

def patientsave(savepath,array,
                num_repeats = 10, wait=1, verbose=1):
    for i in range(num_repeats):
        try:
            np.save(savepath,array)
            if(i>0 and verbose>0):
                print("Saved {} after {} tries.".format(savepath,i))
            return
        except Exception as e:
            sleep(wait)
            if(verbose>1):
                print("Could not save {} in try {}: {}".format(savepath,i,e))
            if(i==num_repeats-1 and verbose==1): 
                print("Could not save {} in try {}: {}".format(savepath,i,e))

def patientload(savepath,array,
                num_repeats = 10, wait=1, verbose=1):
    for i in range(num_repeats):
        try:
            a = np.load(savepath,array)
            if(i>0 and verbose>0):
                print("Loaded {} after {} tries.".format(savepath,i))

            return a
        except Exception as e:
            sleep(wait)
            if(verbose>1):
                print("Could not load {} in try {}: {}".format(savepath,i,e))
            if(i==num_repeats-1 and verbose==1): 
                print("Could not load {} in try {}: {}".format(savepath,i,e))


def patientsavetxt(savepath,array,
                num_repeats = 10, wait=1, verbose=1):
    for i in range(num_repeats):
        try:
            np.savetxt(savepath,array)
            if(i>0 and verbose>0):
                print("Saved {} after {} tries.".format(savepath,i))
            return
        except Exception as e:
            sleep(wait)
            if(verbose>1):
                print("Could not save {} in try {}: {}".format(savepath,i,e))
            if(i==num_repeats-1 and verbose==1): 
                print("Could not save {} in try {}: {}".format(savepath,i,e))

def patientloadtxt(savepath,array,
                num_repeats = 10, wait=1, verbose=1):
    for i in range(num_repeats):
        try:
            a = np.loadtxt(savepath,array)
            if(i>0 and verbose>0):
                print("Loaded {} after {} tries.".format(savepath,i))
            return a
        except Exception as e:
            sleep(wait)
            if(verbose>1):
                print("Could not load {} in try {}: {}".format(savepath,i,e))
            if(i==num_repeats-1 and verbose==1): 
                print("Could not load {} in try {}: {}".format(savepath,i,e))

def read_slics_kappa(z,los):
    fname1 = "/vol/euclid7/euclid7_2/sven/slics_kappa/"+str(z)+"kappa_weight.dat_LOS"+str(los)
    npix = 7745
    # Read binary file into den_map
    try:
        with open(fname1, 'rb') as f1:
            kappa_bin = np.fromfile(f1, dtype=np.float32)
            kappa_map = np.reshape(np.float32(kappa_bin), [npix, npix])
        # Fix normalization
        kappa_map *= 64
        return kappa_map
    except:
        print(fname1," is not available. Available redshifts:")
        print("0.042,0.130,0.221,0.317,0.418,0.525,0.640,0.764,0.897,1.041,1.119,1.972,1.562,1.772,2.007,2.269,2.565,2.899")
        print("Available LOS:400,...,420")

def progressBar(name, value, endvalue, bar_length = 25, width = 20):
    """
    Displays a progress bar with "name", tracking the progess of value/endvalue
    """
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent*bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r{0: <{1}} : [{2}]{3}%".format(name, width, arrow + spaces, int(round(percent*100))))
    sys.stdout.flush()
    if value == endvalue:
         sys.stdout.write('\n\n')


def calculate_multiprocessed(function,processnum,arglist,name="Computing...",verbose=True):
    """
    calls the function with every element of arglist on processnum cores. Name reflects the progress bar.
    """
    length = len(arglist)
    exitflag = True
    counter = 0
    while(exitflag):
        if(verbose):
            progressBar(name,counter,length)
        for i in range(processnum):
            if(counter+i<length):
                job = mp.Process(target = function, args = [arglist[counter+i]])
                job.start()
            else:
                exitflag = False
        for i in range(processnum):
            if(counter+i<length):
                job.join()
                job.terminate()
        counter = counter+processnum

def azimuthalAverage(f,mask,n_bins,map_fieldsize,calculate_bins=False):
    sx, sy = f.shape
    X, Y = np.ogrid[0:sx, 0:sy]

    r = np.hypot(X - sx/2, Y - sy/2)

    rbin = (n_bins* r/r.max()).astype(np.int)
    rbin[mask] = -1
    radial_mean = ndimage.mean(f, labels=rbin, index=np.arange(n_bins))
    bins = np.arange(n_bins)*map_fieldsize*np.sqrt(2)/n_bins
    if(calculate_bins):
        print("Calculating bins!!")
        bins = ndimage.mean(r*2*map_fieldsize/r.shape[0], labels=rbin, index = np.arange(n_bins))
    return np.array([bins,radial_mean])

def EfunctionTimesArea(_x,_y,fieldsize):
    # equates to correlate(np.ones(map.shape),np.ones(map.shape))
    x = np.abs(_x)
    y = np.abs(_y)
    result = (fieldsize-x)*(fieldsize-y)
    result[x>fieldsize] = 0
    result[y>fieldsize] = 0
    return result


def calculate_2d_correlationfunction(map_fieldsize,map_field_1,map_field_2=None):
    mapCorr = correlate(map_field_1,map_field_2,'full','fft')
   
    idx,idy = np.indices(mapCorr.shape)
    idx = idx * map_fieldsize/map_field_1.shape[0]
    idy = idy * map_fieldsize/map_field_1.shape[0]
    idx = idx - map_fieldsize*idx.shape[0]/map_field_1.shape[0]/2
    idy = idy - map_fieldsize*idx.shape[0]/map_field_1.shape[0]/2
    norm = EfunctionTimesArea(idx,idy,map_fieldsize)
    mask = norm/map_fieldsize**2<1e-6   
    return mapCorr/norm*(map_fieldsize/map_field_1.shape[0])**2,mask

def calculate_correlationfunction(map_fieldsize,map_field_1,map_field_2,n_bins,calculate_bins=False):
    mapCorr,mask = calculate_2d_correlationfunction(map_fieldsize,map_field_1,map_field_2)
    corrf = azimuthalAverage(mapCorr,mask,n_bins,map_fieldsize,calculate_bins=calculate_bins)
    return corrf

def get_one_mapmapmap_treecorr(los,z=0.897,
                    fpath = "/vol/euclid2/euclid2_raid2/sven/mapCorr_SLICS_for_T4_kappa/"):
    theta_ap_array = [2,4,8,16]
    # savepath_test = fpath+"ggcorr_mapcu_{}_{}_{}_mapcu_{}_{}_{}_los_{}".format(theta_ap_array[-1],
    #             theta_ap_array[-1],
    #             theta_ap_array[-1],
    #             theta_ap_array[-1],
    #             theta_ap_array[-1],
    #             theta_ap_array[-1],
    #             los)
    # if os.path.exists(savepath_test+".npy"):
    #     print("{} already computed. Skipping.".format(los))
    #     return

    ac = aperture_mass_computer(7745,2,10.*60)
    kappa_field = read_slics_kappa(z,los)
    if(kappa_field is None):
        return


    idx_cut = int(round(4.*16 * 7745 / (10.*60)))
    cut_fieldsize = 10.*(7745-2*idx_cut)/7745
    fields = np.zeros((len(theta_ap_array),7745-2*idx_cut,7745-2*idx_cut))
    n_theta = len(theta_ap_array)
    for x,theta_ap in enumerate(theta_ap_array):
        ac.change_theta_ap(theta_ap)
        field = ac.Map_fft_from_kappa(kappa_field,periodic_boundary=False)
        cut_field = field[idx_cut:-idx_cut,idx_cut:-idx_cut]
        fields[x] = cut_field
    print("Map calculation for los {} done. Resuming with T1!".format(los))

    map_single = np.zeros((n_theta))
    map_squared = np.zeros((n_theta,n_theta))
    map_cubed = np.zeros((n_theta,n_theta,n_theta))
    map_four = np.zeros((n_theta,n_theta,n_theta,n_theta))
    map_six = np.zeros((n_theta,n_theta,n_theta,n_theta,n_theta,n_theta))

    for i in range(n_theta):
        map_single[i] = np.mean(fields[i])
    patientsave(fpath+"map_single_los_{}".format(los),map_single)

    for i in range(n_theta):
        for j in range(i,n_theta):
            savepath = fpath+"ggcorr_map_{}_map_{}_los_{}".format(theta_ap_array[i],
                                                                        theta_ap_array[j],
                                                                        los)
            if not os.path.exists(savepath+".npy"):
                gg = calculate_correlationfunction(cut_fieldsize,fields[i],fields[j],2000)
                patientsave(savepath,gg)

            meanfield = np.mean(fields[i]*fields[j])
            for _i,_j in permutations([i,j]):
                map_squared[_i,_j] = meanfield
            for k in range(j,n_theta):
                meanfield = np.mean(fields[i]*fields[j]*fields[k])
                for _i,_j,_k in permutations([i,j,k]):
                    map_cubed[_i,_j,_k] = meanfield
                for ii in range(k,n_theta):
                    meanfield = np.mean(fields[i]*fields[j]*fields[k]*fields[ii])
                    for _i,_j,_k,_ii in permutations([i,j,k,ii]):
                        map_four[_i,_j,_k,_ii] = meanfield
                    for jj in range(ii,n_theta):
                        for kk in range(jj,n_theta):
                            meanfield = np.mean(fields[i]*fields[j]*fields[k]*fields[ii]*fields[jj]*fields[kk])
                            for _i,_j,_k,_ii,_jj,_kk in permutations([i,j,k,ii,jj,kk]):
                                map_six[_i,_j,_k,_ii,_jj,_kk] = meanfield
    
    patientsave(fpath+"map_squared_los_{}".format(los),map_squared)
    patientsave(fpath+"map_cubed_los_{}".format(los),map_cubed)
    patientsave(fpath+"map_four_los_{}".format(los),map_four)
    patientsave(fpath+"map_six_los_{}".format(los),map_six)
            

    for i in range(n_theta):
        for j in range(i,n_theta):
            for k in range(n_theta):
                savepath = fpath+"ggcorr_mapsq_{}_{}_map_{}_los_{}".format(theta_ap_array[i],
                                                                                theta_ap_array[j],
                                                                                theta_ap_array[k],
                                                                                los)
                if not os.path.exists(savepath+".npy"):
                    gg = calculate_correlationfunction(cut_fieldsize,fields[i]*fields[j],fields[k],2000)
                    patientsave(savepath,gg)


    for i in range(n_theta):
        for j in range(i,n_theta):
            for ii in range(n_theta):
                for jj in range(ii,n_theta):
                    savepath = fpath+"ggcorr_mapsq_{}_{}_mapsq_{}_{}_los_{}.dat".format(theta_ap_array[i],
                                                            theta_ap_array[j],
                                                            theta_ap_array[ii],
                                                            theta_ap_array[jj],
                                                            los)
                    if not os.path.exists(savepath+".npy"):
                        gg = calculate_correlationfunction(cut_fieldsize,fields[i]*fields[j],fields[ii]*fields[jj],2000)
                        patientsave(savepath,gg)
                    

    for i in range(n_theta):
        for j in range(i,n_theta):
            for k in range(j,n_theta):
                for ii in range(n_theta):
                    savepath = fpath+"ggcorr_map_{}_mapcu_{}_{}_{}_los_{}".format(theta_ap_array[ii],
                                                                                theta_ap_array[i],
                                                                                theta_ap_array[j],
                                                                                theta_ap_array[k],
                                                                                los)
                    if not os.path.exists(savepath+".npy"):
                        gg = calculate_correlationfunction(cut_fieldsize,fields[ii],fields[i]*fields[j]*fields[k],2000)
                        patientsave(savepath,gg)


    for i in range(n_theta):
        for j in range(i,n_theta):
            for k in range(j,n_theta):
                for ii in range(n_theta):
                    for jj in range(ii,n_theta):
                        for kk in range(jj,n_theta):
                            savepath = fpath+"ggcorr_mapcu_{}_{}_{}_mapcu_{}_{}_{}_los_{}".format(theta_ap_array[i],
                                                            theta_ap_array[j],
                                                            theta_ap_array[k],
                                                            theta_ap_array[ii],
                                                            theta_ap_array[jj],
                                                            theta_ap_array[kk],
                                                            los)
                            if not os.path.exists(savepath+".npy"):
                                gg = calculate_correlationfunction(cut_fieldsize,fields[i]*fields[j]*fields[k],
                                                                fields[ii]*fields[jj]*fields[kk],2000)
                                patientsave(savepath,gg)

def get_bins(los=420,z=0.897,
                    fpath = "/vol/euclid2/euclid2_raid2/sven/mapCorr_SLICS_for_T4_kappa/"):
    ac = aperture_mass_computer(7745,2,10.*60)
    theta_ap_array = [2]
    kappa_field = read_slics_kappa(z,los)
    if(kappa_field is None):
        return


    idx_cut = int(round(4.*16 * 7745 / (10.*60)))
    cut_fieldsize = 10.*(7745-2*idx_cut)/7745
    fields = np.zeros((len(theta_ap_array),7745-2*idx_cut,7745-2*idx_cut))
    n_theta = len(theta_ap_array)
    for x,theta_ap in enumerate(theta_ap_array):
        ac.change_theta_ap(theta_ap)
        field = ac.Map_fft_from_kappa(kappa_field,periodic_boundary=False)
        cut_field = field[idx_cut:-idx_cut,idx_cut:-idx_cut]
        fields[x] = cut_field
    print("Map calculation for los {} done. Resuming with T1!".format(los))

    savepath = fpath+"theta_bins"
    # if not os.path.exists(savepath+".npy"):
    gg = calculate_correlationfunction(cut_fieldsize,fields[0],fields[0],2000,calculate_bins=True)
    patientsave(savepath,gg[0])



if(len(sys.argv)>1 and int(sys.argv[1])==1):
    print("Calculating bins!")
    get_bins()
else:
    calculate_multiprocessed(get_one_mapmapmap_treecorr,16,np.arange(74,802))
