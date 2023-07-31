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

def get_SLICS(los,filepath = "/vol/euclid2/euclid2_raid2/sven/HOWLS/shear_catalogues/SLICS_LCDM/",
    filenotfound="warn"):
    # fieldsize = 600.
    # npix = 4096
    if(filenotfound=="error"):
        field = fits.open(filepath+"GalCatalog_LOS_cone{}.fits_s333{}_zmin0.0_zmax3.0_sys_3.fits".format(los,los))
    else:
        try:
            field = fits.open(filepath+"GalCatalog_LOS_cone{}.fits_s333{}_zmin0.0_zmax3.0_sys_3.fits".format(los,los))
        except Exception as e:
            if(filenotfound=="warn"):
                warnings.warn("Error loading file for los {}: {}".format(los,e))
            return -1

    data = field[1].data

    X_pos = data['ra_gal']*60.
    Y_pos = data['dec_gal']*60.
    shear_noise = -data['gamma1_noise']+1.0j*data['gamma2_noise']
    return X_pos,Y_pos,shear_noise


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

def get_one_mapmapmap_treecorr(los,
                    fpath = "/vol/euclid2/euclid2_raid2/sven/mapCorr_SLICS_for_T4_new/"):
    ac = aperture_mass_computer(4096,2,10.*60)
    theta_ap_array = [2,4,8,16]
    myfield = get_SLICS(los,filenotfound="warn")
    if(myfield==-1):
        return
    X_pos,Y_pos,shear_catalogue = myfield
    shears,norm = ac.normalize_shear(X_pos,Y_pos,shear_catalogue)


    idx_cut = int(round(4.*16 * 4096 / (10.*60)))
    cut_fieldsize = 10.*(4096-2*idx_cut)/4096
    fields = np.zeros((len(theta_ap_array),4096-2*idx_cut,4096-2*idx_cut))
    n_theta = len(theta_ap_array)
    for x,theta_ap in enumerate(theta_ap_array):
        ac.change_theta_ap(theta_ap)
        field = ac.Map_fft(shears,norm=norm,return_mcross=False,periodic_boundary=False)
        cut_field = field[idx_cut:-idx_cut,idx_cut:-idx_cut]
        fields[x] = cut_field
    print("Map calculation for los {} done. Resuming with T1!".format(los))

    map_squared = np.zeros((n_theta,n_theta))
    map_cubed = np.zeros((n_theta,n_theta,n_theta))
    map_four = np.zeros((n_theta,n_theta,n_theta,n_theta))
    map_six = np.zeros((n_theta,n_theta,n_theta,n_theta,n_theta,n_theta))

    for i in range(n_theta):
        for j in range(i,n_theta):
            savepath = fpath+"ggcorr_map_{}_map_{}_los_{}".format(theta_ap_array[i],
                                                                        theta_ap_array[j],
                                                                        los)
            if not os.path.exists(savepath+".npy"):
                gg = calculate_correlationfunction(cut_fieldsize,fields[i],fields[j],2000)
                np.save(savepath,gg)

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
    
    np.save(fpath+"map_squared_los_{}".format(los),map_squared)
    np.save(fpath+"map_cubed_los_{}".format(los),map_cubed)
    np.save(fpath+"map_four_los_{}".format(los),map_four)
    np.save(fpath+"map_six_los_{}".format(los),map_six)
            

    for i in range(n_theta):
        for j in range(i,n_theta):
            for k in range(n_theta):
                savepath = fpath+"ggcorr_mapsq_{}_{}_map_{}_los_{}".format(theta_ap_array[i],
                                                                                theta_ap_array[j],
                                                                                theta_ap_array[k],
                                                                                los)
                if not os.path.exists(savepath+".npy"):
                    gg = calculate_correlationfunction(cut_fieldsize,fields[i]*fields[j],fields[k],2000)
                    np.save(savepath,gg)


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
                        np.save(savepath,gg)
                    

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
                        np.save(savepath,gg)


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
                                np.save(savepath,gg)

def get_bins(los=74,
                    fpath = "/vol/euclid2/euclid2_raid2/sven/mapCorr_SLICS_for_T4_new/"):
    ac = aperture_mass_computer(4096,2,10.*60)
    theta_ap_array = [2]
    myfield = get_SLICS(los,filenotfound="warn")
    if(myfield==-1):
        return
    X_pos,Y_pos,shear_catalogue = myfield
    shears,norm = ac.normalize_shear(X_pos,Y_pos,shear_catalogue)


    idx_cut = int(round(4.*16 * 4096 / (10.*60)))
    cut_fieldsize = 10.*(4096-2*idx_cut)/4096
    fields = np.zeros((len(theta_ap_array),4096-2*idx_cut,4096-2*idx_cut))
    n_theta = len(theta_ap_array)
    for x,theta_ap in enumerate(theta_ap_array):
        ac.change_theta_ap(theta_ap)
        field = ac.Map_fft(shears,norm=norm,return_mcross=False,periodic_boundary=False)
        cut_field = field[idx_cut:-idx_cut,idx_cut:-idx_cut]
        fields[x] = cut_field
    print("Map calculation for los {} done. Resuming with T1!".format(los))

    savepath = fpath+"theta_bins"
    # if not os.path.exists(savepath+".npy"):
    gg = calculate_correlationfunction(cut_fieldsize,fields[0],fields[0],2000,calculate_bins=True)
    np.save(savepath,gg[0])



if(len(sys.argv)>1 and int(sys.argv[1])==1):
    print("Calculating bins!")
    get_bins()
else:
    calculate_multiprocessed(get_one_mapmapmap_treecorr,128,np.arange(74,1101))