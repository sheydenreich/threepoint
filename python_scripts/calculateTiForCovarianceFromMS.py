from file_loader import get_kappa_millennium
from compute_aperture_mass import aperture_mass_computer
import numpy as np
import sys
import multiprocessing as mp
import treecorr
from itertools import permutations
import os

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

def treecorr_2pcf(map_fieldsize,map_field_1,map_field_2=None):
    idx,idy = np.indices(map_field_1.shape)
    idx = idx*map_fieldsize/map_field_1.shape[0]
    idy = idy*map_fieldsize/map_field_1.shape[0]
    # print(idx.ravel().shape)
    # print(map_field_2.shape,map_field_1.shape,idx.shape,idy.shape)
    cat1 = treecorr.Catalog(x=idx.ravel(),y=idy.ravel(),k=map_field_1.ravel(),x_units='deg',y_units='deg')
    gg = treecorr.KKCorrelation(nbins=50,min_sep=0.05,max_sep=180,sep_units='arcmin',num_threads=16,
                                )
    if(map_field_2 is not None):
        cat2 = treecorr.Catalog(x=idx.ravel(),y=idy.ravel(),k=map_field_2.ravel(),
                    x_units='deg',y_units='deg')
        gg.process(cat1,cat2)
    else:
        gg.process(cat1)
    return gg

def get_one_mapmapmap_treecorr(los,
                    fpath = "/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_MR/terms_for_T4/"):
    ac = aperture_mass_computer(4096,2,4.*60)
    theta_ap_array = [2,4,8,16]
    kappa_ms = get_kappa_millennium(los)
    # kappa_ms = np.random.normal(size=(4096,4096))
    idx_cut = int(round(4.*16 * 4096 / (4.*60)))
    cut_fieldsize = 4.*(4096-2*idx_cut)/4096
    fields = np.zeros((len(theta_ap_array),4096-2*idx_cut,4096-2*idx_cut))
    n_theta = len(theta_ap_array)
    for x,theta_ap in enumerate(theta_ap_array):
        ac.change_theta_ap(theta_ap)
        field = ac.Map_fft_from_kappa(kappa_ms)
        cut_field = field[idx_cut:-idx_cut,idx_cut:-idx_cut]
        fields[x] = cut_field
    print("Map calculation for los {} done. Resuming with T1!".format(los))

    map_map_means = np.zeros((n_theta,n_theta))
    for i in range(n_theta):
        for j in range(i,n_theta):
            if not os.path.exists(fpath+"ggcorr_map_{}_map_{}_los_{}.dat".format(theta_ap_array[i],
                                                                        theta_ap_array[j],
                                                                        los)):
                gg = treecorr_2pcf(cut_fieldsize,fields[i],fields[j])
                gg.write(fpath+"ggcorr_map_{}_map_{}_los_{}.dat".format(theta_ap_array[i],
                                                                            theta_ap_array[j],
                                                                            los))
            meanfield = np.mean(fields[i]*fields[j])
            map_map_means[i,j] = meanfield
            map_map_means[j,i] = meanfield
    np.save(fpath+"means_map_map_los_{}".format(los),map_map_means)
    print("los {} gg calculation for T1 done. Resuming with T4".format(los))

    mapsq_map_means = np.zeros((n_theta,n_theta,n_theta))
    for i in range(n_theta):
        for j in range(i,n_theta):
            for k in range(n_theta):
                if not os.path.exists(fpath+"ggcorr_mapsq_{}_{}_map_{}_los_{}.dat".format(theta_ap_array[i],
                                                                                theta_ap_array[j],
                                                                                theta_ap_array[k],
                                                                                los)):
                    gg = treecorr_2pcf(cut_fieldsize,fields[i]*fields[j],fields[k])
                    gg.write(fpath+"ggcorr_mapsq_{}_{}_map_{}_los_{}.dat".format(theta_ap_array[i],
                                                                                theta_ap_array[j],
                                                                                theta_ap_array[k],
                                                                                los))
                meanfield = np.mean(fields[i]*fields[j]*fields[k])
                mapsq_map_means[i,j,k] = meanfield
                mapsq_map_means[j,i,k] = meanfield

    np.save(fpath+"means_mapsq_map_los_{}".format(los),mapsq_map_means)
    print("los {} gg calculation for T4 done. Resuming with T7".format(los))

    mapsq_mapsq_means = np.zeros((n_theta,n_theta,n_theta,n_theta))
    for i in range(n_theta):
        for j in range(i,n_theta):
            for ii in range(n_theta):
                for jj in range(ii,n_theta):
                    savename = fpath+"ggcorr_mapsq_{}_{}_mapsq_{}_{}_los_{}.dat".format(theta_ap_array[i],
                                                            theta_ap_array[j],
                                                            theta_ap_array[ii],
                                                            theta_ap_array[jj],
                                                            los)
                    if not os.path.exists(savename):
                        gg = treecorr_2pcf(cut_fieldsize,fields[i]*fields[j],fields[ii]*fields[jj])
                        gg.write(savename)
                    
                    meanfield = np.mean(fields[i]*fields[j]*fields[ii]*fields[jj])
                    mapsq_mapsq_means[i,j,ii,jj] = meanfield
                    mapsq_mapsq_means[j,i,ii,jj] = meanfield
                    mapsq_mapsq_means[i,j,jj,ii] = meanfield
                    mapsq_mapsq_means[j,i,jj,ii] = meanfield

    np.save(fpath+"means_mapsq_mapsq_los_{}".format(los),mapsq_mapsq_means)


    map_mapcu_means = np.zeros((n_theta,n_theta,n_theta,n_theta))
    for i in range(n_theta):
        for j in range(i,n_theta):
            for k in range(j,n_theta):
                for ii in range(n_theta):
                    if not os.path.exists(fpath+"ggcorr_map_{}_mapcu_{}_{}_{}_los_{}.dat".format(theta_ap_array[ii],
                                                                                theta_ap_array[i],
                                                                                theta_ap_array[j],
                                                                                theta_ap_array[k],
                                                                                los)):
                        gg = treecorr_2pcf(cut_fieldsize,fields[ii],fields[i]*fields[j]*fields[k])
                        gg.write(fpath+"ggcorr_map_{}_mapcu_{}_{}_{}_los_{}.dat".format(theta_ap_array[ii],
                                                                                    theta_ap_array[i],
                                                                                    theta_ap_array[j],
                                                                                    theta_ap_array[k],
                                                                                    los))
                    meanfield = np.mean(fields[ii]*fields[i]*fields[j]*fields[k])
                    map_mapcu_means[ii,i,j,k] = meanfield
                    map_mapcu_means[ii,i,k,j] = meanfield
                    map_mapcu_means[ii,j,i,k] = meanfield
                    map_mapcu_means[ii,j,k,i] = meanfield
                    map_mapcu_means[ii,k,i,j] = meanfield
                    map_mapcu_means[ii,k,j,i] = meanfield

    np.save(fpath+"means_map_mapcu_los_{}".format(los),map_mapcu_means)
    print("los {} gg calculation for T4 done. Resuming with T7".format(los))


    mapcu_mapcu_means = np.zeros((n_theta,n_theta,n_theta,n_theta,n_theta,n_theta))
    for i in range(n_theta):
        for j in range(i,n_theta):
            for k in range(j,n_theta):
                for ii in range(n_theta):
                    for jj in range(ii,n_theta):
                        for kk in range(jj,n_theta):
                            if not os.path.exists(fpath+"ggcorr_mapcu_{}_{}_{}_mapcu_{}_{}_{}_los_{}.dat".format(theta_ap_array[i],
                                                            theta_ap_array[j],
                                                            theta_ap_array[k],
                                                            theta_ap_array[ii],
                                                            theta_ap_array[jj],
                                                            theta_ap_array[kk],
                                                            los)):
                                gg = treecorr_2pcf(cut_fieldsize,fields[i]*fields[j]*fields[k],
                                                                fields[ii]*fields[jj]*fields[kk])
                                gg.write(fpath+"ggcorr_mapcu_{}_{}_{}_mapcu_{}_{}_{}_los_{}.dat".format(theta_ap_array[i],
                                                                theta_ap_array[j],
                                                                theta_ap_array[k],
                                                                theta_ap_array[ii],
                                                                theta_ap_array[jj],
                                                                theta_ap_array[kk],
                                                                los))
                            meanfield = np.mean(fields[i]*fields[j]*fields[k]*fields[ii]*fields[jj]*fields[kk])
                            for idi,idj,idk in permutations([i,j,k]):
                                for idii,idjj,idkk in permutations([ii,jj,kk]):
                                    mapcu_mapcu_means[idi,idj,idk,idii,idjj,idkk] = meanfield

    np.save(fpath+"means_mapcu_mapcu_los_{}".format(los),mapcu_mapcu_means)
    print("los {} gg calculation for T7 done.".format(los))



calculate_multiprocessed(get_one_mapmapmap_treecorr,16,np.arange(64))
