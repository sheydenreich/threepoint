import numpy as np
from compute_aperture_masses import progressBar

results = np.zeros((7,7,7,64))
theta_ap = [0.5,1,2,4,8,16,32]
counter = 0
for los in range(64):
        for i in range(7):
                field1,err = np.load("/vol/euclid7/euclid7_2/sven/maps_millennium/theta_"+str(theta_ap[i])+"_los_"+str(los)+".npy")
                for j in range(7):
                        field2,err = np.load("/vol/euclid7/euclid7_2/sven/maps_millennium/theta_"+str(theta_ap[j])+"_los_"+str(los)+".npy")
                        for k in range(7):
                                field3,err = np.load("/vol/euclid7/euclid7_2/sven/maps_millennium/theta_"+str(theta_ap[k])+"_los_"+str(los)+".npy")                        
                                progressBar("Reading fields",counter,64*7**3)
                                maxtheta = theta_ap[np.max([i,j,k])]
                                index_maxtheta = int(maxtheta/(4*60)*4096)
                                field1_cut = field1[index_maxtheta:(4096-index_maxtheta),index_maxtheta:(4096-index_maxtheta)]
                                field2_cut = field2[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]
                                field3_cut = field3[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]

                                results[i,j,k,los] = np.mean(field1_cut*field2_cut*field3_cut)
                                results[i,j,k,los] *= (4*60/4096)**6 #account for incorrect normalization of aperture mass fields
                                counter += 1

np.save("results/map_cubed_fft",results)
