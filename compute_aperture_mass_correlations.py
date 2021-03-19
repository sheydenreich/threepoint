import numpy as np
from compute_aperture_masses import progressBar

# results = np.zeros((7,7,7,8,64))
# theta_ap = [0.5,1,2,4,8,16,32]
# counter = 0
# for los in range(64):
#         for i in range(7):
#                 try:
#                         field1,error1 = np.load("/vol/euclid7/euclid7_2/sven/maps_millennium/theta_"+str(theta_ap[i])+"_los_"+str(los)+".npy")
#                 except Exception as inst:
#                         print(inst)
#                         break
#                 for j in range(7):
#                         try:
#                                 field2,error2 = np.load("/vol/euclid7/euclid7_2/sven/maps_millennium/theta_"+str(theta_ap[j])+"_los_"+str(los)+".npy")
#                         except Exception as inst:
#                                 print(inst)
#                                 break
#                         for k in range(7):
#                                 try:
#                                         field3,error3 = np.load("/vol/euclid7/euclid7_2/sven/maps_millennium/theta_"+str(theta_ap[k])+"_los_"+str(los)+".npy") 
#                                 except Exception as inst:
#                                         print(inst)
#                                         break
                       
#                                 progressBar("Reading fields",counter,64*7**3)
#                                 maxtheta = theta_ap[np.max([i,j,k])]
#                                 index_maxtheta = int(maxtheta/(4*60)*4096)*2 #take double the aperture radius and cut it off
#                                 field1_cut = field1[index_maxtheta:(4096-index_maxtheta),index_maxtheta:(4096-index_maxtheta)]
#                                 field2_cut = field2[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]
#                                 field3_cut = field3[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]
#                                 error1_cut = error1[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]
#                                 error2_cut = error2[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]
#                                 error3_cut = error3[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]



#                                 results[i,j,k,0,los] = np.mean(field1_cut*field2_cut*field3_cut)
#                                 results[i,j,k,1,los] = np.mean(field1_cut*field2_cut*error3_cut)
#                                 results[i,j,k,2,los] = np.mean(field1_cut*error2_cut*field3_cut)
#                                 results[i,j,k,3,los] = np.mean(error1_cut*field2_cut*field3_cut)
#                                 results[i,j,k,4,los] = np.mean(error1_cut*error2_cut*field3_cut)
#                                 results[i,j,k,5,los] = np.mean(error1_cut*field2_cut*error3_cut)
#                                 results[i,j,k,6,los] = np.mean(field1_cut*error2_cut*error3_cut)
#                                 results[i,j,k,7,los] = np.mean(error1_cut*error2_cut*error3_cut)

#                                 counter += 1

# results *= (4*60/4096)**6 #account for incorrect normalization of aperture mass fields
# np.save("results/map_cubed_fft",results)


results = np.zeros((7,7,7,8,512))
theta_ap = [0.5,1,2,4,8,16,32]
counter = 0
for los in range(512):
        for i in range(7):
                try:
                        field1,error1 = np.load("/vol/euclid7/euclid7_2/sven/maps_slics/theta_"+str(theta_ap[i])+"_los_"+str(los+74)+".npy")
                except Exception as inst:
                        print(inst)
                        break
                for j in range(7):
                        try:
                                field2,error2 = np.load("/vol/euclid7/euclid7_2/sven/maps_slics/theta_"+str(theta_ap[j])+"_los_"+str(los+74)+".npy")
                        except Exception as inst:
                                print(inst)
                                break
                        for k in range(7):
                                try:
                                        field3,error3 = np.load("/vol/euclid7/euclid7_2/sven/maps_slics/theta_"+str(theta_ap[k])+"_los_"+str(los+74)+".npy") 
                                except Exception as inst:
                                        print(inst)
                                        break
                       
                                progressBar("Reading fields",counter,512*7**3)
                                maxtheta = theta_ap[np.max([i,j,k])]
                                index_maxtheta = int(maxtheta/(10*60)*4096)*2 #take double the aperture radius and cut it off
                                field1_cut = field1[index_maxtheta:(4096-index_maxtheta),index_maxtheta:(4096-index_maxtheta)]
                                field2_cut = field2[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]
                                field3_cut = field3[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]
                                error1_cut = error1[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]
                                error2_cut = error2[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]
                                error3_cut = error3[index_maxtheta:4096-index_maxtheta,index_maxtheta:4096-index_maxtheta]



                                results[i,j,k,0,los] = np.mean(field1_cut*field2_cut*field3_cut)
                                results[i,j,k,1,los] = np.mean(field1_cut*field2_cut*error3_cut)
                                results[i,j,k,2,los] = np.mean(field1_cut*error2_cut*field3_cut)
                                results[i,j,k,3,los] = np.mean(error1_cut*field2_cut*field3_cut)
                                results[i,j,k,4,los] = np.mean(error1_cut*error2_cut*field3_cut)
                                results[i,j,k,5,los] = np.mean(error1_cut*field2_cut*error3_cut)
                                results[i,j,k,6,los] = np.mean(field1_cut*error2_cut*error3_cut)
                                results[i,j,k,7,los] = np.mean(error1_cut*error2_cut*error3_cut)

                                counter += 1

results *= (10*60/4096)**6 #account for incorrect normalization of aperture mass fields
np.save("results/map_cubed_fft_slics",results)
