import numpy as np

def get_all_combis_withrepeat(zbins= ['1','2','3','4','5'],all_theta_ap= ['4','8','16','32']):
      
    n_zbins = len(zbins)
    n_theta = len(all_theta_ap)
    combis = []

    z_combis = []
    for zbin_1 in zbins:
        for zbin_2 in zbins:
                z_combis.append([zbin_1,zbin_2])

    for zbin_1 in zbins:
        for zbin_2 in zbins:
            for zbin_3 in zbins:
                z_combis.append([zbin_1,zbin_2,zbin_3])


    for z_combi in z_combis:
        if(len(z_combi)<3):
            for y1 in range(n_theta):
                theta_1 = all_theta_ap[y1]
                if(z_combi[0]<=z_combi[1]):
                    combis.append([z_combi,[theta_1,theta_1]])  
        else:
            for y1 in range(n_theta):
                theta_1 = all_theta_ap[y1]
                for y2 in range(y1,n_theta):
                    theta_2 = all_theta_ap[y2]
                    for y3 in range(y2,n_theta):
                        theta_3 = all_theta_ap[y3]

                        if((theta_1==theta_2)&(theta_2==theta_3)):
                            if((z_combi[0]<=z_combi[1])&(z_combi[1]<=z_combi[2])):
                                combis.append([z_combi,[theta_1,theta_2,theta_3]])
                            
                        if((theta_1==theta_2)&(theta_1!=theta_3)):
                            if((z_combi[0]<=z_combi[1])):
                                combis.append([z_combi,[theta_1,theta_2,theta_3]])

                        if((theta_1==theta_3)&(theta_1!=theta_2)):
                            if((z_combi[0]<=z_combi[2])):
                                combis.append([z_combi,[theta_1,theta_2,theta_3]])

                        if((theta_2==theta_3)&(theta_1!=theta_2)):
                            if((z_combi[1]<=z_combi[2])):
                                combis.append([z_combi,[theta_1,theta_2,theta_3]])

                        if((theta_2!=theta_3)&(theta_1!=theta_2)):
                            combis.append([z_combi,[theta_1,theta_2,theta_3]])
    return np.array(combis)


def get_all_combis(zbins= ['1','2','3','4','5'],all_theta_ap= ['4','8','16','32']):

    n_zbins = len(zbins)
    n_theta = len(all_theta_ap)
    combis = []

    for x1 in range(n_zbins):
        zbin_1 = str(zbins[x1])
        for x2 in range(x1,n_zbins):
            zbin_2 = str(zbins[x2])
            for y1 in range(n_theta):
                theta_1 = all_theta_ap[y1]
                combis.append([[zbin_1,zbin_2,'0'],[theta_1,theta_1,'0']])

    for x1 in range(n_zbins):
        zbin_1 = str(zbins[x1])
        for x2 in range(x1,n_zbins):
            zbin_2 = str(zbins[x2])
            for x3 in range(x2,n_zbins):
                zbin_3 = str(zbins[x3])
                for y1 in range(n_theta):
                    theta_1 = all_theta_ap[y1]
                    for y2 in range(y1,n_theta):
                        theta_2 = all_theta_ap[y2]
                        for y3 in range(y2,n_theta):
                            theta_3 = all_theta_ap[y3]
                            combis.append([[zbin_1,zbin_2,zbin_3],[theta_1,theta_2,theta_3]])
                            
    return np.array(combis)


def get_indices_wo_repeatition(combis,combis_double):

    stacked_combis = []
    stacked_combis_double = []

    for i in range(len(combis)):
        if(len(combis[i][0])==2):
            stacked_combis.append(combis[i][0][0]+combis[i][0][1]+combis[i][1][0]+combis[i][1][1])
        else:
            stacked_combis.append(combis[i][0][0]+combis[i][0][1]+combis[i][0][2]+combis[i][1][0]+combis[i][1][1]+combis[i][1][2])

    for i in range(len(combis_double)):
        if(len(combis_double[i][0])==2):
            stacked_combis_double.append(combis_double[i][0][0]+combis_double[i][0][1]+combis_double[i][1][0]+combis_double[i][1][1])
        else:
            stacked_combis_double.append(combis_double[i][0][0]+combis_double[i][0][1]+combis_double[i][0][2]+combis_double[i][1][0]+combis_double[i][1][1]+combis_double[i][1][2])


    indices_wo_repeatition = []
    for i in range(len(stacked_combis)):
        indices_wo_repeatition.append(np.where(np.array(stacked_combis_double)==np.array(stacked_combis[i]))[0][0])
    return np.array(indices_wo_repeatition)


def get_auto_tomo_indices(combis):

    indices = []
    for i in range(len(combis)):
        combi = combis[i]
        
        if(len(combi[0])==2):
            if((combi[0][0]==combi[0][1])):
                indices.append(i)
        else:
            if((combi[0][0]==combi[0][1])&(combi[0][0]==combi[0][2])):
                indices.append(i)

    return indices


def get_Map3_equi_tri_autobins_indices(combis):
    indices = []
    for i in range(len(combis)):
        combi = combis[i]
        if(combi[0][2]!='0'):
            if((combi[0][0]==combi[0][1])&(combi[0][0]==combi[0][2])):
                if((combi[1][0]==combi[1][1])&(combi[1][0]==combi[1][2])):
                    indices.append(i)
    return indices

def get_equi_tri_indices(combis):
    indices = []
    for i in range(len(combis)):
        combi = combis[i]
        if(len(combi[1])==2):
            if((combi[1][0]==combi[1][1])):
                indices.append(i)
        else:
            if((combi[1][0]==combi[1][1])&(combi[1][0]==combi[1][2])):
                indices.append(i)
    return indices

def get_selected_radii_indices(combis,all_theta_ap):
    indices = []
    for i in range(len(combis)):
        combi = combis[i]
        if(len(combi[1])==2):
            if((combi[1][0] in all_theta_ap)&(combi[1][1] in all_theta_ap)):
                indices.append(i)
        else:
            if((combi[1][0] in all_theta_ap)&(combi[1][1] in all_theta_ap)&(combi[1][2] in all_theta_ap)):
                indices.append(i)
    return indices



def get_all_equi_tri_only_at_cross_tomo_indices(combis):
    indices = []
    for i in range(len(combis)):
        combi = combis[i]
        if(len(combi[1])==2):
            if((combi[0][0]==combi[0][1])):
                indices.append(i)
            else:
                if(combi[1][0]==combi[1][1]):
                    indices.append(i)
        else:
            if((combi[0][0]==combi[0][1])&(combi[0][0]==combi[0][2])):
                indices.append(i)
            else:
                if((combi[1][0]==combi[1][1])&(combi[1][0]==combi[1][2])):
                    indices.append(i)

    return indices



def get_all_equi_tri_only_at_auto_tomo_indices(combis):
    indices = []
    for i in range(len(combis)):
        combi = combis[i]
        if(len(combi[1])==2):
            indices.append(i)
        else:
            if((combi[0][0]==combi[0][1])&(combi[0][0]==combi[0][2])):
                if((combi[1][0]==combi[1][1])&(combi[1][0]==combi[1][2])):
                    indices.append(i)
            else:
                indices.append(i)

    return indices

def get_only_equi_tri_and_only_auto_tomo_indices(combis):
    indices = []
    for i in range(len(combis)):
        combi = combis[i]
        if(len(combi[1])==2):
            if((combi[0][0]==combi[0][1])):
                if((combi[1][0]==combi[1][1])):
                    indices.append(i)
        else:
            if((combi[0][0]==combi[0][1])&(combi[0][0]==combi[0][2])):
                if((combi[1][0]==combi[1][1])&(combi[1][0]==combi[1][2])):
                    indices.append(i)
    return indices


def get_Niek_indices(combis,radii,zbin):
    indices = []
    for i in range(len(combis)):
        combi = combis[i]
        if(combi[0][2]=='0'):
            if((combi[0][0]==combi[0][1])&(combi[0][0]==zbin)):
                if((combi[1][0]==combi[1][1])&(np.isin(combi[1][0],radii))):
                    indices.append(i)
        else:
            if((combi[0][0]==combi[0][1])&(combi[0][0]==combi[0][2])&(combi[0][0]==zbin)):
                if((np.isin(combi[1][0],radii))&(np.isin(combi[1][1],radii))&(np.isin(combi[1][2],radii))):
                    indices.append(i)

    return indices


def get_sub_indices(combis):

    indices = []
    for i in range(len(combis)):
        combi = combis[i]
        if(len(combi[0])==2):
            if((combi[0][0]==combi[0][1])):
                indices.append(i)
        
    return indices


def get_specific_sbins_indices(combis,sbins):

    indices = []
    for i in range(len(combis)):
        combi = combis[i][0]
        if(len(combi)==2):
            if((combi[0] in sbins)&(combi[1] in sbins)):
                indices.append(i)
        else:
            if((combi[0] in sbins)&(combi[1] in sbins)&(combi[2] in sbins)):
                indices.append(i)

    return indices

def get_Map2_indices(combis):

    indices = []
    for i in range(len(combis)):
        combi = combis[i][0]
        if(combi[2]=='0'):
            indices.append(i)

    return indices

def get_Map3_indices(combis):

    indices = []
    for i in range(len(combis)):
        combi = combis[i][0]
        if(combi[2]!='0'):
            indices.append(i)

    return indices


def remove_corrupted_indices(combis,indices):

    indices_valid = []
    for i in indices:
        combi = combis[i]
        if(len(combi[0])==2):
            indices_valid.append(i)
        else:
            if((combi[1][0]=='4')&(combi[1][1]=='32')&(combi[1][2]=='32')):
                continue
            else:
                indices_valid.append(i)

    return np.array(indices_valid)



def get_mbias_powers(combis):

    Map2_zbins = []
    Map2_indices = []

    Map3_zbins = []
    Map3_indices = []
    for i in range(len(combis[:,0])):
        combi = combis[i,0]
        if(combi[2]=='0'):
            Map2_zbins.append(combi)
            Map2_indices.append(i)
        else:
            Map3_zbins.append(combi)
            Map3_indices.append(i)
    
    Map2_zbins = np.array(Map2_zbins)
    Map2_indices = np.array(Map2_indices)

    Map3_zbins = np.array(Map3_zbins)
    Map3_indices = np.array(Map3_indices)

    indices = []
    for zbin in ['1','2','3','4','5']:
        indices_temp = []
        if(len(Map2_indices)>0):
            indices_temp = indices_temp +  list(Map2_indices[np.where(Map2_zbins[:,0]==zbin)])
            indices_temp = indices_temp +  list(Map2_indices[np.where(Map2_zbins[:,1]==zbin)])
        if(len(Map3_indices)>0):
            indices_temp = indices_temp +  list(Map3_indices[np.where(Map3_zbins[:,0]==zbin)])
            indices_temp = indices_temp +  list(Map3_indices[np.where(Map3_zbins[:,1]==zbin)])
            indices_temp = indices_temp +  list(Map3_indices[np.where(Map3_zbins[:,2]==zbin)])

        indices.append(indices_temp)

    indices = np.array(indices)
    power_all = []
    for i in range(5):
        power_temp = []
        for j in np.arange(0,len(combis)):
            power_temp.append(np.where(indices[i]==j)[0].shape[0])
        power_all.append(power_temp)
    power_all = np.array(power_all)

    return power_all


all_theta_ap = ['4','8','16','32']
n_theta = len(all_theta_ap)
theta_size_combis = []
delete_32 = []

counter = 0
for x1 in range(n_theta):
    theta_1 = all_theta_ap[x1]
    theta_size_combis.append(theta_1+"' "+theta_1)
    counter = counter + 1

counter = 0
for x1 in range(n_theta):
    theta_1 = all_theta_ap[x1]
    for x2 in range(x1,n_theta):
        theta_2 = all_theta_ap[x2]
        for x3 in range(x2,n_theta):
            theta_3 = all_theta_ap[x3]
            theta_size_combis.append(theta_1+"' "+theta_2+"' "+theta_3+"'")
            counter = counter + 1


def get_indices_Map2_at_theta(combis,zbin,theta):
    indices = []
    for i in range(len(combis)):
        combi = combis[i]
      
        if(len(combi[0])==2):
            if((combi[0][0]==zbin)&(combi[1][0]==theta)):
                indices.append(i)
    return indices


def get_indices_Map3_at_theta(combis,zbin1,zbin2,theta1,theta2,theta3):
    indices = []
    for i in range(len(combis)):
        combi = combis[i]
        
        if(len(combi[0])>2):
            if((combi[0][0]==zbin1)&(combi[0][1]==zbin2)&(combi[1][0]==theta1)&(combi[1][1]==theta2)&(combi[1][2]==theta3)):
                indices.append(i)

    return indices