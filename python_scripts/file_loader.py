import numpy as np

def get_kappa_millennium(los):
    return get_millennium(los)[:,:,4]


def get_millennium(los):
    los_no1 = los//8
    los_no2 = los%8
    ms_field = np.loadtxt("/vol/euclid2/euclid2_raid2/sven/millennium_maps/41_los_8_"+ str(los_no1) +"_"+ str(los_no2) +".ascii")
    ms_field = ms_field.reshape(4096,4096,5)
    return ms_field