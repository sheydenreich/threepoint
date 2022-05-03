import numpy as np
import matplotlib.pyplot as plt
import treecorr
from astropy.io import fits

bins_r = 16
bins_u = 16
bins_v = 16

min_sep = 0.1
max_sep = 150

nprocs = 250

kids_data_file = fits.open("/vol/euclid2/euclid2_raid2/sven/varying_depth_randoms/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits")

kids_data = kids_data_file[1].data

cat = treecorr.Catalog(ra=kids_data['RAJ2000'],dec=kids_data['DECJ2000'],
                       ra_units='deg',dec_units='deg',
                       g1=kids_data['e1'],
                       g2=kids_data['e2'],
                       w=kids_data['weight'])

ggg = treecorr.GGGCorrelation(nbins=bins_r,min_sep=min_sep,max_sep=max_sep,sep_units='arcmin',
        nubins=bins_u,min_u=0,max_u=1,nvbins=bins_v,min_v=0,max_v=1,verbose=2,num_threads=nprocs)

ggg.process(cat)

ggg.write("/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/results_KiDS/GGGcorrelation_16_bins_0p1_to_150_arcmin.dat")