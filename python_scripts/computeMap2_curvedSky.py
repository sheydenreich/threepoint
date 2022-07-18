from aperture_mass_computer import extract_Map2_curved_sky
import healpy as hp
import numpy as np


filepath=""
savepath=""

kappa=hp.read_map(filepath)
nside=2**12
thetas=[2,4,8,16,32]

result=extract_Map2_curved_sky(kappa,nside,thetas)

np.savetxt(savepath+"/map_squared_curvedSky.dat", result)