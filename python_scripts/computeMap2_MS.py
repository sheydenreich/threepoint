from aperture_mass_computer import Map2_MS_parallelised
from scipy.interpolate import interp1d
import numpy as np
import argparse
from os.path import exists
from os import makedirs
docstring = """ Script for extracting the Map2 from the MS.
"""


# CLI parsing
parser = argparse.ArgumentParser(
    description=docstring)

parser.add_argument(
    '--ngal', default=0, metavar='FLOAT', type=float,
    help='Galaxy number density. If 0, just use the pixel grid. default: %(default)s'
)

parser.add_argument(
    '--shapenoise', default=0, metavar='FLOAT', type=float,
    help='Shapenoise that is to be added to the shear. default: %(default)s'
)

parser.add_argument(
    '--processes', default=64, metavar='INT', type=int,
    help='Number of processes for parallel computation. default: %(default)s'
)

parser.add_argument(
    '--savepath', default="./", help="Outputpath"
)

args = parser.parse_args()


if(__name__ == '__main__'):

    if (args.ngal==0) and (args.shapenoise==0): # Without shapenoise and for pixelgrid
        result = Map2_MS_parallelised(thetas=[2,4,8,16],n_processes=args.processes)
    elif args.ngal==0: # With shapenoise but on pixel grid
        result = Map2_MS_parallelised(thetas=[2,4,8,16], shapenoise=args.shapenoise, n_processes=args.processes)
    else: # Not on pixel grid
        result = Map2_MS_parallelised(thetas=[2,4,8,16],shapenoise=args.shapenoise, numberdensity=args.ngal, n_processes=args.processes)
    
    # Save results
    savepath=args.savepath
           
    if not exists(savepath):
        makedirs(savepath)
    savename = 'ngal_'+str(args.ngal)+'_shapenoise_'+str(args.shapenoise)

    np.save(savepath+'map_squared_'+savename,result)