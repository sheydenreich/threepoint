from aperture_mass_computer import Map3_Gaussian_Random_Field_parallelised
from scipy.interpolate import interp1d
import numpy as np
import argparse
from os.path import exists
from os import makedirs
docstring = """ Script for creating Gaussian Random Fields and extracting the Map3 from them.
"""


# CLI parsing
parser = argparse.ArgumentParser(
    description=docstring)

parser.add_argument(
    '--npix', default=4096, metavar='INT', type=int,
    help='Number of pixels in the aperture mass map. default: %(default)s'
)

parser.add_argument(
    '--fieldsize', default=10, metavar='FLOAT', type=float,
    help='Sidelength of the field in degrees. default: %(default)s'
)

parser.add_argument(
    '--calculate_mcross', action='store_true',
    help='Also compute the cross-aperture statistics. default: %(default)s'
)

parser.add_argument(
    '--power_spectrum', default=0, metavar='INT', type=int,
    help='Type of power spectrum used. \n -1:\t power spectrum from input file \n 0:\t constant\n 1:\t (x/1e4)^2*exp(-(x/1e4)^2)\n 2:\t (x/1e4)*exp(-(x/1e4))\n default: %(default)s'
)

parser.add_argument(
    '--power_spectrum_filename',
    help='if power_spectrum=-1, filename for the power spectrum'
)

parser.add_argument(
    '--processes', default=64, metavar='INT', type=int,
    help='Number of processes for parallel computation. default: %(default)s'
)

parser.add_argument(
    '--realisations', default=1024, metavar='INT', type=int,
    help='Number of realisations computed. default: %(default)s'
)

parser.add_argument(
    '--savepath', default="", help="Outputpath"
)

parser.add_argument(
    '--cutOutFromBiggerField', action='store_true',
    help='Cut out the random fields from a random field with size 10x field_size.'
)

args = parser.parse_args()


if(__name__ == '__main__'):

    # Fieldsize in Radians
    fieldsize_rad = args.fieldsize*np.pi/180


    # Decide which powerspectrum to use
    if(args.power_spectrum < 0):
        print("Using power spectrum from input file: ",
              args.power_spectrum_filename)
        power_spectrum_array = np.loadtxt(args.power_spectrum_filename)

        power_spectrum = interp1d(
            power_spectrum_array[:, 0], power_spectrum_array[:, 1], fill_value=0)

    if(args.power_spectrum == 0):
        print("Using constant powerspectrum")

        def power_spectrum(x):
            return 0.3**2/(2.*args.npix**2/fieldsize_rad**2)*np.ones(x.shape)

    if(args.power_spectrum == 1):
        print("Using (x/1e4)^2*exp(-(x/1e4)^2) powerspectrum")

        def power_spectrum(x):
            return (x/10000)**2*np.exp(-(x/10000)**2)

    if(args.power_spectrum == 2):
        print("Using (x/1e4)*exp(-(x/1e4)) powerspectrum")

        def power_spectrum(x):
            return x/10000*np.exp(-x/10000)

    # Do Calculation
    result = Map3_Gaussian_Random_Field_parallelised(power_spectrum_array, thetas=[2, 4, 8, 16], npix=args.npix,
                                                     fieldsize=fieldsize_rad, n_realisations=args.realisations, 
                                                     n_processes=args.processes, cutOutFromBiggerField=args.cutOutFrOmBiggerField)


    # Save results
    savepath=args.savepath
           
    if not exists(savepath):
        makedirs(savepath)
    savename = 'npix_'+str(args.npix)+'_fieldsize_'+str(np.int(np.round(args.fieldsize)))
    if(args.substract_mean):
        savename += '_mean_substracted'
    np.save(savepath+'map_cubed_'+savename,result)