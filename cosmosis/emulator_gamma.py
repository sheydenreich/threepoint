from cosmosis.datablock import names, option_section
# import GPR_Classes_new
import numpy as np
import sys

sys.path.append('/users/sven/Documents/code/EmulateLSS/')
from emulatorpy2 import Emulator

# import sys

# print(sys.version_info)

# import subprocess

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install("pickle")
# install("_pickle")


# from astropy.io import ascii
# import pickle

# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.
cosmo = names.cosmological_parameters
emulator_path = 'emulator/'

def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.

    #Use this syntax to get a single parameter from the ini file section
    #for this module.  There is no type checking here - you get whatever the user
    #put in.

    mode = options[option_section, "mode"] # e.g. Peaks-DESY1, DSS, tomo bin, etc...

    print("-- Initializing the Emulator with mode:")
    print(mode)

    emu = Emulator(emulator_path+mode)
    # Filter = np.str(Filter_r)

    #The call above will crash if "mode" is not found in the ini file.
    #Sometimes you want a default if nothing is found:
    #high_accuracy = options.get(option_section, "high_accuracy", default=False)

    # Maybe specify the input parameter here? Maybe do more?        
    #Whatever you return here will be saved by the system and the function below
    #will get it back as 'config'.  You could return 0 if you won't need anything.
    return emu   # 0 #loaded_data


def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any 
    #earlier modules, and the config is what we loaded earlier.

    # Just a simple rename for clarity.
    emu = config

    #------
    #This loads values from the section "cosmological_parameters" that we read above.
    omega_m = block[cosmo, "omega_m"]
    h0 = block[cosmo, "h0"]
    sigma8 = block[cosmo, "sigma8_input"]
    w = block[cosmo, "w"]

    # The emulator expects [omegam S8 h w0]
    S8 = sigma8*np.sqrt(omega_m/0.3)
    Trial_Nodes =  np.array([omega_m, S8, h0, w])

    model_datavector = emu(Trial_Nodes)[1]
    # print(model_datavector)
    # Now we have got a result we save it back to the block like this.
    # I should probably create a new pblock for this...
    #block[cosmo, "PeakCount"] = GP_Pred[:][0]

    # block[mode, "principal_components"] = model_datavector
    block["threepoint","pc_gamma"] = model_datavector
    # print("Done GPR execute") 

    #We tell CosmoSIS that everything went fine by returning zero
    return 0

