#include "/vol/software/software/astro/cosmosis/cosmosis-1.6/cosmosis/cosmosis/datablock/datablock.hh"
#include "/vol/software/software/astro/cosmosis/cosmosis-1.6/cosmosis/cosmosis/datablock/section_names.h"
#include "helpers.cuh"
#include "bispectrum.cuh"
#include "apertureStatistics.cuh"

#include <iostream>

/*
    Interface for cosmosis
    @author Laila Linke, llinke@astro.uni-bonn.de
*/

#define VERBOSE true

extern "C"
{


/**
 * @brief Container containing parameters necessary for "execute", which are read from the "threepoint" section of the Cosmosis pipeline
 *  
 */
 class data
 {
 public:
     double zMax;                             // Maximal redshift
     std::vector<double> nz;                  //n(z) values (zBins linear spaced bins, from 0 to zMax), is assigned from file
     std::vector<double> thetas_Map2;         // thetas for which Map2 should be calculated, shape: N [rad]
     std::vector<std::vector<double>> thetas_Map3; // thetas, for which Map3 should be calculated, shape: (N, 3) [rad]
     double shapenoise_powerspectrum;         // shapenoise contribution of power spectrum (relevant for Map^2)
     bool calculate_Map2,calculate_Map3;      // if true: calculate <Map^2/3> for specified thetas
     bool output_Map2,output_Map3;            // if true: Write computed <Map^2/3> into a file in output_dir
     std::string output_dir;                  // Outputdirectory, into which computed <Map^2/3> are written, if output==True
     data(){};                                // Empty constructor
 };


// Macro to catch Cosmosis errors in runtime
#define COSMOSIS_SAFE_CALL(ans)                    \
    {                                              \
        cosmosisAssert((ans), __FILE__, __LINE__); \
    }
    /**
 * @brief Assert function for cosmosis, catching runtime errors
 * 
 * @param status Cosmosisstatus, can be various values, if this is 0 or DBS_SUCCESS, everything is fine!
 * @param file Src-Code file in which assert is triggered
 * @param line Srd-Code line in which assert is triggered
 * @param abort If true: Code execution will be stopped after assert. Optional, default: True
 */
    inline void cosmosisAssert(DATABLOCK_STATUS status, const char *file, int line, bool abort = true)
    {
        if (status != 0)
        {
            fprintf(stderr, "cosmosisAssert: %s %s %d\n", datablock_status_str(status), file, line);
            if (abort)
                exit(status);
        };
    }


/**
 * @brief Setup function for cosmosis module
 * This function does the following:
 * 1. Read in n(z) from file "nz_file" specified in the "threepoint" section of the pipeline (gets normalized and converted to our binning)
 * 2. Sets the aperture radii for which <Map^3> is computed (from "theta_min", "theta_max", "theta_bins" and "diag_only" in pipeline)
 * 3. Specifies, if and where <Map³> should be written to file (from "output" and "output_dir" in pipeline)
 * 4. Copies some constants to the GPU
 * @param options Datablock containing everything given in pipeline.ini
 * @return void* Object from type data, which contains all the information needed for execute
 */
void* setup(cosmosis::DataBlock *options)
{
    // Initialize data
    data *config_data = new data;       //Object from type data which will contain information needed for execute
    std::string section = "threepoint"; //Section in pipeline corresponding to this module

    bool quiet;
    COSMOSIS_SAFE_CALL(options->get_val(std::string("pipeline"),"quiet",quiet));
    set_verbose(!quiet);
    #if VERBOSE
            std::cerr << "Setting up threepoint" << std::endl;
    #endif // VERBOSE
    
    // Read in n(z)
    std::vector<double> nz;                                                     //Array of n(z)
    std::string fn_nz;                                                          //Filename of n(z)
    COSMOSIS_SAFE_CALL(options->get_val(section, "nz_file", fn_nz));            //Read filename of n(z)
    COSMOSIS_SAFE_CALL(options->get_val(section, "zMax", config_data->zMax));   //Read maximal z
    read_n_of_z(fn_nz, n_redshift_bins, config_data->zMax, config_data->nz); // Read in n(z) and normalize it
    #if VERBOSE
            std::cerr << "Finished reading n of z from " << fn_nz << std::endl;
            std::cerr << "Converted to " << n_redshift_bins << " redshift bins, up to " << config_data->zMax << std::endl;
    #endif //VERBOSE

    // Set thetas (including conversion to rad)
    std::string unit;                                                        // Unit in which thetas are given, can be "rad", "deg", or "arcmin"
    COSMOSIS_SAFE_CALL(options->get_val(section, "theta_unit", unit));       //Read in unit of angles
    double theta_min, theta_max;                                             // Minimal and maximal aperture radius
    COSMOSIS_SAFE_CALL(options->get_val(section, "theta_min", theta_min));   // Read in minimal aperture radius
    COSMOSIS_SAFE_CALL(options->get_val(section, "theta_max", theta_max));   // Read in maximal aperture radius
    int theta_bins;                                                          // Number of different aperture radii
    COSMOSIS_SAFE_CALL(options->get_val(section, "theta_bins", theta_bins)); // Read in number of aperture radii
    #if VERBOSE
        std::cerr << "Considering " << theta_bins << " aperture radii between "
                    << theta_min << unit << " and " << theta_max << unit << std::endl;
    #endif                                                                    //VERBOSE

    bool calculate_Map2,calculate_Map3;
    COSMOSIS_SAFE_CALL(options->get_val(section, "calculate_Map2", calculate_Map2));
    COSMOSIS_SAFE_CALL(options->get_val(section, "calculate_Map3", calculate_Map3));

    bool diag;                                                        // If true: only do calculation for equal aperture radii (theta1=theta2=theta3)
    COSMOSIS_SAFE_CALL(options->get_val(section, "diag_only", diag)); // Read in if calculation should be only for equal aperture radii

    theta_min = convert_angle_to_rad(theta_min, unit);                         //Convert minimal aperture radius to rad
    theta_max = convert_angle_to_rad(theta_max, unit);                         //Convert maximal aperture radius to rad
    double delta_theta = (log(theta_max) - log(theta_min)) / (theta_bins - 1); //Logarithmic step between aperture radii

    if(calculate_Map2)
    {
        #if VERBOSE
            std::cerr << "Calculating Map2" << std::endl;
        #endif //VERBOSE

        double galaxy_density,shapenoise_single_galaxy,powerspectrum_contribution;
        COSMOSIS_SAFE_CALL(options->get_val(section, "galaxy_shapenoise", shapenoise_single_galaxy));
        COSMOSIS_SAFE_CALL(options->get_val(section, "galaxy_density", galaxy_density));
        galaxy_density = galaxy_density / pow(convert_angle_to_rad(1., "arcmin"),2);
        powerspectrum_contribution = shapenoise_single_galaxy*shapenoise_single_galaxy/2./galaxy_density;
        config_data->shapenoise_powerspectrum = powerspectrum_contribution;

        #if VERBOSE
            std::cerr << "galaxy density: " << galaxy_density << "rad^-2" << std::endl;
            std::cerr << "powerspectrum_contribution: " << config_data->shapenoise_powerspectrum << std::endl;
        #endif //VERBOSE


        // Set aperture radii
        for (int i = 0; i < theta_bins; i++)
        {
            double theta1 = theta_min * exp(i * delta_theta);
            config_data->thetas_Map2.push_back(theta1);
        }
    };

    if(calculate_Map3)
    {

        #if VERBOSE
            std::cerr << "Calculating Map3" << std::endl;
            if (diag)
            {
                std::cerr << "Looking only at equal aperture radii" << std::endl;
            }
            else
            {
                std::cerr << "Looking at all independent combinations of aperture radii" << std::endl;
            };
        #endif //VERBOSE

        // Set aperture radii
        for (int i = 0; i < theta_bins; i++)
        {
            double theta1 = theta_min * exp(i * delta_theta);
            if (diag) //If true: set theta2=theta3=theta1
            {
                std::vector<double> thetas = {theta1, theta1, theta1};
                config_data->thetas_Map3.push_back(thetas);
            }
            else //Otherwise: Assign all independen combinations
            {
                for (int j = i; j < theta_bins; j++)
                {
                    double theta2 = theta_min * exp(j * delta_theta);
                    for (int k = j; k < theta_bins; k++)
                    {
                        double theta3 = theta_min * exp(k * delta_theta);
                        std::vector<double> thetas = {theta1, theta2, theta3};
                        config_data->thetas_Map3.push_back(thetas);
                    }
                }
            }
        }
    };

    // Set output of <Map^3>
    COSMOSIS_SAFE_CALL(options->get_val(section, "output_Map2", config_data->output_Map2));    //Read in if <Map³> should be put out
    COSMOSIS_SAFE_CALL(options->get_val(section, "output_Map3", config_data->output_Map3));    //Read in if <Map³> should be put out
    COSMOSIS_SAFE_CALL(options->get_val(section, "output_dir", config_data->output_dir)); //Read in directory into which <Map^3> should be put out
    
    // Copy constants to GPU
    copyConstants();     
    
       
       
       
       
       
    return (void *)config_data;
}




DATABLOCK_STATUS execute(cosmosis::DataBlock *block, void *config)
{
       // Get Data
       data *config_data = (data *)config;

       // Load cosmological parameters from the datablock
       cosmology cosmo; // model cosmology

       std::string section_cosmology = "cosmological_parameters";                                   // Section in pipeline containing cosmo parameters
       COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("omega_m"), cosmo.om));     // Read in Omega_m
       COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("sigma_8"), cosmo.sigma8)); // Read in sigma_8
       COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("h0"), cosmo.h));           // Read in h
       COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("omega_b"), cosmo.omb));    // Read in Omega_b
       COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("n_s"), cosmo.ns));         // Read in n_s
       COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("w"), cosmo.w));            // Read in w

       cosmo.omc = cosmo.om - cosmo.omb; // Omega_cdm, this is Omega_m - Omega_b
       cosmo.ow = 1 - cosmo.om;          // Omega_lambda, for flat universes, this is 1-Omega_m
       cosmo.zmax = config_data->zMax;

       // Load power spectrum from the datablock
       std::vector<double> k;

       COSMOSIS_SAFE_CALL(block->get_val("matter_power_lin", "k_h", k));

    std::map<double, double> linearPk;
    {
       std::vector<size_t> extents_Pk;
       COSMOSIS_SAFE_CALL(block->get_array_shape<double>(std::string("matter_power_lin"), std::string("p_k"), extents_Pk));

       std::vector<double> P_k_vals;
       cosmosis::ndarray<double> P_k(P_k_vals, extents_Pk);

       COSMOSIS_SAFE_CALL(block->get_val("matter_power_lin", "p_k", P_k));


       for (unsigned int i = 0; i < k.size(); i++)
       {
           linearPk[k[i]] = P_k(uint(0), i);
       };
    }
    std::vector<double> Pk_lin;
    double kMin, kMax, dk;
    convert_Pk(linearPk, n_kbins, kMin, kMax, dk, Pk_lin);

    // Set Cosmology and compute non-linear scales of bispectrum
    set_cosmology(cosmo, &config_data->nz, &Pk_lin, dk, kMin, kMax);

    // Calculation of Map2
    if(config_data->calculate_Map2)
    {
        // Open output file, if wanted
        std::ofstream out;       // Outputstream for the currently calculated <Map^3>
        if (config_data->output_Map2) // Check if output should be written to file
        {
            std::string fn_out = config_data->output_dir + "/Map2_om_" + std::to_string(cosmo.om) + "_sig8_" + std::to_string(cosmo.sigma8) + "_h_" + std::to_string(cosmo.h) + "_w_" + std::to_string(cosmo.w) + ".dat"; // Outputfilename, contains current cosmology

            out.open(fn_out);   // Open filestream
            if (!out.is_open()) //Check if file could be opened, exits otherwise
            {
                std::cerr << "Could not open " << fn_out << std::endl;
                exit(1);
            };
            out << "#theta[rad] Map^2" << std::endl; // First line in file
        };

        // Calculate model aperture statistics
        int N = config_data->thetas_Map2.size(); // Number of aperture radii combinations
        std::vector<double> Map2s;          // Array containing calculated <Map^3>s
    #if VERBOSE
        std::cerr << "Started <MapMap> Calculation" << std::endl;
    #endif //VERBOSE

        for (int i = 0; i < N; i++)
        {
        #if VERBOSE
            //Progress for the impatient user (Thetas in arcmin)
            std::cerr << i + 1 << "/" << N << ": Theta:"
                    << convert_rad_to_angle(config_data->thetas_Map2[i]) << " arcmin \r";
            std::cerr.flush();
        #endif                                                                          //VERBOSE
            double Map2_now = Map2(config_data->thetas_Map2[i],config_data->shapenoise_powerspectrum); //Do calculation
            Map2s.push_back(Map2_now);                                              // Store calculation in array
            if (config_data->output_Map2)                                            // Write to file, if wanted
            {
                out << config_data->thetas_Map2[i] << " "
                    << Map2_now << std::endl;
            };
        };

        COSMOSIS_SAFE_CALL(block->put_val("threepoint", "Map2s", Map2s)); // Store the <Map^2>s in the DataBlock
    };

    if(config_data->calculate_Map3)
    {

        // Open output file, if wanted
        std::ofstream out;       // Outputstream for the currently calculated <Map^3>
        if (config_data->output_Map3) // Check if output should be written to file
        {
            std::string fn_out = config_data->output_dir + "/Map3_om_" + std::to_string(cosmo.om) + "_sig8_" + std::to_string(cosmo.sigma8) + "_h_" + std::to_string(cosmo.h) + "_w_" + std::to_string(cosmo.w) + ".dat"; // Outputfilename, contains current cosmology

            out.open(fn_out);   // Open filestream
            if (!out.is_open()) //Check if file could be opened, exits otherwise
            {
                std::cerr << "Could not open " << fn_out << std::endl;
                exit(1);
            };
            out << "#theta1[rad] theta2[rad] theta3[rad] Map^3" << std::endl; // First line in file
        };

        // Calculate model aperture statistics
        int N = config_data->thetas_Map3.size(); // Number of aperture radii combinations
        std::vector<double> Map3s;          // Array containing calculated <Map^3>s
    #if VERBOSE
        std::cerr << "Started <MapMapMap> Calculation" << std::endl;
    #endif //VERBOSE

        for (int i = 0; i < N; i++)
        {
    #if VERBOSE
            //Progress for the impatient user (Thetas in arcmin)
            std::cerr << i + 1 << "/" << N << ": Thetas:"
                    << convert_rad_to_angle(config_data->thetas_Map3[i][0]) << " arcmin "
                    << convert_rad_to_angle(config_data->thetas_Map3[i][1]) << " arcmin "
                    << convert_rad_to_angle(config_data->thetas_Map3[i][2]) << " arcmin \r";
            std::cerr.flush();
    #endif                                                                          //VERBOSE
            double Map3 = MapMapMap(config_data->thetas_Map3[i]); //Do calculation
            Map3s.push_back(Map3);                                              // Store calculation in array
            if (config_data->output_Map3)                                            // Write to file, if wanted
            {
                out << config_data->thetas_Map3[i][0] << " "
                    << config_data->thetas_Map3[i][1] << " "
                    << config_data->thetas_Map3[i][2] << " "
                    << Map3 << std::endl;
            };
        };

        COSMOSIS_SAFE_CALL(block->put_val("threepoint", "Map3s", Map3s)); // Store the <Map^3>s in the DataBlock
    };

    DATABLOCK_STATUS status = DBS_SUCCESS; // Success :)
    return status;
}


/**
 * @brief Cleanup function for cosmosis module. Deletes the data object allocated in setup.
 * 
 * @param config data object that is deleted
 */
 void cleanup(void *config)
 {
     data *config_data = (data *)config;
     delete config_data;
 }


} //extern "C"