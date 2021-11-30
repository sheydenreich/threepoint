//#include "/home/laila/cosmosis/cosmosis/datablock/c_datablock.h"
#include "/home/laila/cosmosis/cosmosis/datablock/datablock.hh"
#include "/home/laila/cosmosis/cosmosis/datablock/section_names.h"
#include "apertureStatistics.hpp"
#include "helper.hpp"
#include <vector>
#include <chrono>
#include "../Eigen/Dense"

/* 
	Interface for cosmosis
    @author Laila Linke, llinke@astro.uni-bonn.de
*/

#define VERBOSE true

extern "C"
{
class data
{
public:
    int zBins;
    double zMax;
    std::vector<double> nz;
    std::vector<std::vector<double>> thetas;
    std::vector<double> measurements;
    std::vector<std::vector<double>> inverse_covariance;

    data(){};


} ;


// Macro to catch Cosmosis errors in runtime
#define COSMOSIS_SAFE_CALL(ans) {cosmosisAssert((ans), __FILE__, __LINE__);}
inline void cosmosisAssert(DATABLOCK_STATUS status, const char* file, int line, bool abort=true)
{
    if (status != 0)
    {
        fprintf(stderr, "cosmosisAssert: %s %s %d\n", datablock_status_str(status), file, line);
        if(abort) exit(status);
    };
}


void * setup(cosmosis::DataBlock * options)
{
    // Initialize data   
    data* config_data = new data;
    std::string section = "threepoint";
#if VERBOSE
    std::cerr<<"Setting up threepoint"<<std::endl; 
#endif // VERBOSE
    // Read in n(z)

    std::vector<double> nz;
    std::string fn_nz;

    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("nz_file"), fn_nz)); 
#if VERBOSE
    std::cerr<<"Reading n(z) from "<<fn_nz<<std::endl;
#endif // VERBOSE
    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("zBins"), config_data->zBins));
    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("zMax"), config_data->zMax)); 
    
    read_n_of_z(fn_nz, config_data->zBins, config_data->zMax, config_data->nz);
#if VERBOSE 
    std::cerr<<"Finished reading n of z"<<std::endl;
    std::cerr<<"Converted to "<<config_data->zBins<<" redshift bins, up to "<<config_data->zMax<<std::endl;
#endif //VERBOSE

    // Read in thetas and measurements (include conversion to rad)  
    std::string fn_measurements;
    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("data_file"), fn_measurements));
    std::string unit;
    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("theta_unit"), unit));

#if VERBOSE
    std::cerr<<"Reading in measurements from "<<fn_measurements<<std::endl;
    std::cerr<<"Assuming angular units in "<<unit<<std::endl;
#endif
    read_measurement(fn_measurements, config_data->thetas, 
                            config_data->measurements, unit);   
#if VERBOSE 
    std::cerr<<"Finished reading of measurement"<<std::endl;
    std::cerr<<"First datapoint "<<config_data->thetas.at(0)[0]<<" "
                                <<config_data->thetas.at(0)[1]<<" "
                                <<config_data->thetas.at(0)[2]<<" "
                                <<config_data->measurements.at(0)<<std::endl;

#endif //VERBOSE

    double surveyArea;
    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("survey_area"), surveyArea));
#if VERBOSE
    std::cerr<<"Survey area:"<<surveyArea<<" "<<unit<<"^2"<<std::endl;
#endif
    
#if VERBOSE
    std::cerr<<"Started calculating covariance"<<std::endl;
#endif

    // Load fiducial cosmological parameters from the datablock for covariance calculation
    struct cosmology cosmo; // Struct containing model cosmology


    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("omega_m_fid"), cosmo.om));
    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("sigma8_fid"), cosmo.sigma8));
    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("h0_fid"), cosmo.h));
    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("omega_b_fid"), cosmo.omb));
    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("n_s_fid"), cosmo.ns));
    COSMOSIS_SAFE_CALL(options->get_val(section, std::string("w_fid"), cosmo.w));

    cosmo.omc = cosmo.om-cosmo.omb;
    cosmo.ow = 1-cosmo.om;

    // Initialize bispectrum 
    BispectrumCalculator bispectrum(cosmo, config_data->nz, config_data->zBins, config_data->zMax);

    // Initialize aperture statistics
    ApertureStatistics apertureStatistics(&bispectrum);

    int N=config_data->thetas.size();

    // Calculate Covariance of MapMapMap (using Eigen library for inversion)
 
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Covariance_eig;
    Covariance_eig.resize(N,N);

#if VERBOSE
    int step=0;
#endif //VERBOSE

    for(int i=0; i<N; i++)
    {
        for(int j=i; j<N; j++)
        {
#if VERBOSE
	    //Progress for the impatient user (Thetas in arcmin)
	    std::cerr<<step<<"/"<<N*(N+1)/2<<": Thetas:"
                        <<config_data->thetas[i][0]<<" "
                        <<config_data->thetas[i][1]<<" "
                        <<config_data->thetas[i][2]<<" "
                        <<config_data->thetas[j][0]<<" "
                        <<config_data->thetas[j][1]<<" "
                        <<config_data->thetas[j][2]<<" "
                        <<" \r";
	    std::cerr.flush();
        step++;
#endif //VERBOSE
            double Cov=apertureStatistics.MapMapMap_covariance_Gauss(config_data->thetas[i], config_data->thetas[j], surveyArea);
            Covariance_eig(i,j)=Cov;
            Covariance_eig(j,i)=Cov;
        }
    }


#if VERBOSE
    std::ofstream cov_file;
    cov_file.open("/home/laila/Coderepos/threepoint/cosmosis/Cov.dat");
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            cov_file<<Covariance_eig(i,j)<<" ";
        }
        cov_file<<std::endl;
    };
    std::cerr<<"Started inverting covariance"<<std::endl;
#endif //VERBOSE
    // Invert covariance using Eigen library
    Covariance_eig=Covariance_eig.inverse();

    // Cast inverted covariance to C++ vectors for later handling
    std::vector<std::vector<double>> inverse_Covariance(N,std::vector<double>(N));
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            inverse_Covariance[i][j]=Covariance_eig(i,j);
        }
    }

    config_data->inverse_covariance=inverse_Covariance;

 #if VERBOSE
    std::ofstream covInv_file;
    covInv_file.open("/home/laila/Coderepos/threepoint/cosmosis/Cov_Inv.dat");
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            covInv_file<<inverse_Covariance[i][j]<<" ";
        }
        covInv_file<<std::endl;
    };

    std::cerr<<"Finished setup"<<std::endl;
#endif   

    return (void*) config_data;


}


DATABLOCK_STATUS execute(cosmosis::DataBlock * block, void * config)
{
    std::chrono::steady_clock::time_point begin=std::chrono::steady_clock::now();
    // Get Data
    data * config_data= (data*) config;


    // Load cosmological parameters from the datablock
    struct cosmology cosmo; // Struct containing model cosmology

    std::string section_cosmology="cosmological_parameters";
    COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("omega_m"), cosmo.om));
    COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("sigma8"), cosmo.sigma8));
    COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("h0"), cosmo.h));
    COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("omega_b"), cosmo.omb));
    COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("n_s"), cosmo.ns));
    COSMOSIS_SAFE_CALL(block->get_val(section_cosmology, std::string("w"), cosmo.w));

    cosmo.omc = cosmo.om-cosmo.omb;
    cosmo.ow = 1-cosmo.om;

    // Initialize bispectrum 
    BispectrumCalculator bispectrum(cosmo, config_data->nz, config_data->zBins, config_data->zMax);

    // Initialize aperture statistics
    ApertureStatistics apertureStatistics(&bispectrum);

    // Calculate difference between model aperture statistics and data
    std::vector<double> Diffs;

    int N=config_data->thetas.size();

#if VERBOSE
    std::cerr<<"Started <MapMapMap> Calculation"<<std::endl;
#endif //VERBOSE

    //Calculate <MapMapMap>(theta1, theta2, theta3) 
    for (int i=0; i<N; i++)
    {
#if VERBOSE
	    //Progress for the impatient user (Thetas in arcmin)
	    // std::cerr<<i+1<<"/"<<N<<": Thetas:"
        //                 <<config_data->thetas[i][0]<<" "<<config_data->thetas[i][1]<<" "
        //                 <<config_data->thetas[i][2]<<" \r";
	    //std::cerr.flush();
#endif //VERBOSE
	    double MapMapMap=apertureStatistics.MapMapMap(config_data->thetas[i]); //Do calculation
        Diffs.push_back(MapMapMap-config_data->measurements.at(i));
#if VERBOSE
        std::cerr<<MapMapMap<<" "<<config_data->measurements.at(i)<<std::endl;
#endif
    	
    };



    // Calculate Likelihood

    double ChiSq=0;
#if VERBOSE
    std::cerr<<"Started Chi^2 Calculation"<<std::endl;
#endif //VERBOSE

    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            ChiSq+=Diffs.at(i)*config_data->inverse_covariance.at(i).at(j)*Diffs.at(j);
        }
    }

    double likelihood=-0.5*ChiSq;

    COSMOSIS_SAFE_CALL(block->put_val("LIKELIHOODS", "threepoint_like", likelihood));
    std::chrono::steady_clock::time_point end=std::chrono::steady_clock::now();

#if VERBOSE
    std::cerr<<"Needed "
            <<std::chrono::duration_cast<std::chrono::milliseconds> (end-begin).count()<<" milliseconds for one point."
            <<std::endl;
#endif //VERBOSE

	//Signal success.
    DATABLOCK_STATUS status=DBS_SUCCESS;
	return status;
}

void cleanup(void * config)
{
	// Simple tidy up - just free what we allocated in setup
	data * config_data = (data*) config;
    delete config_data;
}
}
