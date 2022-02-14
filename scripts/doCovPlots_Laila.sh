# Executes all Covariance Plots with filepaths specified to Lailas files

# # First: Shapenoise only

# DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_shapenoise/


# python3 ../python_scripts/plotCov_diffCmeasCmodel.py --cov_type shapenoise --sigma 0.3 --dir $DIR
# python3 ../python_scripts/plotCov_diffCmeasCmodelUncertainty.py --cov_type shapenoise --sigma 0.3 --dir $DIR
# python3 ../python_scripts/plotCov_diffT1T1inf.py --cov_type shapenoise --sigma 0.3 --dir $DIR
# python3 ../python_scripts/plotCov_heatmaps.py --sidelength 5 --cov_type shapenoise --sigma 0.3 --dir $DIR
# python3 ../python_scripts/plotCov_heatmaps.py --sidelength 10 --cov_type shapenoise --sigma 0.3 --dir $DIR
# python3 ../python_scripts/plotCov_heatmaps.py --sidelength 15 --cov_type shapenoise --sigma 0.3 --dir $DIR
# python3 ../python_scripts/plotCov_RatioT2Cmodel.py --cov_type shapenoise --sigma 0.3 --dir $DIR

# # # Second : Cosmic shear

# DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/GaussianRandomFields_cosmicShear/

#  python3 ../python_scripts/plotCov_diffCmeasCmodel.py --cov_type cosmicShear --sigma 0.0 --dir $DIR
#  python3 ../python_scripts/plotCov_diffCmeasCmodelUncertainty.py --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_diffT1T1inf.py --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_heatmaps.py --sidelength 5 --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_heatmaps.py --sidelength 10 --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_heatmaps.py --sidelength 15 --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_RatioT2Cmodel.py --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_CovTimesArea.py --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_CovTimesAreaRatio.py --cov_type cosmicShear --sigma 0.0 --dir $DIR

DIR=/home/laila/OneDrive/1_Work/5_Projects/02_3ptStatistics/Map3_Covariances/SLICS/

python3 ../python_scripts/plotCov_heatmaps.py --sidelength 10 --cov_type slics --sigma 0.26 --dir $DIR
python3 ../python_scripts/plotCov_diffCmeasCmodelUncertainty.py --cov_type slics --sigma 0.26 --dir $DIR
python3 ../python_scripts/plotCov_diffCmeasCmodel.py --cov_type slics --sigma 0.26 --dir $DIR