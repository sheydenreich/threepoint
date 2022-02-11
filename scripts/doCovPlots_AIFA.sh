# Executes all Covariance Plots with filepaths specified to AIFA system

# First: Shapenoise only

# DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/GaussianRandomFields_shapenoise/

# python3 ../python_scripts/plotCov_diffCmeasCmodel.py --cov_type shapenoise --sigma 0.3 --dir $DIR
# python3 ../python_scripts/plotCov_diffCmeasCmodelUncertainty.py --cov_type shapenoise --sigma 0.3 --dir $DIR
# python3 ../python_scripts/plotCov_diffT1T1inf.py --cov_type shapenoise --sigma 0.3 --dir $DIR
# python3 ../python_scripts/plotCov_heatmaps.py --sidelength 10 --cov_type shapenoise --sigma 0.3 --dir $DIR
# python3 ../python_scripts/plotCov_heatmaps.py --sidelength 15 --cov_type shapenoise --sigma 0.3 --dir $DIR
# python3 ../python_scripts/plotCov_RatioT2Cmodel.py --cov_type shapenoise --sigma 0.3 --dir $DIR
# # Second : Cosmic shear

# DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/GaussianRandomFields_cosmicShear/

# python3 ../python_scripts/plotCov_diffCmeasCmodel.py --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_diffCmeasCmodelUncertainty.py --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_diffT1T1inf.py --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_heatmaps.py --sidelength 10 --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_heatmaps.py --sidelength 15 --cov_type cosmicShear --sigma 0.0 --dir $DIR
# python3 ../python_scripts/plotCov_RatioT2Cmodel.py --cov_type cosmicShear --sigma 0.0 --dir $DIR

# Third: SLICS

DIR=/vol/euclid6/euclid6_ssd/sven/threepoint_with_laila/Map3_Covariances/SLICS/

python3 ../python_scripts/plotCov_heatmaps.py --sidelength 10 --cov_type slics --sigma 0.26 --dir $DIR
python3 ../python_scripts/plotCov_diffCmeasCmodelUncertainty.py --cov_type slics --sigma 0.26 --dir $DIR
python3 ../python_scripts/plotCov_diffCmeasCmodel.py --cov_type slics --sigma 0.26 --dir $DIR