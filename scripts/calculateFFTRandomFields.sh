#python ../python_scripts/compute_aperture_mass_correlations_of_random_fields.py --npix 4096 --fieldsize 15 --power_spectrum 0 --processes 8 --realisations 4096

# python ../python_scripts/compute_aperture_mass_correlations_of_random_fields.py --npix 4096 --fieldsize 10 --power_spectrum 0 --processes 8 --realisations 4096

# python ../python_scripts/compute_aperture_mass_correlations_of_random_fields.py --npix 4096 --fieldsize 20 --power_spectrum 0 --processes 8 --realisations 4096

# python ../python_scripts/compute_aperture_mass_correlations_of_random_fields.py --npix 4096 --fieldsize 5 --power_spectrum 0 --processes 8 --realisations 4096


#python ../python_scripts/compute_aperture_mass_correlations_of_random_fields.py --npix 4096 --fieldsize 10 --power_spectrum -1 --power_spectrum_filename ../necessary_files/p_ell_slics_shapenoise_0.37.dat --processes 6 --realisations 928

#python ../python_scripts/compute_aperture_mass_correlations_of_lognormal_fields.py --npix 4096 --fieldsize 10 --power_spectrum_filename ../necessary_files/p_ell_slics_shapenoise_0.37.dat --alpha 0.5 --processes 128 --realisations 4096

python ../python_scripts/compute_aperture_mass_correlations_of_lognormal_fields.py --npix 4096 --fieldsize 10 --power_spectrum_filename ../necessary_files/p_ell_slics_shapenoise_0.37.dat --alpha 1 --processes 256 --realisations 4096

python ../python_scripts/compute_aperture_mass_correlations_of_lognormal_fields.py --npix 4096 --fieldsize 10 --power_spectrum_filename ../necessary_files/p_ell_slics_shapenoise_0.37.dat --alpha 2 --processes 256 --realisations 4096