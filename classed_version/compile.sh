g++ test_bispectrum.cpp -c -lm -lgsl -lgslcblas -fopenmp
g++ bispectrum.cpp -c -lm -lgsl -lgslcblas
g++ gamma.cpp -c -lm -lgsl -lgslcblas -fopenmp
g++ ../cubature/hcubature.c -c -lm -fopenmp

g++ test_bispectrum.o bispectrum.o gamma.o hcubature.o -o tests.out -lm -lgsl -lgslcblas -fopenmp
