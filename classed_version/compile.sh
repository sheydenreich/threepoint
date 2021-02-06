g++ test_bispectrum.cpp -c -lm -lgsl -lgslcblas -fopenmp
g++ bispectrum.cpp -c -lm -lgsl -lgslcblas
g++ gamma.cpp -c -lm -lgsl -lgslcblas -fopenmp
# g++ Levin.cpp -c -lm

g++ test_bispectrum.o bispectrum.o gamma.o Levin.o -lm -lgsl -lgslcblas -fopenmp
