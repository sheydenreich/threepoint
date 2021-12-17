# define the Compiler to use
GCC = g++

# define various compiler flags
CFLAGS = -std=c++17 -fopenmp -W -Wall -Wpointer-arith -Wformat-nonliteral -Winit-self -g -fPIC

# define library paths in addition to /usr/lib
LFLAGS= -L /home/laila/cosmosis/cosmosis/datablock/ -Wl,-R /home/laila/cosmosis/cosmosis/datablock/  -L /usr/local/lib/

# define include
INCLUDES= -I /usr/local/include/

# define any libraries
LIBS = -lgsl -lgslcblas -lcosmosis -lm

# define the source files
SRCS = $(wildcard *.cpp)

# define the executables (ADD ALL EXECUTABLES HERE!!!!)
EXECS = calculateBispectrumAndCovariance.x testApertureStatistics_covariance.x calculateApertureStatisticsCovariance.x testPowerSpectrum_nonlinear.x calculateGamma.x calculateApertureStatistics.x testGamma_analytical_bispectrum.x testGamma_integrand.x testApertureStatistics_integrand.x testApertureStatistics_bispectrum.x testApertureStatistics_integral_phi.x testApertureStatistics_integral_l2.x testApertureStatistics_integral_l1.x calculateDerivativeApertureStatistics.x calculateApertureStatistics_varyingCosmos.x testCalculateCovarianceGaussianWhiteNoise.x testCalculateEquation93.x

# define the object files
OBJS = $(SRCS:.cpp=.o)

# define the dependencies, created in first compilation
# These files tell the compiler on which header files a file depends on
DEPS = $(OBJS:%.o=%.d)

# First make target, will be compiled, if "make" is called without arguments
all: threepoint_cpu.so $(EXECS)
	@echo Succesfully compiled whole project


threepoint_cpu.so: $(OBJS)
	$(GCC) $(CFLAGS) -Wl,-z,defs $(INCLUDES) $(filter-out $(EXECS:.x=.o), $(OBJS)) -shared -o $@ $(LFLAGS) $(LIBS)
	mv threepoint_cpu.so ../cosmosis/

# Linking of all .o Files to create the EXECS
$(EXECS): $(OBJS)
	$(GCC) $(CFLAGS) $(INCLUDES) $(filter-out $(EXECS:.x=.o), $(OBJS)) $(@:.x=.o) -o $@ $(LFLAGS) $(LIBS)

# Include all .d files
-include $(DEPS)

# Build target for every single object file
# -MMD flag creates a .d file with the same name as the .o file
%.o: %.cpp
	$(GCC) $(CFLAGS) $(INCLUDES) -MMD -c $< -o $@ $(LFLAGS) $(LIBS)

# Cleaning of directory, removes .o, .x, and .d files
clean:
	rm -f *.o *.x *.d


test:
	@echo "Alas, threepoint-module has no tests"