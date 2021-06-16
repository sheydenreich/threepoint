# define the Compiler to use
GCC = g++

# define various compiler flags
CFLAGS = -std=c++17 -fopenmp -W -Wall -Wpointer-arith -Wformat-nonliteral -Winit-self -g

# define library paths in addition to /usr/lib
LFLAGS= -L

# define any libraries
LIBS = -lm -lgsl -lgslcblas

# define the source files
SRCS = $(wildcard *.cpp)

# define the executables (ADD ALL EXECUTABLES HERE!!!!)
EXECS = calculateGamma.x calculateApertureStatistics.x testGamma_analytical_bispectrum.x testGamma_integrand.x testApertureStatistics_integrand.x testApertureStatistics_bispectrum.x testApertureStatistics_integral_phi.x testApertureStatistics_integral_l2.x testApertureStatistics_integral_l1.x calculateDerivativeApertureStatistics.x

# define the object files
OBJS = $(SRCS:.cpp=.o)

# define the dependencies, created in first compilation
# These files tell the compiler on which header files a file depends on
DEPS = $(OBJS:%.o=%.d)

# First make target, will be compiled, if "make" is called without arguments
all:	$(EXECS)
	@echo Succesfully compiled whole project

# Linking of all .o Files to create the EXECS
$(EXECS): $(OBJS)
	$(GCC) $(CFLAGS) $(INCLUDES) $(filter-out $(EXECS:.x=.o), $(OBJS)) $(@:.x=.o) -o $@ $(LFLAGS) $(LIBS)

# Include all .d files
-include $(DEPS)

# Build target for every single object file
# -MMD flag creates a .d file with the same name as the .o file
%.o: %.cpp
	$(GCC) $(CFLAGS) $(INCLUDES) -MMD -c $< -o $@ $(LIBS)

# Cleaning of directory, removes .o, .x, and .d files
clean:
	rm -f *.o *.x *.d
