# define the Compiler to use
NVCC = nvcc -std=c++11 -rdc=true -Xcompiler -fopenmp -O3

# define the C++ compiler to use
GCC = g++

# define library paths in addition to /usr/lib
LFLAGS= -L

# define any libraries
LIBS = -lm -lgsl -lgslcblas


# define the cuda source files
SRCS_CU = $(wildcard *.cu)

# define c++ source filese
SRCS_CPP = $(wildcard *.cpp)

# define the CUDA executables (ADD ALL HERE)
EXECS_CU = calculateSecondOrderAperturestatistics.x calculatePowerspectrum.x calculatePowerspectrum_varyingcosmos.x calculateApertureStatisticsCovariance.x testPowerspectrum.x calculateApertureStatistics_varyingCosmos.x calculateApertureStatistics.x calculateDerivativeApertureStatistics.x calculateGamma_input_fisher.x calculateGamma.x testGamma.x calculateGamma_varyingCosmos.x

# define cpp object files
OBJS_CPP = $(SRCS_CPP:.cpp=.o)

# define the cuda object files
OBJS_CU = $(SRCS_CU:.cu=.obj)

# define the dependencies, created in first compilation
# These files tell the compiler on which header files a file depends on
DEPS = $(OBJS:%.o=%.d)


# First make target, will be compiled, if "make" is called without arguments
all:	$(EXECS_CU)
	@echo Successfully compiled whole project

# Linking of all object Files to create the execs
$(EXECS_CU): $(OBJS_CU) $(OBJS_CPP)
	$(NVCC) $(INCLUDES) $(filter-out $(EXECS_CU:.x=.obj),  $(OBJS_CU)) $(OBJS_CPP) $(@:.x=.obj) -o $@ $(LFLAGS) $(LIBS)


# Build target for every single object file
%.obj: %.cu
	$(NVCC) -c $< -o $@ $(LIBS)

%.o: %.cpp
	$(GCC) -c $< -o $@ $(LIBS)


# Cleaning of directory, removes .o, .x, and .d files
clean:
	rm -f *.o *.x *.d *~ *.obj '#*'
