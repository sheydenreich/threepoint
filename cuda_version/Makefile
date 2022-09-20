# define the Compiler to use
# NVCC = /usr/local/cuda-11.6/bin/nvcc -std=c++17 -rdc=true -Xcompiler -fopenmp -O3 -arch=sm_86
NVCC = nvcc -std=c++11 -rdc=true -Xcompiler -fopenmp -O3 -arch=sm_75

# define the C++ compiler to use
GCC = g++

# define various compiler flags
CFLAGS = -std=c++17 -fopenmp -W -Wall -Wpointer-arith -Wformat-nonliteral -Winit-self -g -fPIC 

NVCCFLAGS = -g -Xcompiler -fPIC

# define library paths in addition to /usr/lib 
# Needed: Cosmosis and Cuba!
#LFLAGS= -L /vol/software/software/astro/cosmosis/cosmosis-1.6/cosmosis/cosmosis/datablock/ -Xcompiler \"-Wl,-R,/vol/software/software/astro/cosmosis/cosmosis-1.6/cosmosis/cosmosis/datablock/\"  -L /usr/local/lib/ -L ../Cuba-4.2.2/
LFLAGS= -L /home/laila/cosmosis/cosmosis/datablock/ -Xcompiler \"-Wl,-R,/home/laila/cosmosis/cosmosis/datablock/\"  -L /usr/local/lib/ -L /usr/local/Cuba-4.2.2/

# define any libraries
LIBS = -lm -lgsl -lgslcblas -lcosmosis -lcuba

# define Path for includes
INCLUDES = -I /home/laila/cosmosis/cosmosis/datablock/
#INCLUDES = -I /vol/software/software/astro/cosmosis/cosmosis-1.6/cosmosis/cosmosis/datablock/
# define the cuda source files
SRCS_CU = $(wildcard *.cu)

# define c++ source filese
SRCS_CPP = $(wildcard *.cpp)

# define the CUDA executables (ADD ALL HERE)
EXECS_CU = calculateApertureStatistics.x calculateApertureStatisticsCovariance.x calculateGamma.x calculateMap2Covariance.x calculateHMF.x calculateMap4.x calculateMap6.x calculateNFW.x calculatePentaspectrum_halomodel.x calculatePowerspectrum_halomodel.x calculatePowerspectrum.x calculateSecondOrderAperturestatistics.x calculateTrispectrum_halomodel.x calculateMap2Map3Covariance.x


# define cpp object files
OBJS_CPP = $(SRCS_CPP:.cpp=.o)

# define the cuda object files
OBJS_CU = $(SRCS_CU:.cu=.obj)

# define the dependencies, created in first compilation
# These files tell the compiler on which header files a file depends on
DEPS = $(OBJS:%.o=%.d)


# First make target, will be compiled, if "make" is called without arguments
all: threepoint_gpu.so $(EXECS_CU)
	@echo Successfully compiled whole project

threepoint_gpu.so: $(OBJS_CU) $(OBJS_CPP)
	$(NVCC) $(INCLUDES) $(filter-out $(EXECS_CU:.x=.obj), $(OBJS_CU) $(OBJS_CPP)) -shared -o $@ $(LFLAGS) $(LIBS)
	mv threepoint_gpu.so ../cosmosis/

# Linking of all object Files to create the execs
$(EXECS_CU): $(OBJS_CU) $(OBJS_CPP)
	$(NVCC) $(INCLUDES) $(filter-out $(EXECS_CU:.x=.obj),  $(OBJS_CU)) $(OBJS_CPP) $(@:.x=.obj) -o $@ $(LFLAGS) $(LIBS)


# Build target for every single object file
%.obj: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@ $(LFLAGS) $(LIBS)

%.o: %.cpp
	$(GCC) $(CFLAGS) $(INCLUDES)  -c $< -o $@ $(LIBS)


# Cleaning of directory, removes .o, .x, and .d files
clean:
	rm -f *.o *.x *.d *~ *.obj '#*'
