# define the Compiler to use
GCC = g++

# define various compiler flags
CFLAGS = -std=c++11 -fopenmp -W -Wall -Wpointer-arith -Wformat-nonliteral -Winit-self -g

# ignoring of compiler flags for certain files
Levin.o : CFLAGS+=-Wno-implicit-fallthrough

# define directories containing header files other than /usr/include
INCLUDES = -I ./

# define library paths in addition to /usr/lib
LFLAGS= -L

# define any libraries
LIBS = -lm -lgsl -lgslcblas

# define the source files
SRCS = $(wildcard *.cpp)

# define the executables (ADD ALL EXECUTABLES HERE!!!!)
EXECS = test_bispectrum.x

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
	$(GCC) $(CFLAGS) $(INCLUDES) $(OBJS) -o $@ $(LFLAGS) $(LIBS)

# Include all .d files
-include $(DEPS)

# Build target for every single object file
# -MMD flag creates a .d file with the same name as the .o file
%.o: %.cpp
	$(GCC) $(CFLAGS) -MMD -c $< -o $@ $(LIBS)

# Cleaning of directory, removes .o, .x, and .d files
clean:
	rm -f *.o *.x *.d
