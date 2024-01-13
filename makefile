#CC = g++
#PROJECT = new_output
#SRC = aufgabe5.cpp
#LIBS = `pkg-config --cflags --libs opencv4`
#MPI_LIBS = -lmpi_cc
#$(PROJECT) : $(SRC)
#	$(CC) $(SRC) -o $(PROJECT) $(LIBS) -fopenmp	$(MPI_LIBS)

#CXX = g++
#CXXFLAGS = -Wall -Wextra -std=c++11 -I /home/dario/openmpi-4.1.6/ompi/include/


# Pfade zu den OpenCV-, OpenMP- und OpenMPI-Bibliotheken
#OPENCV_LIBS = `pkg-config --cflags --libs opencv4`
#OPENMP_LIBS = -fopenmp
#OPENMPI_LIBS = -lmpi_cxx

#all: aufgabe5

#aufgabe5: aufgabe5.cpp$(CXX) $(CXXFLAGS) -o aufgabe5 aufgabe5.cpp $(OPENCV_LIBS) $(OPENMP_LIBS) $(OPENMPI_LIBS)

#clean:
#	rm -f aufgabe5


CC = mpic++
LDFLAGS = -fopenmp -I /home/dario/openmpi-4.1.6/ompi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
LIBS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs

all: aufgabe5

aufgabe5: aufgabe5.cpp
	$(CC) $(LDFLAGS) $< -o $@ $(LIBS) `pkg-config --cflags --libs opencv4`

clean:
	rm -f aufgabe5