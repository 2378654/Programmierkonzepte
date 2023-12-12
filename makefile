CC = g++
PROJECT = new_output
SRC = aufgabe5.cpp
LIBS = `pkg-config --cflags --libs opencv4`
$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS) -fopenmp