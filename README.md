# Programmierkonzepte

Um ein anderes Bild zu wechseln: Bei Zeile 25 eine der drei JPEG auswählen.

Makefile wurde benutzt, um das Program lokal auszuführen.
Dafür werden allerdings lokale Pfade verwendet, weshalb es nicht auf anderen Systemen funktioniert.
Um das Programm zu kompilieren:
#### mpic++ -fopenmp -I [MPI INCLUDE PATH] -L [MPI LIBRARY PATH] -lmpi aufgabe5.cpp -o aufgabe5 -lopencv_core -lopencv_highgui -lopencv_imgcodecs `pkg-config --cflags --libs opencv4`
