INC = -I ./headers/ 
BUILD = -c -o build/
CXX = icpc

ifeq ($(CXX),icpc)
	CXXFLAGS = -O3 -std=c++11 -Wno-unused-parameter -fno-inline -qopenmp -no-multibyte-chars -qopt-report=3 -xHost   
else
	CXXFLAGS = -O3 -std=c++11 -fopt-info-vec-missed=build/report.fopt -fopenmp -Wunknown-pragmas -Wall 
endif

run: compile
	./main

compile: build/main.o build/vars.o build/info.o build/adi.o
	$(CXX) $(CXXFLAGS) $(INC) build/*.o -o main
		

build/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) $(INC) src/main.cpp $(BUILD)main.o

build/vars.o: src/vars.cpp
	$(CXX) $(CXXFLAGS) $(INC) src/vars.cpp $(BUILD)vars.o

build/info.o: src/info.cpp
	$(CXX) $(CXXFLAGS) $(INC) src/info.cpp $(BUILD)info.o

build/adi.o: src/adi.cpp
	$(CXX) $(CXXFLAGS) $(INC) src/adi.cpp $(BUILD)adi.o

clean:
	rm main build/*
