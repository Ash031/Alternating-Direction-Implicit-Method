CC = g++
SRC = src
BUILD = build
BIN = bin
KERNELS = kernels
FLAGS = -O3
OCLLIB = /usr/lib/x86_64-linux-gnu/libOpenCL.so

# -L/home/ash/intel/system_studio_2020/opencl/SDK/lib64

compile: $(BUILD)/ADI.o $(BUILD)/Vars.o $(BUILD)/OpenCLHelper.o
	$(CC) $(BUILD)/* $(FLAGS) $(SRC)/main.cpp $(OCLLIB) -o $(BIN)/main
	cp kernels/* bin/

$(BUILD)/ADI.o:$(SRC)/ADI.cpp
	$(CC) $(FLAGS) -c -o $(BUILD)/ADI.o $(SRC)/ADI.cpp
$(BUILD)/Vars.o:$(SRC)/Vars.cpp
	$(CC) $(FLAGS) -c -o $(BUILD)/Vars.o $(SRC)/Vars.cpp
$(BUILD)/OpenCLHelper.o:$(SRC)/OpenCLHelper.cpp
	$(CC) $(FLAGS) -c -o $(BUILD)/OpenCLHelper.o $(SRC)/OpenCLHelper.cpp

testTridiagonal:
	$(CC)  $(FLAGS) $(SRC)/main.cpp $(SRC)/tridiagonals.cpp $(SRC)/OpenCLHelper.cpp $(OCLLIB) -o $(BIN)/testTridiagonals

clean:
	rm build/*
	rm bin/main

run:
	$(BIN)/main