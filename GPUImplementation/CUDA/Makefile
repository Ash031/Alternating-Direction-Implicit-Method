INCLUDES = -I./headers/
BUILD = build/
SRC = src/
CC = nvcc
FLAGS = -Xptxas -O3 -ccbin=gcc 

compile:$(BUILD)vars.o $(BUILD)adi.o $(BUILD)Helper.o 
	$(CC) $(INCLUDES) $(FLAGS) build/*  src/main.cpp -o main

$(BUILD)vars.o:
	$(CC) $(INCLUDES) $(FLAGS) src/vars.cpp -c -o build/vars.o
$(BUILD)adi.o:
	$(CC) $(INCLUDES) $(FLAGS) src/adi.cu -c -o build/adi.o
$(BUILD)Helper.o:
	$(CC) $(INCLUDES) $(FLAGS) src/Helper.cu -c -o build/Helper.o
	
clean:
	rm ./main build/*
