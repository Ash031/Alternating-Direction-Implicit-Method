FLAGS = -ccbin=gcc -Xptxas -O3
CC = nvcc

build:
	$(CC) $(FLAGS) -L/usr/local/cuda/10.1/lib64 -I/usr/local/cuda/10.1/include -c tridiagonalSolvers.cu -o tridiagonalSolvers.o	
compare:
	$(CC) $(FLAGS) -L/usr/local/cuda/10.1/lib64 -I/usr/local/cuda/10.1/include tridiagonalSolvers.cu compare.cu -o ./main