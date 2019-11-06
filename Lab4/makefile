CC=nvcc
CFALGS = -Wall

lab2: $(OBJ)
	$(CC) -dc lab4_simd.cu
	$(CC) -o lab4_simd lab4_simd.o /fs/project/PAS1421/transform_cuda.o