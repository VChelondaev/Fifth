#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iomanip>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <mpi.h>

// Заводим глобальные указатели для матриц
double 	*matrixA 			= nullptr, 
		*matrixB			= nullptr,
	 	*deviceMatrixAPtr 	= nullptr, 
		*deviceMatrixBPtr	= nullptr, 
		*deviceError 		= nullptr, 
		*errorMatrix 		= nullptr, 
		*tempStorage 		= nullptr;

#define CALCULATE(matrixA, matrixB, size, i, j) \
	matrixB[i * size + j] = 0.25 * (matrixA[i * size + j - 1] + matrixA[(i - 1) * size + j] + \
			matrixA[(i + 1) * size + j] + matrixA[i * size + j + 1]);	

__global__
void calculateBoundaries(double* matrixA, double* matrixB, size_t size, size_t sizePerGpu)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0 || idx > size - 2) return;
	
	if(idx < size)
	{
		CALCULATE(matrixA, matrixB, size, 1, idx);
		CALCULATE(matrixA, matrixB, size, (sizePerGpu - 2), idx);
	}
}

// Главная функция - расчёт поля 
__global__
void calculateMatrix(double* matrixA, double* matrixB, size_t size, size_t sizePerGpu)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if((i < sizePerGpu - 2) && (i > 1) && (j > 0) && (j < size - 1))
	{
		CALCULATE(matrixA, matrixB, size, i, j);
	}
}

// Функция, подсчитывающая разницу матриц
__global__
void getErrorMatrix(double* matrixA, double* matrixB, double* outputMatrix, size_t size, size_t sizePerGpu)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

	size_t idx = i * size + j;
	if(!(j == 0 || i == 0 || j == size - 1 || i == sizePerGpu - 1))
	{
		outputMatrix[idx] = std::abs(matrixB[idx] - matrixA[idx]);
	}
}

int findNearestPowerOfTwo(size_t num) {
    int power = 1;
    while (power < num) {
        power <<= 1;
    }
    return power;
}

void print(int size, int rank, double* matrixA, int sizeOfAreaForOneProcess){
		if (rank == 0){
		printf("start rank 0\n");
		for (int i = 0; i < sizeOfAreaForOneProcess; i++){
			for (int j = 0; j < size; j++){
				printf("%0.4lf ", matrixA[i * size + j]);
			}
			printf("\n");
		}
		printf("end rank 0\n");
	}
	if (rank == 1){
		printf("start rank 1\n");
		for (int i = 0; i < sizeOfAreaForOneProcess; i++){
			for (int j = 0; j < size; j++){
				printf("%0.4lf ", matrixA[i * size + j]);
			}
			printf("\n");
		}
		printf("end rank 1\n");
	}
	if (rank == 2){
		printf("start rank 2\n");
		for (int i = 0; i < sizeOfAreaForOneProcess; i++){
			for (int j = 0; j < size; j++){
				printf("%0.4lf ", matrixA[i * size + j]);
			}
			printf("\n");
		}
		printf("end rank 2\n");
	}

	if (rank == 3){
		printf("start rank 3\n");
		for (int i = 0; i < sizeOfAreaForOneProcess; i++){
			for (int j = 0; j < size; j++){
				printf("%0.4lf ", matrixA[i * size + j]);
			}
			printf("\n");
		}
		printf("end rank 3\n");
	}

}

int main(int argc, char** argv)
{

	if (argc != 4)
	{
		std::cout << "Invalid parameters" << std::endl;
		std::exit(-1);
	}

	// Получаем значения из командной строки
	const double minError = std::atof(argv[1]);
	const int size = std::atoi(argv[2]);
	const int maxIter = std::atoi(argv[3]);
	const size_t totalSize = size * size;

	int rank, sizeOfTheGroup;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfTheGroup);

	cudaSetDevice(rank);

	// Размечаем границы между устройствами

	size_t sizeOfAreaForOneProcess; 
	size_t startYIdx;

	if (size % sizeOfTheGroup == 0) {
		sizeOfAreaForOneProcess = size / sizeOfTheGroup;

	}
	else {
		if (sizeOfTheGroup == 2) {
			if (size % 2 != 0){
				sizeOfAreaForOneProcess = std::ceil(size / (sizeOfTheGroup)) + 1;

				if (rank == (sizeOfTheGroup - 1)) sizeOfAreaForOneProcess = (size / (sizeOfTheGroup));
				}
			else {
				sizeOfAreaForOneProcess = size / (sizeOfTheGroup);
			}
		}
		if (sizeOfTheGroup == 4) {
			sizeOfAreaForOneProcess = size / (sizeOfTheGroup - 1);

			if (rank == (sizeOfTheGroup - 1)) sizeOfAreaForOneProcess = size % (sizeOfTheGroup - 1);
		}
	}

	
	if (sizeOfTheGroup == 2){ 
		if (size % 2 == 0){
			startYIdx = (size / sizeOfTheGroup) * rank;
		}
		else {
		startYIdx = (size / sizeOfTheGroup) * rank;
		if (rank == 1) startYIdx++;
		}
	}
	if (sizeOfTheGroup == 4) {	
		if (size % sizeOfTheGroup == 0) startYIdx = (size / (sizeOfTheGroup)) * rank;
		else {startYIdx = (size / (sizeOfTheGroup - 1)) * rank;}
		}
	if (sizeOfTheGroup == 1) {startYIdx = 0;}

	//printf("Size of group: %d \n", sizeOfTheGroup);

	
	
	if (rank == 0) printf("rank = %d %d\n", rank, sizeOfAreaForOneProcess);
	if (rank == 1) printf("rank = %d %d\n", rank, sizeOfAreaForOneProcess);
	if (rank == 2) printf("rank = %d %d\n", rank, sizeOfAreaForOneProcess);
	if (rank == 3) printf("rank = %d %d\n", rank, sizeOfAreaForOneProcess);

	// Выделение памяти на хосте
    cudaMallocHost(&matrixA, sizeof(double) * totalSize);    //Выделение pinned память 
    cudaMallocHost(&matrixB, sizeof(double) * totalSize);

	std::memset(matrixA, 0, size * size * sizeof(double));

	// Заполнение граничных условий
	matrixA[0] = 10;
	matrixA[size - 1] = 20;
	matrixA[size * size - 1] = 30;
	matrixA[size * (size - 1)] = 20;

	const double step = 1.0 * (20 - 10) / (size - 1);
	for (int i = 1; i < size - 1; i++)
	{
		matrixA[i] = 10 + i * step;
		matrixA[i * size] = 10 + i * step;
		matrixA[size - 1 + i * size] = 20 + i * step;
		matrixA[size * (size - 1) + i] = 20 + i * step;
	}
	
	std::memcpy(matrixB, matrixA, totalSize * sizeof(double));

	//print(size, rank, matrixA, sizeOfAreaForOneProcess);


	// Расчитываем, сколько памяти требуется процессу

	if(sizeOfTheGroup == 1) sizeOfAreaForOneProcess = sizeOfAreaForOneProcess;

	else {
		if (rank != 0 && rank != sizeOfTheGroup - 1) {
		sizeOfAreaForOneProcess += 2;
		}
		else {
		sizeOfAreaForOneProcess += 1;
		}
	}

	size_t sizeOfAllocatedMemory = size * sizeOfAreaForOneProcess;

	// Выделяем память на девайсе
	cudaMalloc((void**)&deviceMatrixAPtr, sizeOfAllocatedMemory * sizeof(double));
	cudaMalloc((void**)&deviceMatrixBPtr, sizeOfAllocatedMemory * sizeof(double));
	cudaMalloc((void**)&errorMatrix, sizeOfAllocatedMemory * sizeof(double));
	cudaMalloc((void**)&deviceError, sizeof(double));

	// Копируем часть заполненной матрицы в выделенную память, начиная с 1 строки
	size_t offset = (rank != 0) ? size : 0;
 	cudaMemcpy(deviceMatrixAPtr, matrixA + (startYIdx * size) - offset, 
					sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMatrixBPtr, matrixB + (startYIdx * size) - offset, 
					sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice);

	// Здесь мы получаем размер временного буфера для редукции и выделяем память для этого буфера
	size_t tempStorageSize = 0;
	cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, size * sizeOfAreaForOneProcess);
	cudaMalloc((void**)&tempStorage, tempStorageSize);

	double* error;
	cudaMallocHost(&error, sizeof(double));
	*error = 1.0;

	cudaStream_t stream, matrixCalculationStream;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&matrixCalculationStream);

	unsigned int threads_x = std::min((size), 1024);
    unsigned int blocks_y = sizeOfAreaForOneProcess;
    unsigned int blocks_x = (size) / threads_x;

    dim3 blockDim(threads_x, 1);
    dim3 gridDim(blocks_x, blocks_y);

	cudaMemcpy(matrixA, deviceMatrixAPtr, sizeof(double) * size * sizeOfAreaForOneProcess, cudaMemcpyDeviceToHost);

	//print(size, rank, matrixA, sizeOfAreaForOneProcess);

	int iter = 0; 
	
	// Главный алгоритм 
	clock_t begin = clock();
	while((iter < maxIter) && (*error) > minError)
	{
		iter++;

		// Расчитываем границы, которые потом будем отправлять другим процессам
		if(sizeOfAreaForOneProcess > 2){ 
			calculateBoundaries<<<size, 1, 0, stream>>>(deviceMatrixAPtr, deviceMatrixBPtr, 
										size, sizeOfAreaForOneProcess);
		

			cudaStreamSynchronize(stream);
		// Расчет матрицы
			calculateMatrix<<<gridDim, blockDim, 0, matrixCalculationStream>>>
							(deviceMatrixAPtr, deviceMatrixBPtr, size, sizeOfAreaForOneProcess);
		
		}

		// Расчитываем ошибку каждую сотую итерацию
		if (iter % 100 == 0)
		{
			getErrorMatrix<<<gridDim, blockDim, 0, matrixCalculationStream>>>(deviceMatrixAPtr, deviceMatrixBPtr, errorMatrix,
															size, sizeOfAreaForOneProcess);

			cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, sizeOfAllocatedMemory, matrixCalculationStream);
			
			cudaStreamSynchronize(matrixCalculationStream);
			
			// Находим максимальную ошибку среди всех и передаём её всем процессам
			MPI_Allreduce((void*)deviceError, (void*)deviceError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

			cudaMemcpy(error, deviceError, sizeof(double), cudaMemcpyDeviceToHost);
		}
		
		

		if(sizeOfTheGroup > 1){ 
			// Обмен "граничными" условиями каждой области
			// Обмен верхней границей
			if (rank != 0)
			{
				MPI_Sendrecv(deviceMatrixBPtr + size + 1, size - 2, MPI_DOUBLE, rank - 1, 0, 
				deviceMatrixBPtr + 1, size - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			// Обмен нижней границей
			if (rank != sizeOfTheGroup - 1)
			{
				MPI_Sendrecv(deviceMatrixBPtr + (sizeOfAreaForOneProcess - 2) * size + 1, size - 2, MPI_DOUBLE, rank + 1, 0,
								deviceMatrixBPtr + (sizeOfAreaForOneProcess - 1) * size + 1, 
								size - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
	
		cudaStreamSynchronize(matrixCalculationStream);
		// Обмен указателей
		std::swap(deviceMatrixAPtr, deviceMatrixBPtr);
	}

	clock_t end = clock();
	if (rank == 0)
	{
		std::cout << "Time: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
		std::cout << "Iter: " << iter << " Error: " << *error << std::endl;
		
	}
	
	cudaMemcpy(matrixA, deviceMatrixAPtr, sizeof(double) * size * sizeOfAreaForOneProcess, cudaMemcpyDeviceToHost);

	print(size, rank, matrixA, sizeOfAreaForOneProcess);

	MPI_Finalize();

    if (deviceMatrixAPtr) 	cudaFree(deviceMatrixAPtr);
	if (deviceMatrixBPtr) 	cudaFree(deviceMatrixBPtr);
	if (errorMatrix)	  	cudaFree(errorMatrix);
	if (tempStorage) 		cudaFree(tempStorage);
	if (matrixA) 			cudaFree(matrixA);
	if (matrixB) 			cudaFree(matrixB);

	

	return 0;
}
