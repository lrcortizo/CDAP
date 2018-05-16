#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define MAX_CHAR 100
#define DATAFILE "data.txt"
#define RESULTSFILE "resultsCudal.txt"
#define G 6.674e-11
#define NUM_ITER 1000
#define NUM_ITER_SHOW 50

__device__ double atomicAddD(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
		                __double_as_longlong(val +
		                                     __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void asteroid(double * gpu_x, double * gpu_y, double * gpu_vx, double * gpu_vy, double * gpu_m){
	int posx = blockIdx.x * blockDim.x + threadIdx.x;
	int posy = blockIdx.y * blockDim.y + threadIdx.y;

	if (posx!=posy) {
		double d = sqrt(pow( (gpu_x[posx]-gpu_x[posy]),2.0) + pow( (gpu_y[posx]-gpu_y[posy]),2.0));
		double f = G*gpu_m[posx]*gpu_m[posy]/pow(d,2.0);
		double fx = f*(gpu_x[posy]-gpu_x[posx])/d;
		double fy = f*(gpu_y[posy]-gpu_y[posx])/d;
		double ax = fx/gpu_m[posx];
		double ay = fy/gpu_m[posx];
		// atomicAddD(gpu_vx+posx, ax);
		// atomicAddD(gpu_vy+posx, ay);
		atomicAddD(&gpu_vx[posx], ax);
		atomicAddD(&gpu_vy[posx], ay);
	}
}

__global__ void positions(double * gpu_x, double * gpu_y, double * gpu_vx, double * gpu_vy, double * gpu_m){
	int i = blockIdx.x * blockIdx.x + threadIdx.x;
	// gpu_x[i] += gpu_vx[i];
	// gpu_y[i] += gpu_vy[i];
	atomicAddD(&gpu_x[i], gpu_vx[i]);
	atomicAddD(&gpu_y[i], gpu_vy[i]);
}

int main(){

	clock_t start, end;
	double time_used;
	char str[MAX_CHAR];
	FILE *file;
	int noOfObjects;
	int i;

	file = fopen( DATAFILE, "r");
	fscanf(file,"%s",str);
	noOfObjects = atoi(str);
	printf("Number of objects: %d\n",noOfObjects);

	const int ARRAY_BYTES = noOfObjects * sizeof(double);

	double *x = (double *) malloc(ARRAY_BYTES);
	double *y = (double *) malloc(ARRAY_BYTES);
	double *vx = (double *) malloc(ARRAY_BYTES);
	double *vy = (double *) malloc(ARRAY_BYTES);
	double *m = (double *) malloc(ARRAY_BYTES);

	double *x_new = (double *) malloc(ARRAY_BYTES);
	double *y_new = (double *) malloc(ARRAY_BYTES);
	double *vx_new = (double *) malloc(ARRAY_BYTES);
	double *vy_new = (double *) malloc(ARRAY_BYTES);

	// declare GPU memory pointers
	double * gpu_x;
	double * gpu_y;
	double * gpu_vx;
	double * gpu_vy;
	double * gpu_m;
	double * gpu_x_new;
	double * gpu_y_new;
	double * gpu_vx_new;
	double * gpu_vy_new;

	// allocate GPU memory
	cudaMalloc((void**) &gpu_x, ARRAY_BYTES);
	cudaMalloc((void**) &gpu_y, ARRAY_BYTES);
	cudaMalloc((void**) &gpu_vx, ARRAY_BYTES);
	cudaMalloc((void**) &gpu_vy, ARRAY_BYTES);
	cudaMalloc((void**) &gpu_m, ARRAY_BYTES);
	cudaMalloc((void**) &gpu_x_new, ARRAY_BYTES);
	cudaMalloc((void**) &gpu_y_new, ARRAY_BYTES);
	cudaMalloc((void**) &gpu_vx_new, ARRAY_BYTES);
	cudaMalloc((void**) &gpu_vy_new, ARRAY_BYTES);

	// launch the kernel
	for (i=0; i < noOfObjects; i++) {
		fscanf(file,"%s",str);
		x[i] = atof(str);
		x_new[i] = atof(str);
		fscanf(file,"%s",str);
		y[i] = atof(str);
		y_new[i] = atof(str);
		fscanf(file,"%s",str);
		vx[i] = atof(str);
		vx_new[i] = atof(str);
		fscanf(file,"%s",str);
		vy[i] = atof(str);
		vy_new[i] = atof(str);
		fscanf(file,"%s",str);
		m[i] = atof(str);
	}
	fclose(file);

	// transfer the array to the GPU
	cudaMemcpy(gpu_x, x, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_y, y, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_vx, vx, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_vy, vy, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_m, m, ARRAY_BYTES, cudaMemcpyHostToDevice);

	start=clock();

	dim3 blocksPerGrid(32,32);              // blocks per grid
	dim3 threadsPerBlock(32,32);            // threads per block

	for (int niter=0; niter<=NUM_ITER; niter++) {

		asteroid<<<blocksPerGrid, threadsPerBlock>>>(gpu_x, gpu_y, gpu_vx, gpu_vy, gpu_m);
		positions<<<1, 1024>>>(gpu_x, gpu_y, gpu_vx, gpu_vy, gpu_m);

		if (niter%NUM_ITER_SHOW == 0)
			printf("Iteration %d/%d\n", niter, NUM_ITER);

	}  // nIter
	end=clock();

	// copy back the result array to the CPU
	cudaMemcpy(x, gpu_x, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, gpu_y, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(vx, gpu_vx, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(vy, gpu_vy, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(m, gpu_m, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	file = fopen( RESULTSFILE, "w");
	fprintf(file, "Movement of objects\n");
	fprintf(file, "-------------------\n");
	for (i = 0; i < noOfObjects; i++) {
		double mov = sqrt(pow( (x_new[i]-x[i]),2.0) + pow( (y_new[i]-y[i]),2.0));
		// printf("  Object %i  - ORIGINAL: (%f,%f) -- NEW: (%f,%f) -> %f meters\n", i, x[i], y[i], x_new[i], y_new[i], mov);
		fprintf(file,"  Object %i  - ORIGINAL: (%f,%f) -- NEW: (%f,%f) -> %f meters\n", i, x[i], y[i], x_new[i], y_new[i], mov);

	}
	int hours = NUM_ITER/3600;
	int mins = (NUM_ITER - hours*3600)/60;
	int secs = (NUM_ITER - hours*3600 - mins*60);
	fprintf(file,"Time elapsed: %i seconds (%i hours, %i minutes, %i seconds)\n",NUM_ITER, hours, mins, secs);

	time_used = ((double)(end-start)) / CLOCKS_PER_SEC;
	fprintf(file,"Processing time: %f sec.\n",time_used);
	fclose(file);

	cudaFree(gpu_x);
	cudaFree(gpu_y);
	cudaFree(gpu_vx);
	cudaFree(gpu_vy);
	cudaFree(gpu_m);

	cudaFree(gpu_x_new);
	cudaFree(gpu_y_new);
	cudaFree(gpu_vx_new);
	cudaFree(gpu_vy_new);

}  // main
