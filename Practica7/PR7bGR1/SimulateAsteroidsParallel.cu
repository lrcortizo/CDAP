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

__global__ void asteroid(double * gpu_x, double * gpu_y, double * gpu_vx, double * gpu_vy, double * gpu_m, double * gpu_x_new, double * gpu_y_new, double * gpu_vx_new, double * gpu_vy_new, int noOfObjects){
	int i = threadIdx.x;     // local or register

    int posx = blockIdx.x * 32 + threadIdx.x;
    int posy = blockIdx.y * 32 + threadIdx.y;

	// printf("Object %d: %d, %d\n", i, gpu_x[i], gpu_y[i]);

	if (i < noOfObjects) {
		gpu_x_new[i]=gpu_x[i];
		gpu_y_new[i]=gpu_y[i];
		gpu_vx_new[i]=gpu_vx[i];
		gpu_vy_new[i]=gpu_vy[i];
	}

	if (i < noOfObjects) {
		double ax_total=0;
		double ay_total=0;
		for (int j=0; j < noOfObjects; j++) {
			if (i==j)
				continue;

			double d = sqrt(pow( (gpu_x[i]-gpu_x[j]),2.0) + pow((gpu_y[i]-gpu_y[j]),2.0));
			double f = G*gpu_m[i]*gpu_m[j]/pow(d,2.0);
			double fx = f*(gpu_x[j]-gpu_x[i])/d;
			double ax = fx/gpu_m[i];
			double fy = f*(gpu_y[j]-gpu_y[i])/d;
			double ay = fy/gpu_m[i];

			ax_total += ax;
			ay_total += ay;

		}

		atomicAddD(&gpu_vx_new[i], ax_total);
		atomicAddD(&gpu_vy_new[i], ay_total);

		atomicAddD(&gpu_x_new[i], gpu_vx_new[i]);
		atomicAddD(&gpu_y_new[i], gpu_vy_new[i]);

	}       // noOfObjects
	printf("Object %d: %d, %d\n", i, gpu_x[i], gpu_y[i]);

	if (i < noOfObjects) {
		gpu_x[i]=gpu_x_new[i];
		gpu_y[i]=gpu_y_new[i];
		gpu_vx[i]=gpu_vx_new[i];
		gpu_vy[i]=gpu_vy_new[i];
	}
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

	double *x0 = (double *) malloc(ARRAY_BYTES);
	double *y0 = (double *) malloc(ARRAY_BYTES);
	double *vx0 = (double *) malloc(ARRAY_BYTES);
	double *vy0 = (double *) malloc(ARRAY_BYTES);

	double *x_new = (double *) malloc(ARRAY_BYTES);
	double *y_new = (double *) malloc(ARRAY_BYTES);
	double *vx_new = (double *) malloc(ARRAY_BYTES);
	double *vy_new = (double *) malloc(ARRAY_BYTES);

	printf("\n");

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

	// launch the kernel
	for (i=0; i < noOfObjects; i++) {
		fscanf(file,"%s",str);
		x[i] = atof(str);
		x0[i] = atof(str);
		fscanf(file,"%s",str);
		y[i] = atof(str);
		y0[i] = atof(str);
		fscanf(file,"%s",str);
		vx[i] = atof(str);
		vx0[i] = atof(str);
		fscanf(file,"%s",str);
		vy[i] = atof(str);
		vy0[i] = atof(str);
		fscanf(file,"%s",str);
		m[i] = atof(str);
	}
	fclose(file);

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

	// transfer the array to the GPU
	cudaMemcpy(gpu_x, x, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_y, y, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_vx, vx, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_vy, vy, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_m, m, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_x_new, x_new, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_y_new, y_new, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_vx_new, vx_new, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_vy_new, vy_new, ARRAY_BYTES, cudaMemcpyHostToDevice);

	start=clock();

	dim3 blocks(1024/32,1024/32);   // blocks per grid
	dim3 threads(32,32);            // threads per block

	for (int niter=0; niter<NUM_ITER; niter++) {

		// Call to function
		asteroid<<<blocks, threads>>>(gpu_x, gpu_y, gpu_vx, gpu_vy, gpu_m, gpu_x_new, gpu_y_new, gpu_vx_new, gpu_vy_new, noOfObjects);

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
	cudaMemcpy(x_new, gpu_x_new, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_new, gpu_y_new, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(vx_new, gpu_vx_new, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(vy_new, gpu_vy_new, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	file = fopen( RESULTSFILE, "w");
	fprintf(file, "Movement of objects\n");
	fprintf(file, "-------------------\n");
	for (i=0; i<noOfObjects; i++) {
		double mov = sqrt(pow( (x0[i]-x[i]),2.0) + pow( (y0[i]-y[i]),2.0));
		fprintf(file,"  Object %i  -  %f meters\n", i, mov);
	}
	int hours = NUM_ITER/3600;
	int mins = (NUM_ITER - hours*3600)/60;
	int secs = (NUM_ITER - hours*3600 - mins*60);
	fprintf(file,"Time elapsed: %i seconds (%i hours, %i minutes, %i seconds)\n",NUM_ITER, hours, mins, secs);

	time_used = ((double)(end-start)) / CLOCKS_PER_SEC;
	fprintf(file,"Processing time: %f sec.\n",time_used);
	fclose(file);
	// // copy back the result array to the CPU
	// cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	// // print out the resulting array
	// for (int i =0; i < ARRAY_SIZE; i++) {
	// printf("%f", h_out[i]);
	// printf(((i % 4) != 3) ? "\t" : "\n");
	// }
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
