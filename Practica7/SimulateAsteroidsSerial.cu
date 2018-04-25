#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define MAX_CHAR 100
#define DATAFILE "data.txt"
#define RESULTSFILE "resultsSerial.txt"
#define G 6.674e-11
#define NUM_ITER 1000
#define NUM_ITER_SHOW 1000


__global__ void asteroid(double * gpu_x, double * gpu_y, double * gpu_vx, double * gpu_vy, double * gpu_m, double * gpu_x_new, double * gpu_y_new, double * gpu_vx_new, double * gpu_vy_new){
        int i = threadIdx.x; // local or register

        if (i < noOfObjects){
             gpu_x_new[i]=gpu_x[i];
             gpu_y_new[i]=gpu_y[i];
             gpu_vx_new[i]=gpu_vx[i];
             gpu_vy_new[i]=gpu_vy[i];
         }

         if (i < noOfObjects){
             double ax_total=0;
             double ay_total=0;
             for (int j=0; j < noOfObjects; j++) {
                 if (i==j)
                     continue;

                 double d = sqrt(pow( (gpu_x[i]-gpu_x[j]) ,2.0) + pow( (gpu_y[i]-gpu_y[j]) ,2.0));
                 double f = G*gpu_m[i]*gpu_m[j]/pow(d,2.0);
                 double fx = f*(gpu_x[j]-gpu_x[i])/d;
                 double ax = fx/gpu_m[i];
                 double fy = f*(gpu_y[j]-gpu_y[i])/d;
                 double ay = fy/gpu_m[i];

                 ax_total += ax;
                 ay_total += ay;

             }

             gpu_vx_new[i] += ax_total;
             gpu_vy_new[i] += ay_total;

             gpu_x_new[i] += gpu_vx_new[i];
             gpu_y_new[i] += gpu_vy_new[i];

         }  // noOfObjects

         if (i < noOfObjects){
             x[i]=gpu_x_new[i];
             y[i]=gpu_y_new[i];
             vx[i]=gpu_vx_new[i];
             vy[i]=gpu_vy_new[i];
         }
}

void main(){

    clock_t start, end;
    double time_used;
    char  str[MAX_CHAR];
    FILE *file;
    int noOfObjects;
    int i,j;

    file = fopen( DATAFILE , "r");
    fscanf(file,"%s",str);
    noOfObjects = atoi(str);
    printf("Number of objects: %d\n",noOfObjects);


    double *x = (double *) malloc(sizeof(double)*noOfObjects);
    double *y = (double *) malloc(sizeof(double)*noOfObjects);
    double *vx = (double *) malloc(sizeof(double)*noOfObjects);
    double *vy = (double *) malloc(sizeof(double)*noOfObjects);
    double *m = (double *) malloc(sizeof(double)*noOfObjects);

    double *x0 = (double *) malloc(sizeof(double)*noOfObjects);
    double *y0 = (double *) malloc(sizeof(double)*noOfObjects);
    double *vx0 = (double *) malloc(sizeof(double)*noOfObjects);
    double *vy0 = (double *) malloc(sizeof(double)*noOfObjects);

    double *x_new = (double *) malloc(sizeof(double)*noOfObjects);
    double *y_new = (double *) malloc(sizeof(double)*noOfObjects);
    double *vx_new = (double *) malloc(sizeof(double)*noOfObjects);
    double *vy_new = (double *) malloc(sizeof(double)*noOfObjects);

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

    // allocate GPU memory
    cudaMalloc((void**) &gpu_x, noOfObjects*sizeof(double));
    cudaMalloc((void**) &gpu_y, noOfObjects*sizeof(double));
    cudaMalloc((void**) &gpu_vx, noOfObjects*sizeof(double));
    cudaMalloc((void**) &gpu_vy, noOfObjects*sizeof(double));
    cudaMalloc((void**) &gpu_m, noOfObjects*sizeof(double));
    cudaMalloc((void**) &gpu_x_new, noOfObjects*sizeof(double));
    cudaMalloc((void**) &gpu_y_new, noOfObjects*sizeof(double));
    cudaMalloc((void**) &gpu_vx_new, noOfObjects*sizeof(double));
    cudaMalloc((void**) &gpu_vy_new, noOfObjects*sizeof(double));

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

    // launch the kernel


    for (i=0; i< noOfObjects; i++) {
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

    start=clock();
    for (int niter=0; niter<NUM_ITER; niter++) {

        asteroid<<<1, noOfObjects>>>(d_out, d_in);


        if (niter%NUM_ITER_SHOW == 0)
            printf("Iteration %d/%d\n", niter, NUM_ITER);

    }  // nIter
    end=clock();

    // copy back the result array to the CPU
    cudaMemcpy(x, gpu_x, noOfObjects*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, gpu_y, noOfObjects*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vx, gpu_vx, noOfObjects*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vy, gpu_vy, noOfObjects*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(m, gpu_m, noOfObjects*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(x_new, gpu_x_new, noOfObjects*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_new, gpu_y_new, noOfObjects*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vx_new, gpu_vx_new, noOfObjects*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vy_new, gpu_vy_new, noOfObjects*sizeof(double), cudaMemcpyDeviceToHost);

    file = fopen( RESULTSFILE , "w");
    fprintf(file, "Movement of objects\n");
    fprintf(file, "-------------------\n");
    for (i=0; i<noOfObjects; i++) {
        double mov = sqrt(pow( (x0[i]-x[i]) ,2.0) + pow( (y0[i]-y[i]) ,2.0));
        fprintf(file,"  Object %i  -  %f meters\n", i, mov);
    }
    int hours = NUM_ITER/3600;
    int mins = (NUM_ITER - hours*3600)/60;
    int secs = (NUM_ITER - hours*3600 - mins*60);
    fprintf(file,"Time elapsed: %i seconds (%i hours, %i minutes, %i seconds)\n",NUM_ITER, hours, mins, secs);

    time_used = ((double)(end-start)) / CLOCKS_PER_SEC;
    fprintf(file,"Processing time: %f sec.\n",time_used);
    fclose(file);
    // copy back the result array to the CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    // print out the resulting array
    for (int i =0; i < ARRAY_SIZE; i++) {
    printf("%f", h_out[i]);
    printf(((i % 4) != 3) ? "\t" : "\n");
    }
    cudaFree(d_in);
    cudaFree(d_out);
}  // main
