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

        for (i=0; i< noOfObjects; i++) {
            x_new[i]=x[i];
            y_new[i]=y[i];
            vx_new[i]=vx[i];
            vy_new[i]=vy[i];
        }

        for (i=0; i< noOfObjects; i++) {
            double ax_total=0;
            double ay_total=0;
            for (j=0; j < noOfObjects; j++) {
                if (i==j)
                    continue;

                double d = sqrt(pow( (x[i]-x[j]) ,2.0) + pow( (y[i]-y[j]) ,2.0));
                double f = G*m[i]*m[j]/pow(d,2.0);
                double fx = f*(x[j]-x[i])/d;
                double ax = fx/m[i];
                double fy = f*(y[j]-y[i])/d;
                double ay = fy/m[i];


                ax_total += ax;
                ay_total += ay;

            }
            
            vx_new[i] += ax_total;
            vy_new[i] += ay_total;

            x_new[i] += vx_new[i];
            y_new[i] += vy_new[i];

        }  // noOfObjects

        for (i=0; i< noOfObjects; i++) {
            x[i]=x_new[i];
            y[i]=y_new[i];
            vx[i]=vx_new[i];
            vy[i]=vy_new[i];
        }

        if (niter%NUM_ITER_SHOW == 0) 
            printf("Iteration %d/%d\n", niter, NUM_ITER);

    }  // nIter
    end=clock();

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

}  // main
