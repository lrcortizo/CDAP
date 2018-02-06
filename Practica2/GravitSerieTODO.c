#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#define MAX_CHAR 100
#define MAX_OBJECTS 10
#define DATAFILE "data.txt"
#define G 6.674e-11
#define NUM_ITER 100001
#define NUM_ITER_SHOW 5000
#define verbose false

void main(){
    char  str[MAX_CHAR];
    FILE *file;
    int noOfObjects;
    double x[MAX_OBJECTS], y[MAX_OBJECTS], vx[MAX_OBJECTS], vy[MAX_OBJECTS], m[MAX_OBJECTS];
    double x_new[MAX_OBJECTS], y_new[MAX_OBJECTS], vx_new[MAX_OBJECTS], vy_new[MAX_OBJECTS];
    int i,j;

    file = fopen( DATAFILE , "r");
    fscanf(file,"%s",str);
    noOfObjects = atoi(str);
    printf("Number of objects: %d\n",noOfObjects);
    if (noOfObjects > MAX_OBJECTS) {
        printf("*** ERROR: maximum no. of objects exceeded ***\n");
        exit(0);
    }

    printf("\n");

    for (i=0; i< noOfObjects; i++) {
        fscanf(file,"%s",str);
        x[i] = atof(str);
        fscanf(file,"%s",str);
        y[i] = atof(str);
        fscanf(file,"%s",str);
        vx[i] = atof(str);
        fscanf(file,"%s",str);
        vy[i] = atof(str);
        fscanf(file,"%s",str);
        m[i] = atof(str);
    }

    for (int niter=0; niter<NUM_ITER; niter++) {

        for (i=0; i< noOfObjects; i++) {
            x_new[i]=x[i];
            y_new[i]=y[i];
            vx_new[i]=vx[i];
            vy_new[i]=vy[i];
        }

        _Bool showData = false;

        if (niter % NUM_ITER_SHOW == 0)
            showData = true;

        if (showData)
            printf("***** ITERATION %d *****\n",niter);

        for (i=0; i< noOfObjects; i++) {
            double ax_total=0;
            double ay_total=0;
            for (j=0; j < noOfObjects; j++) {
                if (i==j)
                    continue;

                double d = sqrt((x[j]-x[i])*(x[j]-x[i])+(y[j]-y[i])*(y[j]-y[i]));
                double f = G*((m[j]*m[i])/(d*d));
                double fx = f*((x[j]-x[i])/d)
                double ax = // TODO
                double fy = f*((y[j]-y[i])/d)
                double ay = // TODO

                if (showData && verbose) {
                    printf("  Distance between objects %d and %d: %f m\n",i+1,j+1,d);
                    printf("  Force between objects %d and %d: %f N*m²/kg²\n",i+1,j+1,f);
                    printf("  Force along x axis on object %d made by object %d: %f N*m²/kg²\n",i+1,j+1,fx);
                    printf("  Acceleration along x axis on object %d made by object %d: %f m/s²\n",i+1,j+1,ax);
                    printf("  Force along y axis on object %d made by object %d: %f N*m²/kg²\n",i+1,j+1,fy);
                    printf("  Acceleration along y axis on object %d made by object %d: %f m/s²\n",i+1,j+1,ay);
                }

                ax_total += ax;
                ay_total += ay;

            }

            vx_new[i] += ax_total;
            vy_new[i] += ay_total;

            x_new[i] += vx_new[i];
            y_new[i] += vy_new[i];
            if (showData)
                printf("New position of object %d: %.2f, %.2f\n",i,x_new[i],y_new[i]);

        }  // noOfObjects

        for (i=0; i< noOfObjects; i++) {
            x[i]=x_new[i];
            y[i]=y_new[i];
            vx[i]=vx_new[i];
            vy[i]=vy_new[i];
        }

    }  // nIter

}  // main
