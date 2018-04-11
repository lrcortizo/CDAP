#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#define MAX_CHAR 100
#define MAX_OBJECTS 10
#define DATAFILE "data.txt"
#define G 6.674e-11
#define NUM_ITER 100001
#define NUM_ITER_SHOW 5000
#define verbose false
#define M_EARTH 5.972e24 // mass of earth (kg)
#define M_SAT 50 // mass of satellites (kg)
#define V_SAT 2609.56 // speed of satellites (geosynchronous orbit) (m/s)
#define D_SAT 35786000 // distance of satellites (geosynchronous orbit) (m)
#define G 6.674e-11 // Gravitational constant
#define PI 3.14159265359 //Pi constant

void main(int argc, char *argv[]){
    char  str[MAX_CHAR];
    FILE *file;
    int noOfObjects;
    double x[MAX_OBJECTS], y[MAX_OBJECTS], vx[MAX_OBJECTS], vy[MAX_OBJECTS], m[MAX_OBJECTS];
    double x_new[MAX_OBJECTS], y_new[MAX_OBJECTS], vx_new[MAX_OBJECTS], vy_new[MAX_OBJECTS];
    int i,j;
    double alpha;

    if (argc > 1) {
      noOfObjects = atoi(argv[1]);
      printf("Number of objects: %d\n",noOfObjects);
    }
    else {
        printf("*** ERROR: needed at least one satelite ***\n");
        exit(0);
    }

    printf("\n");

    //Tierra
    alpha = (PI*(360.0/(noOfObjects-1)))/180.0;
    double angle = 0;
    x[0] = 0.0;
    y[0] = 0.0;
    vx[0] = 0.0;
    vy[0] = 0.0;
    m[0] = M_EARTH;

    //Calculos iniciales satelites
    for (i=1; i < noOfObjects;i++) {
      x[i] = cos(angle) * D_SAT;
      y[i] = sin(angle) * D_SAT;
      vx[i] = -sin(angle) * V_SAT;
      vy[i] = cos(angle) * V_SAT;
      m[i] = M_SAT;
      printf("Satelite nº %d \n", i );
      printf("\tAngulo: %f \n", angle );
      printf("\tPosition: %f,%f \n", x[i], y[i] );
      printf("\tVx, Vy: %f,%f \n", vx[i], vy[i] );
      angle+=alpha;
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


        #pragma omp parallel for shared(x_new, y_new, vx_new, vy_new, x, y, m, noOfObjects, showData) private(i, j)
        for (i=0; i< noOfObjects; i++) {
            double ax_total=0;
            double ay_total=0;
            for (j=0; j < noOfObjects; j++) {
                if (i==j)
                    continue;

                double d = sqrt(fabs((x[j]-x[i]) * (x[j]-x[i]) + (y[j]-y[i]) * (y[j]-y[i])));
                double f = G * ((m[j]*m[i])/(d*d));
                double fx = f * ((x[j]-x[i])/d);
                double ax = fx / m[i];
                double fy = f * ((y[j]-y[i])/d);
                double ay = fy / m[i];

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

            if (showData) {
                printf("New position of object %d: %.2f, %.2f\n",i,x_new[i],y_new[i]);
            }
        }  // noOfObjects

        for (i=0; i< noOfObjects; i++) {
            x[i]=x_new[i];
            y[i]=y_new[i];
            vx[i]=vx_new[i];
            vy[i]=vy_new[i];
        }

    }  // nIter

}  // main
