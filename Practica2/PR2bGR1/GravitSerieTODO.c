#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "mpi.h"

#define MAX_CHAR 100
#define MAX_OBJECTS 10
#define DATAFILE "data.txt"
#define G 6.674e-11
#define NUM_ITER 100001
#define NUM_ITER_SHOW 5000
#define verbose true

void main(int argc, char* argv[]){
    char  str[MAX_CHAR];
    FILE *file;
    int noOfObjects;
    double x_other[MAX_OBJECTS-1], y_other[MAX_OBJECTS-1], vx_other[MAX_OBJECTS-1], vy_other[MAX_OBJECTS-1], m_other[MAX_OBJECTS-1];
    double x, y, vx, vy, m;
	double x_new, y_new, vx_new, vy_new;
    double temp;
    int i,j;

    int my_rank;
    int p;
    int source; int dest;
    int tag=0;
    char message[100];

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    file = fopen( DATAFILE , "r");
    fscanf(file,"%s",str);
    noOfObjects = atoi(str);
    printf("Number of objects: %d\n",noOfObjects);
    if (noOfObjects > MAX_OBJECTS) {
        printf("*** ERROR: maximum no. of objects exceeded ***\n");
        exit(0);
    }

    printf("\n");

    if(my_rank == 0)
    {
        fscanf(file,"%s",str);
        x = atof(str);
        fscanf(file,"%s",str);
        y = atof(str);
        fscanf(file,"%s",str);
        vx = atof(str);
        fscanf(file,"%s",str);
        vy = atof(str);
        fscanf(file,"%s",str);
        m = atof(str);

        for (i=1; i< noOfObjects; i++) {
			fscanf(file,"%s",str);
			temp = atof(str);
            MPI_Send(&temp,sizeof(double),MPI_FLOAT,i,0,MPI_COMM_WORLD); //x
			fscanf(file,"%s",str);
			temp = atof(str);
            MPI_Send(&temp,sizeof(double),MPI_FLOAT,i,1,MPI_COMM_WORLD); //y
			fscanf(file,"%s",str);
			temp = atof(str);
            MPI_Send(&temp,sizeof(double),MPI_FLOAT,i,2,MPI_COMM_WORLD); //vx
			fscanf(file,"%s",str);
			temp = atof(str);
            MPI_Send(&temp,sizeof(double),MPI_FLOAT,i,3,MPI_COMM_WORLD); //vy
			fscanf(file,"%s",str);
			temp = atof(str);
            MPI_Send(&temp,sizeof(double),MPI_FLOAT,i,4,MPI_COMM_WORLD); //m
        }
    }
    else
    {
        MPI_Recv(&x,sizeof(double),MPI_FLOAT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
        MPI_Recv(&y,sizeof(double),MPI_FLOAT,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,&status);
        MPI_Recv(&vx,sizeof(double),MPI_FLOAT,MPI_ANY_SOURCE,2,MPI_COMM_WORLD,&status);
        MPI_Recv(&vy,sizeof(double),MPI_FLOAT,MPI_ANY_SOURCE,3,MPI_COMM_WORLD,&status);
        MPI_Recv(&m,sizeof(double),MPI_FLOAT,MPI_ANY_SOURCE,4,MPI_COMM_WORLD,&status);
    }
	printf("Starting Values for %d:: x:%.2f, y:%.2f, vx:%.2f, vy:%.2f, m:%.2f\n",my_rank,x,y,vx,vy,m);

    for (int niter=0; niter<5; niter++) {

        //for (i=0; i< noOfObjects; i++) {
            x_new=x;
            y_new=y;
            vx_new=vx;
            vy_new=vy;
        //}
		printf("Starting Values NEW for %d:: x:%.2f, y:%.2f, vx:%.2f, vy:%.2f, m:%.2f\n",my_rank,x_new,y_new,vx_new,vy_new,m);
        // Gather all data down to all the processes
        MPI_Allgather(&x, noOfObjects, MPI_FLOAT, &x_other, noOfObjects, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather(&y, noOfObjects, MPI_FLOAT, &y_other, noOfObjects, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather(&vx, noOfObjects, MPI_FLOAT, &vx_other, noOfObjects, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather(&vy, noOfObjects, MPI_FLOAT, &vy_other, noOfObjects, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather(&m, noOfObjects, MPI_FLOAT, &m_other, noOfObjects, MPI_FLOAT, MPI_COMM_WORLD);

        _Bool showData = false;

        if (niter % NUM_ITER_SHOW == 0)
            showData = true;

        //if (showData)
        //    printf("***** ITERATION %d *****\n",niter);

        //for (i=0; i< noOfObjects; i++) {
            double ax_total=0;
            double ay_total=0;
            for (j=0; j < noOfObjects-1; j++) {
                if (my_rank==j)
                    continue;

                double d = sqrt(fabs((x_other[j]-x) * (x_other[j]-x) + (y_other[j]-y) * (y_other[j]-y)));
                double f = G * ((m_other[j]*m)/(d*d));
                double fx = f * ((x_other[j]-x)/d);
                double ax = fx / m;
                double fy = f * ((y_other[j]-y)/d);
                double ay = fy / m;

                if (showData && verbose) {
                    printf("  Distance between objects %d and %d: %f m\n",my_rank+1,j+1,d);
                    printf("  Force between objects %d and %d: %f N*m²/kg²\n",my_rank+1,j+1,f);
                    printf("  Force along x axis on object %d made by object %d: %f N*m²/kg²\n",my_rank+1,j+1,fx);
                    printf("  Acceleration along x axis on object %d made by object %d: %f m/s²\n",my_rank+1,j+1,ax);
                    printf("  Force along y axis on object %d made by object %d: %f N*m²/kg²\n",my_rank+1,j+1,fy);
                    printf("  Acceleration along y axis on object %d made by object %d: %f m/s²\n",my_rank+1,j+1,ay);
                }

                ax_total += ax;
                ay_total += ay;

            }

            vx_new += ax_total;
            vy_new += ay_total;

            x_new += vx_new;
            y_new += vy_new;
            if (showData)
                printf("New position of object %d: %.2f, %.2f\n",i,x_new,y_new);

        //}  // noOfObjects

        for (i=0; i< noOfObjects; i++) {
            x=x_new;
            y=y_new;
            vx=vx_new;
            vy=vy_new;
        }

    }  // nIter
}  // main
