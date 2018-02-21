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
#define verbose false

void main(int argc, char* argv[]){
    char  str[MAX_CHAR];
    FILE *file;
    int noOfObjects;
    double x_other, y_other, vx_other, vy_other, m_other;
    double x, y, vx, vy, m;
	double x_new, y_new, vx_new, vy_new;
    double temp;
    int i,j,k;

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
            MPI_Send(&temp,1,MPI_DOUBLE,i,0,MPI_COMM_WORLD); //x
			fscanf(file,"%s",str);
			temp = atof(str);
            MPI_Send(&temp,1,MPI_DOUBLE,i,1,MPI_COMM_WORLD); //y
			fscanf(file,"%s",str);
			temp = atof(str);
            MPI_Send(&temp,1,MPI_DOUBLE,i,2,MPI_COMM_WORLD); //vx
			fscanf(file,"%s",str);
			temp = atof(str);
            MPI_Send(&temp,1,MPI_DOUBLE,i,3,MPI_COMM_WORLD); //vy
			fscanf(file,"%s",str);
			temp = atof(str);
            MPI_Send(&temp,1,MPI_DOUBLE,i,4,MPI_COMM_WORLD); //m
        }
    }

    else
    {
        MPI_Recv(&x,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&status);
        MPI_Recv(&y,1,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&status);
        MPI_Recv(&vx,1,MPI_DOUBLE,0,2,MPI_COMM_WORLD,&status);
        MPI_Recv(&vy,1,MPI_DOUBLE,0,3,MPI_COMM_WORLD,&status);
        MPI_Recv(&m,1,MPI_DOUBLE,0,4,MPI_COMM_WORLD,&status);
    }

    for (int niter=0; niter<NUM_ITER; niter++) {

        //for (i=0; i< noOfObjects; i++) {
            x_new=x;
            y_new=y;
            vx_new=vx;
            vy_new=vy;
        //}


        // Gather all data down to all the processes
        //MPI_Allgather(&x, noOfObjects, MPI_DOUBLE, x_other, noOfObjects, MPI_DOUBLE, MPI_COMM_WORLD);
        //MPI_Allgather(&y, noOfObjects, MPI_DOUBLE, y_other, noOfObjects, MPI_DOUBLE, MPI_COMM_WORLD);
        //MPI_Allgather(&vx, noOfObjects, MPI_DOUBLE, vx_other, noOfObjects, MPI_DOUBLE, MPI_COMM_WORLD);
        //MPI_Allgather(&vy, noOfObjects, MPI_DOUBLE, vy_other, noOfObjects, MPI_DOUBLE, MPI_COMM_WORLD);
        //MPI_Allgather(&m, noOfObjects, MPI_DOUBLE, m_other, noOfObjects, MPI_DOUBLE, MPI_COMM_WORLD);

        // for(i=0; i<noOfObjects; i++){
        //     k=i+1;
        //     MPI_Recv(&x_other[i],1,MPI_DOUBLE,i,0,MPI_COMM_WORLD,&status);
        //     MPI_Recv(&y_other[i],1,MPI_DOUBLE,i,1,MPI_COMM_WORLD,&status);
        //     MPI_Recv(&vx_other[i],1,MPI_DOUBLE,i,2,MPI_COMM_WORLD,&status);
        //     MPI_Recv(&vy_other[i],1,MPI_DOUBLE,i,3,MPI_COMM_WORLD,&status);
        //     MPI_Recv(&m_other[i],1,MPI_DOUBLE,i,4,MPI_COMM_WORLD,&status);
        //
        //     if(k == noOfObjects)
        //         k = 0;
        //     MPI_Send(&x,1,MPI_DOUBLE,k,0,MPI_COMM_WORLD); //x
        //     MPI_Send(&y,1,MPI_DOUBLE,k,1,MPI_COMM_WORLD); //y
        //     MPI_Send(&vx,1,MPI_DOUBLE,k,2,MPI_COMM_WORLD); //vx
        //     MPI_Send(&vy,1,MPI_DOUBLE,k,3,MPI_COMM_WORLD); //vy
        //     MPI_Send(&m,1,MPI_DOUBLE,k,4,MPI_COMM_WORLD); //m
        //
        // }

        for (i=0; i < noOfObjects; i++) {
            if (my_rank==i)
                continue;

            MPI_Send(&x,1,MPI_DOUBLE,i,0,MPI_COMM_WORLD); //x
            MPI_Send(&y,1,MPI_DOUBLE,i,1,MPI_COMM_WORLD); //y
            MPI_Send(&vx,1,MPI_DOUBLE,i,2,MPI_COMM_WORLD); //vx
            MPI_Send(&vy,1,MPI_DOUBLE,i,3,MPI_COMM_WORLD); //vy
            MPI_Send(&m,1,MPI_DOUBLE,i,4,MPI_COMM_WORLD); //m
        }

        _Bool showData = false;

        if (niter % NUM_ITER_SHOW == 0)
            showData = true;

        if (showData)
            printf("***** ITERATION %d *****\n",niter);

        //for (i=0; i< noOfObjects; i++) {
            double ax_total=0;
            double ay_total=0;
            for (j=0; j < noOfObjects; j++) {
                if (my_rank==j)
                    continue;

                MPI_Recv(&x_other,1,MPI_DOUBLE,j,0,MPI_COMM_WORLD,&status);
                MPI_Recv(&y_other,1,MPI_DOUBLE,j,1,MPI_COMM_WORLD,&status);
                MPI_Recv(&vx_other,1,MPI_DOUBLE,j,2,MPI_COMM_WORLD,&status);
                MPI_Recv(&vy_other,1,MPI_DOUBLE,j,3,MPI_COMM_WORLD,&status);
                MPI_Recv(&m_other,1,MPI_DOUBLE,j,4,MPI_COMM_WORLD,&status);

                double d = sqrt(fabs((x_other-x) * (x_other-x) + (y_other-y) * (y_other-y)));
                double f = G * ((m_other*m)/(d*d));
                double fx = f * ((x_other-x)/d);
                double ax = fx / m;
                double fy = f * ((y_other-y)/d);
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
                printf("New position of object %d: %.2f, %.2f\n",my_rank,x_new,y_new);

        //}  // noOfObjects

        //for (i=0; i< noOfObjects; i++) {
            x=x_new;
            y=y_new;
            vx=vx_new;
            vy=vy_new;
        //}

    }  // nIter
    MPI_Finalize();
}  // main
