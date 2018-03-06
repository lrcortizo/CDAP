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
    struct {
        double x;
        double y;
        double vx;
        double vy;
        double m;
    } indata, recdata;

    int lengtharray[5];                     /* Array of lengths */
    MPI_Aint disparray[5];                  /* Array of displacements */
    MPI_Datatype typearray[5];              /* Array of MPI datatypes */
    MPI_Aint startaddress, address;         /* Variables used to displacements */

    lengtharray[0] = lengtharray[1] = lengtharray[2] = lengtharray[3] = lengtharray[4] = 1;
    typearray[0] = typearray[1] = typearray[2] = typearray[3] = typearray[4] = MPI_DOUBLE;
    disparray[0];

    disparray[0] = 0;
    MPI_Address(&indata.x, &startaddress);
    MPI_Address(&indata.y, &address);

    disparray[1] = address-startaddress;
    MPI_Address(&indata.vx, &address);
    disparray[2] = address-startaddress;
    MPI_Address(&indata.vy, &address);
    disparray[3] = address-startaddress;
    MPI_Address(&indata.m, &address);
    disparray[4] = address-startaddress;



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
    MPI_Datatype object_type;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    file = fopen( DATAFILE , "r");
    fscanf(file,"%s",str);
    noOfObjects = atoi(str);
    if (noOfObjects > MAX_OBJECTS) {
        printf("*** ERROR: maximum no. of objects exceeded ***\n");
        exit(0);
    }

    if (p!=3) {
        if (my_rank==0) {
            printf("Error: El número de procesos debe ser igual al número de objetos. Hay %d objetos.\n", noOfObjects);
        }
        exit(0);
    }

    printf("\n");

    if(my_rank == 0)
    {
        fscanf(file,"%s",str);
        recdata.x = atof(str);
        fscanf(file,"%s",str);
        recdata.y = atof(str);
        fscanf(file,"%s",str);
        recdata.vx = atof(str);
        fscanf(file,"%s",str);
        recdata.vy = atof(str);
        fscanf(file,"%s",str);
        recdata.m = atof(str);

        for (i=1; i< noOfObjects; i++) {
			fscanf(file,"%s",str);
			indata.x = atof(str);
			fscanf(file,"%s",str);
			indata.y = atof(str);
			fscanf(file,"%s",str);
			indata.vx = atof(str);
			fscanf(file,"%s",str);
			indata.vy = atof(str);
			fscanf(file,"%s",str);
			indata.m = atof(str);
            MPI_Send(&indata,1,object_type,i,4,MPI_COMM_WORLD); //m
        }
    }

    else
    {
        MPI_Recv(&recdata,1,object_type,0,0,MPI_COMM_WORLD,&status);
    }

    for (int niter=0; niter<NUM_ITER; niter++) {

        x_new=x;
        y_new=y;
        vx_new=vx;
        vy_new=vy;

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

        x=x_new;
        y=y_new;
        vx=vx_new;
        vy=vy_new;

    }  // nIter
    MPI_Finalize();
}  // main
