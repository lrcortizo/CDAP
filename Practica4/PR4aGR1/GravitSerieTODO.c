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
    } data;

    double x, y, vx, vy, m;
	double x_new, y_new, vx_new, vy_new;

    int my_rank;
    int p;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int block_counts[5];                    /* Array of lengths */
    MPI_Aint offsets[5];
    MPI_Aint extent;                        /* Array of displacements */
    MPI_Datatype typearray[5];              /* Array of MPI datatypes */
    MPI_Type_extent(MPI_DOUBLE, &extent);

    MPI_Status status;
    MPI_Datatype object_type;

    block_counts[0] = block_counts[1] = block_counts[2] = block_counts[3] = block_counts[4] = 1;
    typearray[0] = typearray[1] = typearray[2] = typearray[3] = typearray[4] = MPI_DOUBLE;
    offsets[0] = 0;
    offsets[1] = extent;
    offsets[2] = 2*extent;
    offsets[3] = 3*extent;
    offsets[4] = 4*extent;

    MPI_Type_struct(5, block_counts, offsets, typearray, &object_type);
    MPI_Type_commit(&object_type);

    printf("\n");

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

        for (int i=1; i< noOfObjects; i++) {
    			fscanf(file,"%s",str);
    			data.x = atof(str);
    			fscanf(file,"%s",str);
    			data.y = atof(str);
    			fscanf(file,"%s",str);
    			data.vx = atof(str);
    			fscanf(file,"%s",str);
    			data.vy = atof(str);
    			fscanf(file,"%s",str);
    			data.m = atof(str);
          //printf("Estoy aquí\n");
          MPI_Send(&data,1,object_type,i,0,MPI_COMM_WORLD);
        }
    }

    else
    {
        MPI_Recv(&data,1,object_type,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        x = data.x;
        y = data.y;
        vx = data.vx;
        vy = data.vy;
        m = data.m;
    }
    fclose(file);

    printf(" Im %d and I receive data: %f, %f, %f, %f, %f\n", my_rank, x, y, vx, vy, m);

    for (int niter=0; niter<NUM_ITER; niter++) {

        x_new = x;
        y_new = y;
        vx_new = vx;
        vy_new = vy;

        _Bool showData = false;

        if (niter % NUM_ITER_SHOW == 0)
            showData = true;

        if (showData)
            printf("***** ITERATION %d *****\n",niter);

        double ax_total=0;
        double ay_total=0;

        for(int j = 0; j<noOfObjects;j++){

            data.x = x;
            data.y = y;
            data.vx = vx;
            data.vy = vy;
            data.m = m;

            MPI_Bcast(&data, 1, object_type, j, MPI_COMM_WORLD);

            if (my_rank == j){
                continue;
            }

            double d = sqrt((data.x-x) * (data.x-x) + (data.y-y) * (data.y-y));
            double f = G * ((data.m*m)/(d*d));
            double fx = f * ((data.x-x)/d);
            double ax = fx / m;
            double fy = f * ((data.y-y)/d);
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
            printf("New position of object %d: %.2f, %.2f\n", my_rank, x_new, y_new);

        x=x_new;
        y=y_new;
        vx=vx_new;
        vy=vy_new;

    }  // nIter

    MPI_Finalize();

}  // main
