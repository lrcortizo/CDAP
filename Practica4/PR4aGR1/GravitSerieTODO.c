#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "mpi.h"

#define MAX_CHAR 100
#define MAX_OBJECTS 10
#define DATAFILE "data.bin"
#define G 6.674e-11
#define NUM_ITER 100001
#define NUM_ITER_SHOW 5000
#define verbose false

void main(int argc, char* argv[]){
    char  str[MAX_CHAR];
    FILE *file_initial;
    MPI_File file;
    int noOfObjects;
    double buffer[5];
    int bufferaux[1];

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

    MPI_Win positions_win;

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
    MPI_Type_contiguous(5, MPI_DOUBLE, &object_type);
    MPI_Type_commit(&object_type);
    double positions[6];

    printf("\n");
    if (my_rank == 0){
        file_initial = fopen( DATAFILE , "rb+");
        fread(bufferaux,1,sizeof(int),file_initial);

        noOfObjects = bufferaux[0];
        //printf("Hola, soy %d y leí: %d", my_rank, noOfObjects);
        if (p!=noOfObjects) {
            printf("Error: El número de procesos debe ser igual al número de objetos. Hay %d objetos.\n", noOfObjects);
            exit(0);
        }

        if (noOfObjects > MAX_OBJECTS) {
            printf("*** ERROR: maximum no. of objects exceeded ***\n");
            exit(0);
        }
        fclose(file_initial);
        // MPI_Win_create(&positions,sizeof(double)*6,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&positions_win);
    }
    else{
         // MPI_Win_create(MPI_BOTTOM,0,1,MPI_INFO_NULL,MPI_COMM_WORLD,&positions_win);
    }

    MPI_Win_create(&positions,sizeof(double)*6,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&positions_win);


    // MPI_Win_fence(0,positions_win);
    MPI_File_open(MPI_COMM_WORLD, DATAFILE, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    MPI_File_read_at(file, my_rank*5*sizeof(double)+sizeof(int), buffer, 5, MPI_DOUBLE, &status);
    MPI_File_close(&file);

    printf("Hola, soy %d y leí: ",my_rank);
    for (int i=0; i<5; i++){
        printf("%f ",buffer[i]);
    }
    printf("\n");

    data.x = x = buffer[0];
    data.y = y = buffer[1];
    data.vx = vx = buffer[2];
    data.vy = vy = buffer[3];
    data.m = m = buffer[4];

    MPI_File_close(&file);

    //printf(" Im %d and I receive data: %f, %f, %f, %f, %f\n", my_rank, x, y, vx, vy, m);

    for (int niter=0; niter<NUM_ITER; niter++) {

        x_new = x;
        y_new = y;
        vx_new = vx;
        vy_new = vy;

        _Bool showData = false;

        if (niter % NUM_ITER_SHOW == 0)
            showData = true;

        if (showData && my_rank == 0)
            printf("***** ITERATION %d *****\n",niter);

        double ax_total=0;
        double ay_total=0;

        for(int j = 0; j<p;j++){

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



        if (showData) {
            MPI_Win_fence(0,positions_win);
            if(my_rank != 0){
                MPI_Put(&x_new, 1, MPI_DOUBLE, 0, 2*my_rank, 1, MPI_DOUBLE, positions_win);
                MPI_Put(&y_new, 1, MPI_DOUBLE, 0, 2*my_rank+1, 1, MPI_DOUBLE, positions_win);



            }else{
                for(int i = 0; i < p; i++){
                    printf("New position of object %d: %.2f, %.2f\n", i, positions[2*i], positions[2*i+1]);
                }
            }
            MPI_Win_fence(0,positions_win);
        }
        // MPI_Win_fence(0,positions_win);

        x=x_new;
        y=y_new;
        vx=vx_new;
        vy=vy_new;

    }  // nIter
    MPI_Win_free(&positions_win);
    MPI_Finalize();

}  // main
