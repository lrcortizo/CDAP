#include <stdio.h>
#include <stdlib.h>

#define DATAFILE_IN "data.txt"
#define DATAFILE_OUT "data.bin"
#define MAX_CHAR 100

void main() {
    char  str[MAX_CHAR];
    FILE *file_in, *file_out;
    int noOfObjects;
    double data_in;
    int i;

    file_in = fopen(DATAFILE_IN , "r");
    file_out = fopen(DATAFILE_OUT, "wb");

    printf("%s -> %s\n",DATAFILE_IN,DATAFILE_OUT);

    fscanf(file_in,"%s",str);
    noOfObjects = atoi(str);
    printf("Number of objects: %d\n",noOfObjects);
    fwrite(&noOfObjects, sizeof(int), 1, file_out);

    for (i=0; i < noOfObjects*5; i++) {
        fscanf(file_in,"%s",str);
        data_in = atof(str);
        fwrite(&data_in, sizeof(double), 1, file_out);
    }
    printf("Number of data written: %d\n",i);

    fclose(file_in);
    fclose(file_out);
}