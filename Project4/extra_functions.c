#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// **********************************************************
// DEFINITIONS
#ifndef NL1
#define NL1 2 //layer size
#define NL2 2
#define NINPUT 12 //input size
#endif

// **********************************************************
// VARS
extern double input[NINPUT];
extern double WL1[NL1][NINPUT + 1];
extern double WL2[NL2][NL1 + 1];

// **********************************************************
// Implements the logistic sigmoid function.
double logistic(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// **********************************************************
// Prints a vector of vecSize dimensions.
void printvec(double *vec, int vecSize) {
    for (int i = 0; i < vecSize; i++) {
        printf("%0.5f\t", vec[i]);
    }
    printf("\n");
}
// **********************************************************
// Samples a standard normal distribution [~N(0,1)]
// using the Marsaglia polar method: https://en.wikipedia.org/wiki/Marsaglia_polar_method
double gaussrand() {
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if (phase == 0) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while (S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    }
    else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    return X;
}

// **********************************************************
// Initializes neuron synapses' weights with values from a standard normal distribution.
// For Xavier initialization uncomment the first 2 lines and delete the next two.
void initVecs() {
    /* double register stddev1 = sqrt(2/(NINPUT+NL1));
    double register stddev2 = sqrt(2/(NL1+NL2)); */
    double register stddev1 = 1;
    double register stddev2 = 1;
    for (int i = 0; i < NL1; i++) {
        for (int j = 0; j < NINPUT + 1; j++) {
            WL1[i][j] = gaussrand() * stddev1;
        }
    }
    for (int i = 0; i < NL2; i++) {
        for (int j = 0; j < NL1 + 1; j++) {
            WL2[i][j] = gaussrand() * stddev2;
        }
    }
}

// Used for printing the confusion matrix.
// **********************************************************
void printTable(double matrix[NL2][NL2]) {
    for (int i = 0; i < NL2; i++) {
        for (int j = 0; j < NL2; j++) {
            printf("%.2f\t", matrix[i][j]);
        }
        printf("\n");
    }
}

// Used for readint the MNIST fashion dataset.
int readfile(char *filepath, int *Class, double Data[][NINPUT], int numVectors) {
    FILE *fp;
    char B[20001], *p;
    int j;

    fp = fopen(filepath, "r");
    if (fp == NULL)
        return -1;
    if (fgets(B, 20000, fp) != B)
        return -2;
    for (j = 0; j < numVectors; j++) {
        if (fgets(B, 20000, fp) != B)
            return -2;
        p = strtok(B, ",");
        if (p == NULL)
            return -3;
        Class[j] = atoi(p);

        for (int i = 0; i < NINPUT; i++) {
            p = strtok(NULL, ",\n");
            Data[j][i] = atof(p);
        }
    }
    printf("Loaded %d examples\n", j);

    fclose(fp);
    return 0;
}