/*
    Author: Makarios Christakis
    Description: Compiled and ran on a 7th gen i7 on Ubuntu 18.04,
    the algorithm converges after 28 iterations.
    Timed using time() we get:
        real	1m15,233s
        user	1m15,074s
        sys	    0m0,137s
    So approximately each iteration took 43sec to complete.
    Mind that this time includes the initialisation of vectors.
*/
#include <stdio.h>
#include <stdlib.h>
// *********************************************************
#pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline", "unsafe-math-optimizations");
#pragma GCC option("arch=native", "tune=native", "no-zero-upper");
//**********************************************************
//DEFINITIONS **********************************************
#define N 100000           // Number of generated vectors.
#define Nv 1000            // Number of dimensions of each generated vector.
#define Nc 100             // Number of desired classes to group into.
#define THRESHOLD 0.000001 // K-means convergeance threshold.
// GLOBAL VARS *********************************************
float vectors[N][Nv];
float centres[Nc][Nv];
int classes[N];
// **********************************************************

// Initialises vectors to random normalized values.
void initialiseVecs() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < Nv; j++) {
            vectors[i][j] = 1.0 * rand() / RAND_MAX;
        }
    }
}

// **********************************************************
// Copies vector A to vector B.
void cpyVec(float *A, float *B) {
    for (int i = 0; i < Nv; i++) {
        B[i] = A[i];
    }
}

// **********************************************************
// Initialises class centroids using random vectors from the generated dataset.
void initCentres() {
    int temp[Nc];
    int sel, flag = 0;

    for (int i = 0; i < Nc; i++) {
        sel = rand() % N;
        for (int j = 0; j < i; j++) {
            if (sel == temp[j]) {
                flag = 1;
                break;
            }
        }
        if (flag == 1) {
            flag = 0;
            i--;
        }
        else {
            temp[i] = sel;
            cpyVec(&vectors[sel][0], &centres[i][0]);
        }
    }
}

// **********************************************************
// Optimized euclidean distance calculation between 2 vectors.
float dist(float *restrict A, float *restrict B) {
    float sum = 0;
    for (int i = 0; i < Nv; i++) {
        sum += (A[i] - B[i]) * (A[i] - B[i]);
    }
    return sum;
}

// **********************************************************
// Classifies each example vector by finding the centroid with the shortest distance
// to it. This function returns the sum of all the minimum distances in order to check for convergeance.
float computeClasses() {
    float tempdist = 0;
    float sumdists = 0;
    for (int i = 0; i < N; i++) {
        float min = 1.0 * RAND_MAX;
        for (int j = 0; j < Nc; j++) {
            tempdist = dist(&vectors[i][0], &centres[j][0]);
            if (tempdist < min) {
                classes[i] = j;
                min = tempdist;
            }
        }
        sumdists += min;
    }
    return sumdists;
}

// **********************************************************
// Adds vector A to vector B.
void addvec(float *A, float *B) {
    for (int i = 0; i < Nv; i++) {
        B[i] += A[i];
    }
}

// **********************************************************
// Initialises a vector with zeros.
void resetVec(float *A) {
    for (int i = 0; i < Nv; i++) {
        A[i] = 0;
    }
}

// **********************************************************
// Computes all class centroids using the mean of all vectors of each class.
void computeCentres() {
    float count[Nc] = {0};

    for (int i = 0; i < Nc; i++) {
        resetVec(&centres[i][0]);
    }

    for (int i = 0; i < N; i++) {
        int tmp = classes[i];
        count[tmp]++;
        addvec(&vectors[i][0], &centres[tmp][0]);
    }
    
    for (int i = 0; i < Nc; i++) {
        if (count[i] == 0) {
            printf("ERROR, category %d has no vectors.\n", i);
        }
        else {
            float inv = 1 / count[i];
            for (int j = 0; j < Nv; j++) {
                centres[i][j] *= inv;
            }
        }
    }
}

// **********************************************************
// Executes the K-means clustering algorithm, checking for convergeance
// using the %change of the sum of the distances of each vector to it's class centroid.
int main() {
    float sumdist = 1e30, sumdistold;
    int i = 0;
    initialiseVecs();
    initCentres();
    do {
        i++;
        sumdistold = sumdist;
        sumdist = computeClasses();
        computeCentres();
    } while ((sumdistold - sumdist) / sumdistold > THRESHOLD);
    printf("Total distance in loop %d is %0.2f\n", i, sumdist);

    return 0;
}
