/*
    Author: Makarios Christakis
    Compiled and ran on a 7th gen i7 on Ubuntu 18.04,
    the algorithm converges after 31 iterations.
    Timed using time() we get:
    real	0m53,564s
    user	6m40,944s
    sys	    0m0,176s

    NOTE: SIMD optimization in the euclidian distance function causes the program
    to calculate incorrect results. This is unfortunate because that is the most 
    called function, so it would be most beneficial to parallelize it.I left the
    pragma clause I used commented for reference.
*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

//DEFINITIONS **********************************************
#define N 100000           // Number of generated vectors.
#define Nv 1000            // Number of dimensions of each generated vector.
#define Nc 100             // Number of desired classes to group into.
#define THRESHOLD 0.000001 // K-means convergeance threshold.
#define NUM_CORES 8
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
#pragma omp simd
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
float dist(float *A, float *B) {
    float sum = 0;

    // commented out because it causes wrong results sometimes
    //#pragma omp simd reduction(+:sum)
    for (int i = 0; i < Nv; i++) {
        float x = A[i] - B[i];
        sum += x * x;
    }
    return sum;
}

// **********************************************************
// Classifies each example vector by finding the centroid with the shortest distance
// to it. This function returns the sum of all the minimum distances in order to check for convergeance.
float computeClasses() {
    float tempdist = 0;
    float sumdists = 0;
    #pragma omp parallel for private(tempdist) reduction(+:sumdists) shared(classes)
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
#pragma omp simd
    for (int i = 0; i < Nv; i++) {
        B[i] += A[i];
    }
}

// **********************************************************
// Initialises a vector with zeros.
void resetVec(float *A) {
#pragma omp simd
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
#pragma omp parallel for shared(count) schedule(dynamic)
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
        printf("Total distance in loop %d is %0.2f\n", i, sumdist);
        computeCentres();
    } while ((sumdistold - sumdist) / sumdistold > THRESHOLD);

    return 0;
}
