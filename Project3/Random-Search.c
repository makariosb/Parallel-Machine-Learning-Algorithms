/*
Author: Makarios Christakis
Description:
Parallel implementation of the random search algorithm for the
travelling salesman problem.
For the parameters below the algorithm converged to:
    Final total distance: 489587.66
Timed using time() on a 7th gen i7, Ubuntu 18.04 machine we get:
    real	1m34,568s
    user	1m34,540s
    sys	    0m0,021s
Not a noticable improvement over the serial implementation, which
is due to the fact that this algorithm is not really parallelizable
and also the problem is NP-hard.

NOTE: The moveCity function is loop dependant and thus can't be parallelized.
*/
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
// **********************************************************
// DEFINITIONS
#define N_POINTS 10000 // Number of cities to generate
#define ITERATIONS 1e9 // Number of iterations to execute
// **********************************************************
// GLOBAL VARS
float cities[N_POINTS][2] = {0};
int route[N_POINTS + 1] = {0};
float totDist = 0;

// **********************************************************
// Initialises the city coordinate vectors and the chosen route
// between them.
void initVec() {
    for (int i = 0; i < N_POINTS; i++) {
        route[i] = i;
        cities[i][0] = (float)rand() / RAND_MAX * 1e3;
        cities[i][1] = (float)rand() / RAND_MAX * 1e3;
    }
    route[N_POINTS] = 0;
}
// **********************************************************
// Euclidean distance calculation between 2 points in the grid.
float dist(int p1, int p2) {
    float register dx = cities[p1][0] - cities[p2][0];
    float register dy = cities[p1][1] - cities[p2][1];
    return (float)sqrt(dx * dx + dy * dy);
}

// **********************************************************
// Swaps 2 cities and checks if the total distance is shorter.
// If it is, it updates the route taken.
void moveCity() {
    int register index1, index2;
    float tempDist = totDist;
    do {
        index1 = 1 + rand() % (N_POINTS - 1);
        index2 = 1 + rand() % (N_POINTS - 1);
    } while (index1 == index2);
    int register point1 = route[index1];
    int register point2 = route[index2];

    if (abs(index1 - index2) == 1) //when neighboring points are to be swapped, 2 distances change
    {
        //Assure that index1 = min(index1,index2)
        if (index1 > index2) {
            int tmp = index1;
            index1 = index2;
            index2 = tmp;
        }
        // subtract
        tempDist -= dist(route[index1], route[index1 - 1]);
        tempDist -= dist(route[index2], route[index2 + 1]);
        // add
        tempDist += dist(route[index1], route[index2 + 1]);
        tempDist += dist(route[index2], route[index1 - 1]);
    }
    else //In all other cases 4 distances change.
    {
        // subtract
        tempDist -= dist(route[index1], route[index1 - 1]);
        tempDist -= dist(route[index1], route[index1 + 1]);
        tempDist -= dist(route[index2], route[index2 - 1]);
        tempDist -= dist(route[index2], route[index2 + 1]);
        // add
        tempDist += dist(route[index1], route[index2 - 1]);
        tempDist += dist(route[index1], route[index2 + 1]);
        tempDist += dist(route[index2], route[index1 - 1]);
        tempDist += dist(route[index2], route[index1 + 1]);
    }

    if (tempDist < totDist) {
        route[index1] = point2;
        route[index2] = point1;
        totDist = tempDist;
    }
}

int main() {
    initVec();
    // initial total distance calculation
#pragma omp parallel for reduction(+:totDist)
    for (int i = 0; i < N_POINTS; i++) {
        totDist += dist(route[i], route[i + 1]);
    }
    printf("Starting total distance: %.2f\n", totDist);
    float startDist = totDist;
    for (int i = 0; i < ITERATIONS; i++) {
        moveCity();
    }
    printf("Final total distance: %.2f\n", totDist);
    printf("Delta: %.2f\n\n", totDist - startDist);

    return 0;
}
