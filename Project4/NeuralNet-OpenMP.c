/*
    Author: Makarios Christakis
    Description:
    Feedforward multi layer neural network implementation, parallelized for CPUs
    using the OpenMP API.

    Trained using the MNIST fashion dataset, after normalising the pixel values and
    initialising the neuron weights from a standard normal distribution.
    
    Training using the parameters below (500 epochs, 60k training datapoints), the whole
    program terminates in about 20 minutes. It achieves 97.4% accuracy on the training
    dataset and 85.9% accuracy on the testing set. 
    
    More info can be found in execution_info.md
*/
// **********************************************************
// DEFINITIONS
#define NL1 100            // 1st layer size
#define NL2 10             // output layer size
#define NINPUT 784         //input size
#define NTRAIN 60000       //training set size
#define NTEST 10000        //testing set size
#define ITERATIONS 500     //number of epochs
#define ALPHA (double)0.05 //learning rate
// **********************************************************
// INCLUDES
#include "extra_functions.c"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
// **********************************************************
// GLOBAL VARS
double WL1[NL1][NINPUT + 1];
double WL2[NL2][NL1 + 1];
// layer internal states
double DL1[NL1];
double DL2[NL2];
// layer outputs
double OL1[NL1];
double OL2[NL2];
// layer deltas
double delta2[NL2];
double delta1[NL1];
//data
double data_train[NTRAIN][NINPUT];
double data_test[NTEST][NINPUT];
int class_train[NTRAIN];
int class_test[NTEST];
double input[NINPUT];
// **********************************************************
// Implements the feedforward part of the Neural Network using the vector "in" as input. 
void activateNN(double *in){
    // layer1
    #pragma omp parallel for
    for (int i = 0; i < NL1; i++)
    {
        double register sum = 0;
        for (int j = 0; j < NINPUT; j++)
        {
            sum += WL1[i][j] * in[j];
        }
        sum += WL1[i][NINPUT]; //add bias neuron weight
        DL1[i] = sum;
        OL1[i] = logistic(sum);
    }
    // layer2
    #pragma omp parallel for
    for (int i = 0; i < NL2; i++)
    {
        double register sum = 0;
        for (int j = 0; j < NL1; j++)
        {
            sum += WL2[i][j] * OL1[j];
        }
        sum += WL2[i][NL1]; //add bias neuron weight
        DL2[i] = sum;
        OL2[i] = logistic(sum);
    }
}

void trainNN(double *in,double *desired){
   
    // Calculate Neural Network outputs
    activateNN(in);
    // Output layer deltas
    #pragma omp parallel
    {
    #pragma omp for
    for (int i = 0; i < NL2; i++)
    {
        delta2[i] = (OL2[i]-desired[i])*OL2[i]*(1-OL2[i]);
    }
    // Layer 1 Deltas
    #pragma omp for
    for (int i = 0; i < NL1; i++)
    {
        double register sum = 0;
        for (int j = 0; j < NL2; j++)
        {
            sum += WL2[j][i] * delta2[j];
        }
        double register Oi = OL1[i];
        delta1[i] = sum * Oi * (1-Oi);
    }
    }
    
    #pragma omp parallel
    {
    // update weights of layer 2
    #pragma omp for nowait
    for (int i = 0; i < NL2; i++)
    {
        for (int j = 0; j < NL1; j++)
        {
            WL2[i][j] -= ALPHA * OL1[j] * delta2[i];
        }
        WL2[i][NL1] -= ALPHA * delta2[i];//update bias neuron weight
    }
    // update weights of layer 1
    #pragma omp for
    for (int i = 0; i < NL1; i++)
    {
        for (int j = 0; j < NINPUT; j++)
        {
            WL1[i][j] -= ALPHA * in[j] * delta1[i];
        }
        WL1[i][NINPUT] -= ALPHA * delta1[i];//update bias neuron weight
    }
    }
}
// **********************************************************
// Evaluates which class the network predicts that the input belongs to
// by finding the argmax of the output layer. 
// Afterwards it updates the confusion matrix with the prediction.
void evaluate(int inputClass,double confMatrix[NL2][NL2]){
    int maxIndex = 0;
    double maxVal = 0;
    for (int i = 0; i < NL2; i++)
    {
        if (maxVal<OL2[i])
        {
            maxVal = OL2[i];
            maxIndex = i;
        }
    }
    //Edge case where both the correct output layer and another one have the same output value, we consider that a correct classification.
    if (maxVal==OL2[inputClass])
    {
        maxIndex = inputClass;
    }
    confMatrix[maxIndex][inputClass]++;
}

// **********************************************************
// Normalizes the input data into normal distribution N(0,1)
void normalizeData(double in[][NINPUT],int inSize){
    double average[NINPUT] = {0};
    double var[NINPUT] = {0};
    #pragma omp parallel for
    for (int i = 0; i < NINPUT; i++)
    {
        for (int j = 0; j < inSize; j++)//calculate mean
        {
            average[i] += in[j][i];
        }
        average[i] /= inSize;
        double register mean = average[i];
        for (int j = 0; j < inSize; j++)//calculate variance
        {
            var[i] += (in[j][i] - mean)*(in[j][i] - mean);
        }
        var[i] /= inSize-1;
    }
    #pragma omp parallel for
    for (int i = 0; i < NINPUT; i++)
    {
        double register mean = average[i];
        double register stddev = sqrt(var[i]);
        for (int j = 0; j < inSize; j++)
        {
            in[j][i] -= mean;
            in[j][i] /= stddev;
        }
    }
}
// **********************************************************
int main() {
    double desiredOut[NL2]={0};
    double confusionMatrixTrain[NL2][NL2]= {0};
    double confusionMatrixTest[NL2][NL2]= {0};
    readfile("./DATA/fashion-mnist_train.csv",class_train,data_train,NTRAIN);
    readfile("./DATA/fashion-mnist_test.csv",class_test,data_test,NTEST);
    normalizeData(data_test,NTEST);
    normalizeData(data_train,NTRAIN);
    initVecs();//initialise weights
    for (int i = 0; i < NL2; i++)//initialise desired vector values
    {
        desiredOut[i] = 0.1;
    }
    int register tmp = 0;
    for (int i = 0; i < NTRAIN*ITERATIONS; i++)//train the nn
    {
        tmp = rand()%NTRAIN;
        desiredOut[class_train[tmp]] = 0.9;
        trainNN(data_train[tmp],desiredOut);
        desiredOut[class_train[tmp]] = 0.1;
    }
    printf("TRAINING FINISHED!\n\n");

    for (int i = 0; i < NTRAIN; i++)//test with training set
    {
        activateNN(data_train[i]);
        evaluate(class_train[i],confusionMatrixTrain);
    }

    for (int i = 0; i < NTEST; i++)//test with testing set
    {
        activateNN(data_test[i]);
        evaluate(class_test[i],confusionMatrixTest);
    }
    double register testCorrect = 0;
    double register trainCorrect = 0;
    for (int i = 0; i < NL2; i++)
    {
        testCorrect += confusionMatrixTest[i][i];
        trainCorrect += confusionMatrixTrain[i][i];
    }
    double totalCorrect = testCorrect + trainCorrect;
    testCorrect /= (double)NTEST;
    trainCorrect /= (double)NTRAIN;
    totalCorrect /= ((double)NTEST+(double)NTRAIN);

    printf("TRAINING SAMPLES CONFUSION MATRIX:\n");
    printTable(confusionMatrixTrain);
    printf("TESTING SAMPLES CONFUSION MATRIX:\n");
    printTable(confusionMatrixTest);
    printf("Correct rate in training samples: %0.3f\n",trainCorrect);
    printf("Correct rate in testing samples: %0.3f\n",testCorrect);
    printf("Overall hit rate: %0.3f\n",totalCorrect);
    printf("Learning rate = %0.4f\n",ALPHA);
    printf("EPOCHS = %d\n",(int)ITERATIONS);
    return 0;
}