#ifndef MY_MODEL_H
#define MY_MODEL_H

#include "constants.h"
//This is the sigmoid activation funcction declaration
double sigmoid(double x);

//This is the random_double declaration
double random_double(double min, double max);

//This is the declaration to read data from file
int ReadDataFromFile(const char *filename, double data[MAX_ROWS][MAX_COLS]);

//Declaration for dorward pass
void ForwardPass(int num_train, double X_train[][num_inputs], double Y_train[][num_outputs],
                 double W2[][num_inputs], double W3[][num_neurons_layer2], double W4[][num_neurons_layer3],
                 double b2[][1], double b3[][1], double b4[][1],
                 double a2[][num_train], double a3[][num_train], double a4[][num_train]);

//Declaration for backward pass               
void BackwardPass(int num_train, double X_train[][num_inputs], double Y_train[][num_outputs],
                  double W2[][num_inputs], double W3[][num_neurons_layer2], double W4[][num_neurons_layer3],
                  double b2[][1], double b3[][1], double b4[][1],
                  double a2[][num_train], double a3[][num_train], double a4[][num_train]);

//declaration for evaluation
void Evaluation(int ep,int num_train, double X_train[][num_inputs], double Y_train[][num_outputs],
                  double W2[][num_inputs], double W3[][num_neurons_layer2], double W4[][num_neurons_layer3],
                  double b2[][1], double b3[][1], double b4[][1],
                  double a2[][num_train], double a3[][num_train], double a4[][num_train], int num_val, double X_val[][num_inputs], double Y_val[][num_outputs]);

#endif
