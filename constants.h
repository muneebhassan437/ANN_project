#ifndef CONSTANT_H
#define CONSTANT_H

//This is the total number of inputs; x1 and x2
#define num_inputs 2          // N1 = 2

//This is the total number of neurones in the second layer(L=2) which is 40
#define num_neurons_layer2 40 // N2 = 40

//This is the total number of neurones in the third layer(L=3) which is 20
#define num_neurons_layer3 20 // N3 = 20

//This is the total number of outputs; Y1 and Y2
#define num_outputs 2         // N4 = 2

//This represents the range of initial random values used to initialize the weights and biases. 
#define initial_range 0.2

//The learning rate in backward propagation.
#define Learning_rate 0.006 // 0 - 1, 0.01 - 0.0001

//Maximum number of epochs (The total number of training iterations).
#define epochs 15000

//These are the maximum number of rows in the 'data.txt' file
#define MAX_ROWS 48120

//Maximum number of columns. X1,X2,Y1,Y2
#define MAX_COLS 4

//This is the percent of data used for train
#define train_split 0.01 // 0.3 percent of data will be used for train

#endif