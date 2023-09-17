#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sodium.h> // For randombytes_buf function

#include "constants.h"
#include "mymodel.h"


int main()
{
    double data[MAX_ROWS][MAX_COLS];
    ReadDataFromFile("data.txt", data);

    int num_train = MAX_ROWS * train_split;
    int num_val = MAX_ROWS * (1 - train_split);
    // int num_train = 120;
    // int num_val = MAX_ROWS - 120;

    double X_train[num_train][num_inputs];
    double Y_train[num_train][num_outputs];
    double X_val[num_val][num_inputs];
    double Y_val[num_val][num_outputs];

    for (int row = 0; row < num_train; row++)
    {
        X_train[row][0] = data[row][0];
        X_train[row][1] = data[row][1];
        Y_train[row][0] = data[row][2];
        Y_train[row][1] = data[row][3];
    }

    for (int row = num_train; row < MAX_ROWS; row++)
    {
        X_val[row - num_train][0] = data[row][0];
        X_val[row - num_train][1] = data[row][1];
        Y_val[row - num_train][0] = data[row][2];
        Y_val[row - num_train][1] = data[row][3];
    }

    double W2[num_neurons_layer2][num_inputs];
    double b2[num_neurons_layer2][1];

    double W3[num_neurons_layer3][num_neurons_layer2];
    double b3[num_neurons_layer3][1];

    double W4[num_outputs][num_neurons_layer3];
    double b4[num_outputs][1];

    // Initialize W2 and b2 arrays with random values between -a and +a
    for (int i = 0; i < num_neurons_layer2; i++)
    {
        for (int j = 0; j < num_inputs; j++)
        {
            W2[i][j] = random_double(-initial_range, initial_range);
        }
        b2[i][0] = random_double(-initial_range, initial_range);
    }

    // Initialize W3 and b3 arrays with random values between -a and +a
    for (int i = 0; i < num_neurons_layer3; i++)
    {
        for (int j = 0; j < num_neurons_layer2; j++)
        {
            W3[i][j] = random_double(-initial_range, initial_range);
        }
        b3[i][0] = random_double(-initial_range, initial_range);
    }

    // Initialize W4 and b4 arrays with random values between -a and +a
    for (int i = 0; i < num_outputs; i++)
    {
        for (int j = 0; j < num_neurons_layer3; j++)
        {
            W4[i][j] = random_double(-initial_range, initial_range);
        }
        b4[i][0] = random_double(-initial_range, initial_range);
    }

    // the final output of each layer, every column is for a set of inputs
    double a2[num_neurons_layer2][num_train];
    double a3[num_neurons_layer3][num_train];
    double a4[num_outputs][num_train];

    for (int ep = 0; ep <= epochs; ep++)
    {
        // ###################################################### ForwardPass start

        ForwardPass(num_train, X_train, Y_train,
                    W2, W3, W4,
                    b2, b3, b4,
                    a2, a3, a4);
        // ###################################################### end of ForwardPass

        // ###################################################### BackwardPass start
        BackwardPass(num_train, X_train, Y_train,
                     W2, W3, W4,
                     b2, b3, b4,
                     a2, a3, a4);

        // ###################################################### end of BackwardPass

        // QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
        // ###################################################### Evaluation of accuracies starts

        Evaluation(ep, num_train, X_train, Y_train,
                   W2, W3, W4,
                   b2, b3, b4,
                   a2, a3, a4, num_val, X_val, Y_val);
    }
}
