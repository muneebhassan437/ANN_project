# ANN_project
Developed this ANN model that predicts two binary outputs (y1 and y2) with given inputs (x1 and x2) using the text file data.txt. 

# Sensitivity_Analysis
I improved this model using sensitivity analysis. Initially, the model was causing **over-fitting** as there was a huge gap between the train and validation accuracy. I kept the train_split at 0.01, which meant I
1 percent of the data. I started with a lower number of neurons and gradually increased them. For learning rate, I chose a moderate value of 0.006. I kept the epochs value at 15000 which was big enough for values
to converge and small enough to avoid over-shooting. 
With these results, I was able to reach a train_cost accuracy of 90% and validation_cost accuracy of 84%.

# How to run the code
Make sure all the files are in the same directory. Open VS code terminal and type "gcc main.c mymodel.c -lsodium -lm" within the directory where the files are stored. This will create the object code.
To run the object code, type "./a.out"
