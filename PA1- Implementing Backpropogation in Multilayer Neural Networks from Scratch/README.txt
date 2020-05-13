CSE 253 PA2
Austin Choe and Albert Tan

--------------------------
There are two .py files included in this submission. The neuralnet.py implements the functions, classes and methods necessary for training a neural network and testing it. The PA2.py file may be executed from start to finish to replicate the results included in the report.

--------------------------
neuralnet.py

Running this python script will train a neural network using the parameters listed in config.yaml

# Part A
    Validation split is implemented in the main function  

# Part B
    Use PA2.py for these results
    
# Part C
    Set config.yaml to the following parameters and run the .py file
        
        layer_specs: [784, 50, 10]
        activation: "tanh"
        learning_rate: 0.001
        batch_size: 128
        epochs: 100
        early_stop: True
        early_stop_epoch: 5
        L2_penalty: 0
        momentum: True
        momentum_gamma: 0.9

# Part D
    Change L2_penalty and epochs parameters in config.yaml as necessary to adjust regularization
    
# Part E
    Change activation parameter to tanh, sigmoid, or ReLU to change activation function

# Part F
    i) Change 2nd argument in layer_specs parameter to change number of hidden layer units
    ii) Add additional argument after 2nd entry in layer_specs parameter to add hidden layer

--------------------------
PA2.py

Running this python script will yield the results seen in the report. 
The config.yaml parameters should be set to those seen in Part C.

Gradient approximations and backpropagation gradient calculations performed after validation split.

Config parameters are adjusted within the script to demonstrate effects of changing each one.