################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np
import copy
from matplotlib import pyplot as plt


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """
    img=img/255
    return img


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    temp = np.zeros((len(labels),num_classes))
    for i in range(len(labels)):
        temp[i,labels[i].astype(int)] = 1
    labels = temp.copy()
    return labels


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    #raise NotImplementedError("Softmax not implemented")
    exps = np.exp(x - x.max(axis=1)[:,np.newaxis])
    return exps / np.sum(exps, axis=1)[:, np.newaxis]


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        #raise NotImplementedError("Sigmoid not implemented")
        self.x = x.copy()
        exps = 1 + np.exp(-x)
        return 1/exps

    def tanh(self, x):
        """
        Implement tanh here.
        """
        #raise NotImplementedError("Tanh not implemented")
        self.x = x.copy()
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        #raise NotImplementedError("ReLu not implemented")
        self.x = x.copy()
        zero_mat = np.zeros(x.shape)
        return np.maximum(zero_mat, x)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        #raise NotImplementedError("Sigmoid gradient not implemented")
        return self.sigmoid(self.x)*(1-self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        #raise NotImplementedError("tanh gradient not implemented")
        return 1 - self.tanh(self.x)**2

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        #raise NotImplementedError("ReLU gradient not implemented")
        return (self.x > 0).astype(int)


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        m = np.sqrt(in_units)
        self.w = np.random.normal(0, m, size=(in_units, out_units))/m    # Declare the Weight matrix
        self.b = np.zeros((1,out_units))   # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        #raise NotImplementedError("Layer forward pass not implemented.")
        self.x = x.copy()
        self.a = self.x@self.w + self.b
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        #raise NotImplementedError("Backprop for Layer not implemented.")
        self.d_x = delta@self.w.T         # save delta to pass to next layer
        self.d_b = np.sum(delta, axis=0)  # save bias gradient for bias update
        self.d_w = self.x.T@delta         # save weight gradient for weight update
        return self.d_x


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        #raise NotImplementedError("Forward not implemented for NeuralNetwork")
        self.x = x.copy()
        self.y = x.copy()
        self.targets = targets
        for i in range(len(self.layers)):
            self.y = self.layers[i](self.y)
        self.y = softmax(self.y)
        if self.targets is not None:
            return self.y, self.loss(self.y, self.targets)
        else:
            return self.y
        
        
    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        #raise NotImplementedError("Loss not implemented for NeuralNetwork")
        e = -(targets*np.log(logits)).sum()
        return e#/targets.size

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        #raise NotImplementedError("Backprop not implemented for NeuralNetwork")
        delta = self.targets - self.y
        for i in range(len(self.layers)):
            delta = self.layers[-(i+1)].backward(delta)


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """

    #raise NotImplementedError("Train method not implemented")
    # initialize variables
    N = x_train.shape[0]
    N_val = x_valid.shape[0]
    B = config['batch_size']
    NB = int((N+B-1)/B)
    lr = config['learning_rate']
    num_epochs = config['epochs']
    
    early_stop_flag = config['early_stop']
    flag_point = config['early_stop_epoch']
    exit_flag = 0
    
    reg_const = config['L2_penalty']
    
    # initialize storage variables for accuracies and losses
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    # determine momentum variable
    if config['momentum']:
        gamma = config['momentum_gamma']
    else:
        gamma = 0
    
    # iterate through epochs to update weights and minimize CE-loss
    for i in range(num_epochs):
        print('epochs: ', i)
        shuf_ind = np.random.permutation(range(N))
        # initialize velocity terms for momentum GSD
        vel_w = []
        vel_b = []
        acc_accum = 0
        loss_accum = 0
        for layer in model.layers:
            if isinstance(layer, Layer):
                vel_w.append(np.zeros(layer.w.shape))
                vel_b.append(np.zeros(layer.b.shape))
            else:
                vel_w.append(None)
                vel_b.append(None)
        for j in range(NB):
            # use shuffled indices to define mini-batch
            mb_ind = shuf_ind[B*j:min(B*(j+1), N)]
            # pass mini-batch through forward operation of net
            train_out, train_loss = model(x_train[mb_ind, :], targets=y_train[mb_ind, :])
            train_acc = np.sum(np.argmax(train_out, axis=1)==np.argmax(y_train[mb_ind, :], axis=1))/len(mb_ind)
            #accumulate losses and accuracies (averaged across each mini-batch per epoch)
            acc_accum += 1/NB*train_acc
            loss_accum += train_loss
            
            # use backprop to calculate gradients and update weights
            model.backward()
            for k, layer in enumerate(model.layers):
                if isinstance(layer, Layer):
                    vel_w[k] = gamma*vel_w[k] + lr*layer.d_w
                    vel_b[k] = gamma*vel_b[k] + lr*layer.d_b
                    
                    layer.w += vel_w[k]
                    layer.b += vel_b[k]
                    
                    layer.w *= (1-reg_const)
                    layer.b *= (1-reg_const)
        
        # save train accuracy and loss per epoch
        train_accs.append(acc_accum)
        train_losses.append(loss_accum)
        
        # check val loss
        val_out, val_loss = model(x_valid, targets=y_valid)
        val_acc = np.sum(np.argmax(val_out, axis=1)==np.argmax(y_valid, axis=1))/N_val
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        
        print('train accuracy: ', train_accs[i])
        print('train loss: ', train_losses[i])
        
        print('val accuracy: ', val_accs[i])
        print('val loss: ', val_losses[i])
        
        # early stop condition check for monotonically increasing val loss across five epochs
        if early_stop_flag and i > flag_point:
            if val_losses[i] > val_losses[i-1]:
                min_loss = min(val_losses[:i])#.min()
                if val_losses[i] < min_loss:
                    best_model = copy.deepcopy(model)
                exit_flag += 1
            else:
                exit_flag = 0
            if exit_flag >= flag_point:
                i = num_epochs
                model = copy.deepcopy(model)
    
    
    return np.array(train_losses), np.array(train_accs), np.array(val_losses), np.array(val_accs)
                
            
def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """

    #raise NotImplementedError("Test method not implemented")
    # forward pass test set through trained model to calculate loss and accuracy
    test_out, test_loss = model(X_test, targets=y_test)
    test_acc = np.sum(np.argmax(test_out, axis=1)==np.argmax(y_test, axis=1))/X_test.shape[0]
        
    print('test accuracy: ', test_acc)
    print('test loss: ', test_loss)
    return test_acc


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    # x_valid, y_valid = ...
    x_valid, y_valid = x_train[range(5000,15000),:], y_train[range(5000,15000),:]
    x_train, y_train = np.delete(x_train, range(5000,15000), 0), np.delete(y_train, range(5000,15000), 0)

    # train the model
    train_losses, train_accs, val_losses, val_accs = train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)

    # graph losses and accuracies of train and validation sets
    f, axs = plt.subplots(2,2,figsize=(10,5))
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(2,2,1)
    plt.plot([i for i in range(len(train_losses))], train_losses)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.title('Training Loss per Epoch')
    
    plt.subplot(2,2,2)
    plt.plot([i for i in range(len(train_accs))], train_accs)
    plt.xlabel('epochs')
    plt.ylabel('train accuracy')
    plt.title('Training Accuracy per Epoch')
    
    plt.subplot(2,2,3)
    plt.plot([i for i in range(len(val_losses))], val_losses)
    plt.xlabel('epochs')
    plt.ylabel('val loss')
    plt.title('Validation Loss per Epoch')
    
    plt.subplot(2,2,4)
    plt.plot([i for i in range(len(val_accs))], val_accs)
    plt.xlabel('epochs')
    plt.ylabel('val accuracy')
    plt.title('Validation Accuracy per Epoch')
    
    plt.suptitle('Mini-Batch SGD Training Results with Activation: '+config['activation']+' with Learning Rate: '+str(config['learning_rate']))
    plt.show()