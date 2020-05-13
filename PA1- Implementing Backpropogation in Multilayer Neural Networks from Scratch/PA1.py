#!/usr/bin/env python
# coding: utf-8

# In[50]:


import os, gzip
import yaml
import numpy as np
from matplotlib import pyplot as plt
import neuralnet as nn
import copy


# In[51]:


# Load the configuration.
config = nn.load_config("./")

# Load the data
x_train, y_train = nn.load_data(path="./", mode="train")
x_test,  y_test  = nn.load_data(path="./", mode="t10k")


# In[52]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## Create validation set from training set

# In[53]:


x_valid, y_valid = x_train[range(5000,15000),:], y_train[range(5000,15000),:]


# In[54]:


print(x_valid.shape)


# In[55]:


x_train, y_train = np.delete(x_train, range(5000,15000), 0), np.delete(y_train, range(5000,15000), 0)


# In[56]:


print(x_train.shape)


# ## Check gradient approximation versus gradient calculation

# In[8]:


samples = np.zeros((10,784))
samp_lbls = np.zeros((10,10))

for i in range(10):
    for j in range(x_train.shape[0]):
        if y_train[j,i] == 1:
            samples[i,:] = x_train[j,:]
            samp_lbls[i,:] = y_train[j,:]
            break


# In[9]:


f, axs = plt.subplots(1,10,figsize=(20,20))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow((samples[i,:]).reshape([28,28]))
    plt.title(np.where(samp_lbls[i,:]==1)[0][0].astype(int))
plt.suptitle('One sample from each class', y = 0.58, fontsize=17)


# In[89]:


epsilon = 1e-2
# adjust layer at which weights/bias are observed
save_layer = 2 # choose 0 or 2
# adjust which weight is observed
save_ind = 40
save_unit = 5
# adjust which bias is observed
save_b = 2


# In[90]:


my_model = nn.Neuralnetwork(config)


# In[91]:


# calculate loss for weight adjusted by + or - epsilon

my_model_approx = copy.deepcopy(my_model)
my_model_approx.layers[save_layer].w[save_ind, save_unit]+=epsilon
out1, _ = my_model_approx.forward(samples, samp_lbls)
error1 = my_model_approx.loss(out1, samp_lbls)
print(error1)

my_model_approx.layers[save_layer].w[save_ind, save_unit]-=2*epsilon
out2, _ = my_model_approx.forward(samples, samp_lbls)
error2 = my_model_approx.loss(out2, samp_lbls)
print(error2)

# approximate gradient
grad_approx_w = (error1-error2)/(2*epsilon)
print(grad_approx_w)


# In[92]:


# run forward pass through model and backprop for weight update
_, _ = my_model.forward(samples, samp_lbls)
my_model.backward()
my_dw = my_model.layers[save_layer].d_w[save_ind, save_unit]
print(my_dw)


# In[93]:


print("dE/dw diff = ", grad_approx_w+my_dw)


# In[85]:


# need to reinitialize model so only bias is updated
my_model = nn.Neuralnetwork(config)


# In[86]:


# similar to weights, adjust bias by + or - epsilon and calculate loss
my_model_approx = copy.deepcopy(my_model)
my_model_approx.layers[save_layer].b[0][save_b]+=epsilon
out1, _ = my_model_approx.forward(samples, samp_lbls)
error1 = my_model_approx.loss(out1, samp_lbls)
print(error1)

my_model_approx.layers[save_layer].b[0][save_b]-=2*epsilon
out2, _ = my_model_approx.forward(samples, samp_lbls)
error2 = my_model_approx.loss(out2, samp_lbls)
print(error2)

# approximate gradient
grad_approx = (error1-error2)/(2*epsilon)
print(grad_approx)


# In[87]:


# run forward pass and backprop to get bias update
_, _ = my_model.forward(samples, samp_lbls)
my_model.backward()
my_db = my_model.layers[save_layer].d_b[save_b]
print(my_db)


# In[88]:


print("dE/db diff = ", grad_approx+my_db)


# ## Mini-Batch SGD

# In[19]:


model  = nn.Neuralnetwork(config)

train_losses, train_accs, val_losses, val_accs = nn.train(model, x_train, y_train, x_valid, y_valid, config)

test_acc = nn.test(model, x_test, y_test)


# In[20]:


f, axs = plt.subplots(2,2,figsize=(10,5))
plt.subplots_adjust(wspace=0.3,hspace=0.5)
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


# ## Regularization

# In[21]:


config['epochs'] = 110
config['L2_penalty'] = 0.001
model1  = nn.Neuralnetwork(config)

train_losses1, train_accs1, val_losses1, val_accs1 = nn.train(model1, x_train, y_train, x_valid, y_valid, config)

test_acc1 = nn.test(model1, x_test, y_test)

config['L2_penalty'] = 0.0001
model2  = nn.Neuralnetwork(config)

train_losses2, train_accs2, val_losses2, val_accs2 = nn.train(model2, x_train, y_train, x_valid, y_valid, config)

test_acc2 = nn.test(model2, x_test, y_test)


# In[22]:


f, axs = plt.subplots(2,2,figsize=(10,5))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2,2,1)
line1,=plt.plot(np.array(range(len(train_losses1))), train_losses1)
line2,=plt.plot(np.array(range(len(train_losses2))), train_losses2)
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.title('Training Loss per Epoch')
plt.legend([line1, line2], ['lambda=0.001', 'lambda=0.0001'])
    
plt.subplot(2,2,2)
line1,=plt.plot(np.array(range(len(train_accs1))), train_accs1)
line2,=plt.plot(np.array(range(len(train_accs2))), train_accs2)
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.title('Training Accuracy per Epoch')
plt.legend([line1, line2], ['lambda=0.001', 'lambda=0.0001'])
    
plt.subplot(2,2,3)
line1,=plt.plot(np.array(range(len(val_losses1))), val_losses1)
line2,=plt.plot(np.array(range(len(val_losses2))), val_losses2)
plt.xlabel('epochs')
plt.ylabel('val loss')
plt.title('Validation Loss per Epoch')
plt.legend([line1, line2], ['lambda=0.001', 'lambda=0.0001'])
    
plt.subplot(2,2,4)
line1,=plt.plot(np.array(range(len(val_accs1))), val_accs1)
line2,=plt.plot(np.array(range(len(val_accs2))), val_accs2)
plt.xlabel('epochs')
plt.ylabel('val accuracy')
plt.title('Validation Accuracy per Epoch')
plt.legend([line1, line2], ['lambda=0.001', 'lambda=0.0001'])
    
plt.suptitle('Regularization Training Results')


# In[94]:


print('Test accuracy for lambda = 0.001: ', test_acc1)
print('Test accuracy for lambda = 0.0001: ', test_acc2)


# ## Comparing Activations

# In[23]:


config['L2_penalty'] = 0
config['epochs'] = 100
# train model with tanh activation
config['activation'] = 'tanh'
model1  = nn.Neuralnetwork(config)

train_losses1, train_accs1, val_losses1, val_accs1 = nn.train(model1, x_train, y_train, x_valid, y_valid, config)

test_acc1 = nn.test(model1, x_test, y_test)

# train model with sigmoid activation
config['activation'] = 'sigmoid'
model2  = nn.Neuralnetwork(config)

train_losses2, train_accs2, val_losses2, val_accs2 = nn.train(model2, x_train, y_train, x_valid, y_valid, config)

test_acc2 = nn.test(model2, x_test, y_test)

# train model with relu activation
config['activation'] = 'ReLU'
model3  = nn.Neuralnetwork(config)

train_losses3, train_accs3, val_losses3, val_accs3 = nn.train(model3, x_train, y_train, x_valid, y_valid, config)

test_acc3 = nn.test(model3, x_test, y_test)


# In[24]:


f, axs = plt.subplots(2,2,figsize=(10,5))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2,2,1)
line1,=plt.plot(np.array(range(len(train_losses1))), train_losses1)
line2,=plt.plot(np.array(range(len(train_losses2))), train_losses2)
line3,=plt.plot(np.array(range(len(train_losses3))), train_losses3)
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.title('Training Loss per Epoch')
plt.legend([line1, line2, line3], ['tanh', 'sigmoid', 'ReLU'])
    
plt.subplot(2,2,2)
line1,=plt.plot(np.array(range(len(train_accs1))), train_accs1)
line2,=plt.plot(np.array(range(len(train_accs2))), train_accs2)
line3,=plt.plot(np.array(range(len(train_accs3))), train_accs3)
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.title('Training Accuracy per Epoch')
plt.legend([line1, line2, line3], ['tanh', 'sigmoid', 'ReLU'])
    
plt.subplot(2,2,3)
line1,=plt.plot(np.array(range(len(val_losses1))), val_losses1)
line2,=plt.plot(np.array(range(len(val_losses2))), val_losses2)
line3,=plt.plot(np.array(range(len(val_losses3))), val_losses3)
plt.xlabel('epochs')
plt.ylabel('val loss')
plt.title('Validation Loss per Epoch')
plt.legend([line1, line2, line3], ['tanh', 'sigmoid', 'ReLU'])
    
plt.subplot(2,2,4)
line1,=plt.plot(np.array(range(len(val_accs1))), val_accs1)
line2,=plt.plot(np.array(range(len(val_accs2))), val_accs2)
line3,=plt.plot(np.array(range(len(val_accs3))), val_accs3)
plt.xlabel('epochs')
plt.ylabel('val accuracy')
plt.title('Validation Accuracy per Epoch')
plt.legend([line1, line2, line3], ['tanh', 'sigmoid', 'ReLU'])
    
plt.suptitle('Different Activation Training Results')


# In[25]:


print('Test accuracy for tanh: ', test_acc1)
print('Test accuracy for sigmoid: ', test_acc2)
print('Test accuracy for ReLU: ', test_acc3)


# ## Comparing Topology

# In[26]:


# train model with tanh activation
config['activation'] = 'tanh'


model1  = nn.Neuralnetwork(config)

train_losses1, train_accs1, val_losses1, val_accs1 = nn.train(model1, x_train, y_train, x_valid, y_valid, config)

test_acc1 = nn.test(model1, x_test, y_test)

# train model with half the hidden units
config['layer_specs'] = [784, 25, 10]
model2  = nn.Neuralnetwork(config)

train_losses2, train_accs2, val_losses2, val_accs2 = nn.train(model2, x_train, y_train, x_valid, y_valid, config)

test_acc2 = nn.test(model2, x_test, y_test)

# train model with double the hidden units
config['layer_specs'] = [784, 100, 10]
model3  = nn.Neuralnetwork(config)

train_losses3, train_accs3, val_losses3, val_accs3 = nn.train(model3, x_train, y_train, x_valid, y_valid, config)

test_acc3 = nn.test(model3, x_test, y_test)


# In[27]:


f, axs = plt.subplots(2,2,figsize=(10,5))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2,2,1)
line1,=plt.plot(np.array(range(len(train_losses1))), train_losses1)
line2,=plt.plot(np.array(range(len(train_losses2))), train_losses2)
line3,=plt.plot(np.array(range(len(train_losses3))), train_losses3)
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.title('Training Loss per Epoch')
plt.legend([line1, line2, line3], ['50', '25', '100'])
    
plt.subplot(2,2,2)
line1,=plt.plot(np.array(range(len(train_accs1))), train_accs1)
line2,=plt.plot(np.array(range(len(train_accs2))), train_accs2)
line3,=plt.plot(np.array(range(len(train_accs3))), train_accs3)
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.title('Training Accuracy per Epoch')
plt.legend([line1, line2, line3], ['50', '25', '100'])
    
plt.subplot(2,2,3)
line1,=plt.plot(np.array(range(len(val_losses1))), val_losses1)
line2,=plt.plot(np.array(range(len(val_losses2))), val_losses2)
line3,=plt.plot(np.array(range(len(val_losses3))), val_losses3)
plt.xlabel('epochs')
plt.ylabel('val loss')
plt.title('Validation Loss per Epoch')
plt.legend([line1, line2, line3], ['50', '25', '100'])
    
plt.subplot(2,2,4)
line1,=plt.plot(np.array(range(len(val_accs1))), val_accs1)
line2,=plt.plot(np.array(range(len(val_accs2))), val_accs2)
line3,=plt.plot(np.array(range(len(val_accs3))), val_accs3)
plt.xlabel('epochs')
plt.ylabel('val accuracy')
plt.title('Validation Accuracy per Epoch')
plt.legend([line1, line2, line3], ['50', '25', '100'])
    
plt.suptitle('Training Results for Different Number of Hidden Units')


# In[28]:


print('Test accuracy for 50 hidden units: ', test_acc1)
print('Test accuracy for 25 hidden units: ', test_acc2)
print('Test accuracy for 100 hidden units: ', test_acc3)


# In[29]:


# train model with one hidden layer
config['layer_specs'] = [784, 50, 10]
model1  = nn.Neuralnetwork(config)

train_losses1, train_accs1, val_losses1, val_accs1 = nn.train(model1, x_train, y_train, x_valid, y_valid, config)

test_acc1 = nn.test(model1, x_test, y_test)

# train model with two hidden layers
config['layer_specs'] = [784, 25, 25, 10]
model2  = nn.Neuralnetwork(config)

train_losses2, train_accs2, val_losses2, val_accs2 = nn.train(model2, x_train, y_train, x_valid, y_valid, config)

test_acc2 = nn.test(model2, x_test, y_test)


# In[30]:


f, axs = plt.subplots(2,2,figsize=(10,5))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2,2,1)
line1,=plt.plot(np.array(range(len(train_losses1))), train_losses1)
line2,=plt.plot(np.array(range(len(train_losses2))), train_losses2)
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.title('Training Loss per Epoch')
plt.legend([line1, line2], ['one layer', 'two layers'])
    
plt.subplot(2,2,2)
line1,=plt.plot(np.array(range(len(train_accs1))), train_accs1)
line2,=plt.plot(np.array(range(len(train_accs2))), train_accs2)
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.title('Training Accuracy per Epoch')
plt.legend([line1, line2], ['one layer', 'two layers'])
    
plt.subplot(2,2,3)
line1,=plt.plot(np.array(range(len(val_losses1))), val_losses1)
line2,=plt.plot(np.array(range(len(val_losses2))), val_losses2)
plt.xlabel('epochs')
plt.ylabel('val loss')
plt.title('Validation Loss per Epoch')
plt.legend([line1, line2], ['one layer', 'two layers'])
    
plt.subplot(2,2,4)
line1,=plt.plot(np.array(range(len(val_accs1))), val_accs1)
line2,=plt.plot(np.array(range(len(val_accs2))), val_accs2)
plt.xlabel('epochs')
plt.ylabel('val accuracy')
plt.title('Validation Accuracy per Epoch')
plt.legend([line1, line2], ['one layer', 'two layers'])
    
plt.suptitle('Training Results for Different Number of Hidden Layers')


# In[31]:


print('Test accuracy for one hidden layer: ', test_acc1)
print('Test accuracy for two hidden layers: ', test_acc2)


# In[ ]:




