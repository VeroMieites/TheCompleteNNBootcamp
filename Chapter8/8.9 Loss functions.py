import torch
import torch.nn as nn # to use the loss functions
import numpy as np # In this script we will do this also from scratch to learn and understand them


# We want the loss between a prediction and an actual labels
predictions = torch.randn(4,5) # inputs are 4 and outputs are 5

# DOCUMENTATION! torch.nn
print('--> MSE')
# Start with MSE. In the docs, they tell us that the labels (ofc, not random), have to be for the MSE (N,*), where *
# means any dimension, and N has to be the same shape as the input (N,*)
label = torch.randn(4,5)
mse = nn.MSELoss(reduction='none')
loss = mse(input=predictions,target=label)
print('loss is ',loss)
# Usually, we want to specify the mean MSE of all outputs, so we get a single value as the loss
mse2 = nn.MSELoss(reduction='mean') # we can also sum them
loss2 = mse2(input=predictions,target=label)
print('loss with reduction is ',loss2)

# from scratch
loss3 = ((predictions-label)**2).mean()
print('manual loss from scratch is ',loss3)

print('\n--> BCE')
# for the BCE we need the outputs between zero and one
label = torch.zeros(4,5)
# we add a some random integers number from U[0,2) as an in place operation
label.random_(0,2)
print('New labels ',label)
sigmoid = nn.Sigmoid() # used to rescale the outputs as inputs to BCE
bce = nn.BCELoss(reduction='mean')
loss = bce(sigmoid(predictions), label)
print('loss ',loss)

print('\n--> BCE with included sigmoid (bce with logits loss)')
bces = nn.BCEWithLogitsLoss(reduction='mean')
loss = bces(predictions,label)
print('loss ',loss)

# Manual vesion of BCE and BCES
x = predictions.numpy()
y = label.numpy()

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = sigmoid(x)

# 2dim -> 2 for loops
loss_values = []
for i in range(x.shape[0]):
    # we want a new list for every batch
    batch_loss = []
    for j in range(x.shape[1]):
        batch_loss.append(-np.log(x[i,j]) if y[i,j]==1 else -np.log(1-x[i,j]))
    #     if y[i,j] == 1:
    #         loss = -np.log(x[i,j])
    #     elif y[i,j] == 0:
    #         loss = -np.log(1-x[i,j])
    #     batch_loss.append(loss)
    loss_values.append(batch_loss)

loss = np.mean(loss_values)
print('Manual BCE loss ',loss)


