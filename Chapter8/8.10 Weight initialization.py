import torch
import torch.nn as nn
# Docs: torch.nn.init

# Define a tensor to initialize. We create a layer and access its weight
layer = nn.Linear(5,5)
weights = layer.weight.data
print('weights', weights)

# Uniform distribution between a and b as an inplace operation (_)
nn.init.uniform_(tensor=layer.weight.data,a=0,b=3)
print('Uniformly initialized weights ', layer.weight)
# By default, layers have require_grad=True

# Normal distribution as an inplace operation
# layer.weight.data = nn.init.normal(tensor=layer.weight.data,mean=0,std=1) # (DEPRECATED)
nn.init.normal_(tensor=layer.weight.data,mean=0,std=1) # WE SHOULD USE THIS INPLACE INPLEMENTATION
print('Normally initialized weights ', layer.weight)

# We could initialized them as constant values (useful when initializing biases)
nn.init.constant_(layer.bias.data,val=5)
print('Uniformly initialized bias ',layer.bias)
# or all to zeros
nn.init.zeros_(layer.bias.data)
print('Zeroly initialized bias ',layer.bias)

# Xavier init from uniform distribution, but the value of a of the U[-a,a] is calculated based on n_in and n_out
nn.init.xavier_uniform_(layer.weight.data)
print('Xavier-uniform initialized weights ',layer.weight.data)

# Xavier init from Normal distribution, but the value of a of the N[0,a] is calculated based on n_in and n_out
nn.init.xavier_normal_(layer.weight.data)
print('Xavier-normal initialized weights ',layer.weight.data)