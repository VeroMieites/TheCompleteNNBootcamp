# >conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# check for proper  instalation with imports

import torch
import torchvision
import numpy as np

# some prints to test if cuda works
print(f'Is CUDA ready? {torch.cuda.is_available()}')

print('What are we running on?', torch.cuda.current_device()) #cuda: 0 (gpu 0), cuda: 1(gpu 1)..., cpu: cpu
print('What is the name of my device?', torch.cuda.get_device_name(0)) # nombre gpu
print('How many devices do we have available?', torch.cuda.device_count()) # cuantas gpus (jajaxd)
print('What version of CUDA are we running?', torch.version.cuda) # versión de cuda

#first steps, all like numpy basically
x = torch.rand(3,3)
y = torch.empty(3,3)

# basic operations
z = x+y
z2 = torch.add(x,y)
z3 = y
z3.add_(x) # with _ operate over the tensor

# indexing just like numpy
print('\nindexing test ->', x[0,1])

# reshaping is done with "view"
y=x.view(9,-1)
print('Reshaping test ->', y)

# Watch out! All has to be on the same place to operate (GPU o CPU). You can move things between torch <-> numpy
x = torch.ones(2,3)
x_numpy = x.numpy()
print('\ntype x: ',type(x))
print('type x_np: ',type(x_numpy))

y = torch.from_numpy(x_numpy.astype(np.float32)) # OJO con el tipo de dato
print(f'\ny: ', y)

# Upload things to CUDA
print('Initial location of y: ', y.device) # ver donde está
y=y.to('cuda:0') # lo sube a la gpu
print('Location of y after upload: ', y.device)
y=y.to('cpu') # lo trae de vuelta
print('Location of y after download: ', y.device)