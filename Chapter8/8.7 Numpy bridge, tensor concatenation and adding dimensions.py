import torch
import numpy as np

# Converting a torch tensor to a numpy array
a = torch.ones(5)
print('a before', a)
b = a.numpy()
print('converted a', b)
#in place addition
a.add_(1)
print('in-place addition a',a)
print('affected numpy version even though we did not touch it',b)

# Converting a numpy array to a torch tensor
a = np.ones(5)
print('\na before 2.0', a)
b = torch.from_numpy(a)
print('converted a 2.0', b)
np.add(a,1,out=a)
print('in-place addition a 2.0',a)
print('affected torch version even though we did not touch it',b)

print('\n BOTH TENSORS AND ARRAY ARE AND STAY LINKED!')

# Moving a tensor to the gpu
r2 = torch.randn(4,4)
r2 = r2.cuda()
print('\nMoved random tensor to cuda ', r2)

# Easy switching between CPU and GPU
CUDA = torch.cuda.is_available()
print('Is cuda available? ', CUDA)
if CUDA:
    b.cuda()
    print('Moved b to cuda ',b)

# You can also convert a list to a tensor
a = [2,3,45,5]
print('\nlist ', a)
to_list = torch.tensor(a)
print('New tensor from list',to_list,to_list.dtype)

data = [[2,3],[3,45],[2,6],[7,3]]
T = torch.tensor(data)
print('Another example from list ',T,T.dtype)

# Tensor concatenation
first_1 = torch.randn(2,5)
second_1 = torch.randn(3,5)
# concatenate along 0 dimension
con_1 = torch.cat([first_1,second_1])
print('\nConcatenated tensor along 0', con_1)
first_2 = torch.randn(2,3)
second_2 = torch.randn(2,9)
# concatenate along 1 dimension
con_2 = torch.cat([first_2,second_2],dim=1)
print('Concatenated tensor along 1', con_2)

# Adding dimensions to tensors
tensor_1 = torch.tensor([1,2,3,4])
# what happens is that a new first (0) dimension of tensor_1 will get added to tensor a
tensor_a = torch.unsqueeze(tensor_1,0)
print('\nBase tensor ',tensor_1)
print('Added new axis (dim) to tensor 1 at the beginning',tensor_a)
tensor_b = torch.unsqueeze(tensor_1,1)
print('Added new axis (dim) to tensor 1 at the end',tensor_b)
# we can do the prior for anyD tensor