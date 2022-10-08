import torch

# This is a 1D tensor
a = torch.tensor([2, 3, 2])
print('a is ', a)

# This is a 2D tensor
b = torch.tensor([[2, 1, 3, 45], [4, 7, 45, 7], [32, 4, 5, 8]])
print('b is ', b)

# Size of the tensors. Shape its an attribute and size is a method.
print('a shape: ', a.shape)
print('b shape: ', b.shape)
print('a size: ', a.size())
print('b size: ', b.size())

# Get the number of rows of b
print('b rows: ', b.shape[0])

# CAREFUL! we might float our things (float 32)
c = torch.FloatTensor([[4, 2, 5], [7, 2, 3], [7, 7, 2]])
# or we can do c=torch.tensor(...,dtype = torch.float)
print('c and its type', c, c.dtype)

# The same happens with doubles (float 64)
d = torch.DoubleTensor([[34, 1, 2], [7, 4, 4], [24, 65, 6]])
# or d = torch.tensor(...,dtype=torch.double)
print('d and its type', d, d.dtype)

print('mean of c', c.mean())
print('std of d', d.std())

# RESHAPING -------------------
# Note: in PyTorch is called view and if one of the dimensios is -1, its size can be inferred
print('reshaping b to (-1,1) ', b.view(-1, 1))
print('reshaping b to (12) ', b.view(12))
print('reshaping b to (-1,3) ', b.view(-1, 3))
print('reshaping b to (4,3) ', b.view(4, 3))

# Assign b to a new shape
b = b.view(1, -1)
print('Reshaped b to (1,-1) and its shape', b, b.shape)

# We can even reshape a 3d tensor
three_dim = torch.randn(2, 3, 4)  # CHANNELS, ROWS, COLUMNS nomenclature in pytorch
print('original 3D, ', three_dim)
print('reshaped 3D to (12,2) via (12,-1) ', three_dim.view(12, -1))

#   -   -   -   -   -   -   Part 2  -   -   -   -   -   -   -
# create a matrix with random numbers between 0 and 1
r = torch.rand(4,4)
print('\nrandom tensor ',r)

# create a matrix with N(0,1)
r2 = torch.randn(4,4)
print('random tensor, normally distributed ', r2)

# Five random integers in 1D array
in_array = torch.randint(high=10,low=6,size=(5,))
print('\nArray of integers between [6,10) ',in_array)

# Random integers in 2D array
in_array2 = torch.randint(6,10,(3,3))
print('Array of integers between [6,10), but 2D ',in_array2)

# Random integers in 3D array
in_array3 = torch.randint(6,10,(3,3,3))
print('Array of integers between [6,10), but 3D ',in_array3)

# Number of elements in the array
print('Elements in in_array ', torch.numel(in_array))
print('Elements in in_array2 ', torch.numel(in_array2))

# Construct a 3x3 matrix of zeros and a dtype of Long:
z = torch.zeros(size=(3,3),dtype=torch.long)
# or a 3x3 of ones
o = torch.ones(size=(3,3))
print('\nzeros ', z)
print('ones ', o)
print('default type of o ', o.dtype)

# Convert the dtype of a tensor
r2_like = torch.randn_like(r2,dtype=torch.double) # using like will get the SIZE of r2 and apply it to r2_like
print('\nGenerating new random with same size but different type',r2_like)

# Adding two tensors of the same size and dtype
add1 = r + r2
add2= torch.add(r,r2)
print('\nAdding with +',add1)
print('Adding with torch.add(·,·)',add2)

# In-place addition:  r2 = r2 + ..., r2+=...
r2.add_(r)
print('In place added r to r2 by adding _ to add', r2)

# Slicing
print('\nSlicing r2[:,1]',r2[:,1])
print('r2[:,:2]',r2[:,:2])
num_ten = r2[2,3]
print('Extracted item from r2 via slicing',num_ten)
print('Getting the value out of tensor(·)',num_ten.item())
