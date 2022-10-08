import torch

#Autograd
# Remember, if requires_grad=True, the Tensor object keeps track of how it was created
x = torch.tensor([1.,2.,3.], requires_grad=True)
y = torch.tensor([4.,5.,6.], requires_grad=True)
# Since we set those to True, we can calculate gradients w.r.t. x and y
z = x+y
print('z from graph ', z)
# z knows that it was created as a result of addition of x and y, and that it asnt read from a file.
print('How was z created? (grad_fn) ',z.grad_fn)

#So , if we go further on this,
s = z.sum()
print('s from z ',s)
print('how was s created? ',s.grad_fn)


# If we backprop ond s, we can find the gradints with respect to x and y, but NOT w.r.t. z
s.backward()
print('ds/dx', x.grad)
print('ds/dy', y.grad)
print('ds/dz', z.grad)


# If we initially dont have the require,grad, we cant backrpop
x = torch.randn(2,2)
y = torch.randn(2,2)
print('Can we have gradients?',x.requires_grad,y.requires_grad)
z=x+y
print('How was z created?', z.grad_fn)
# So we can add another way to require the grad as
x.requires_grad_()
y.requires_grad_()

z =x+y
# so we can update z to make it part of the graph and remember the gradinents
z.requires_grad_()
print('Can we have ds/dz now?',z.requires_grad)

# What if we want to keep the value in another variable but FORGET the previous computation history? Then we use detach.
# The tensor doesn't remember how it was created. In other words, we have broken the Tensor away from its past history.
new_z = z.detach()
print('new forgotten z',new_z)
print('how was new_z created?',new_z.grad_fn)

# You can also stop autograd from tracking history on Tensors. This concept is useful when applying Transfer learning
print('Can we have ds/dx?', x.requires_grad)
print('Does this sum require grad?',(x+10).requires_grad)
with torch.no_grad():
    print('Does this sum require grad NOW? (by using no_grad())',(x+10).requires_grad)

# Let's walk through one last example
print('\nFINAL EXAMPLE')
x = torch.ones(2,2,requires_grad=True)
print('x',x)
y = x+2
print('y',y)
print('creation of y', y.grad_fn)
z = y*y*3
out = z.mean()
print('z and output',z,out)
out.backward()
print('dout/dx',x.grad)