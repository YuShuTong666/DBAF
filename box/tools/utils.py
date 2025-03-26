import torch
import torch.autograd
x = torch.randn(3, 3, requires_grad=True)
y = x[2][0] - x[1][0]
z = 2*x
f = 3*y+z
print(x.grad)
y.backward()
print(x.grad)