#python test.py
import torch
import torch.nn as nn


# x = torch.Tensor(5,3)
# print(x)
# y = torch.Tensor([[3,2],[1,1]])
# print(y)
# print(y.shape)

x = torch.tensor(data = [2.0,3.0],requires_grad=True)
y=x**2
z = 2*y+3

print(x)
print(y)
print(z)

target = torch.tensor([3.0,4.0])
loss=torch.sum(torch.abs(z-target))
loss.backward()

print(x.grad,y.grad,z.grad)