import torch
import numpy as np 
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) #x data(tensor), shape(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size()) #this give y a noise with x's size

#draw
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
