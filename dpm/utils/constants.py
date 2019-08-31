import torch
import math

euler_mascheroni = torch.tensor(0.57721566490153286060651209008240243104215933593992).float()
catalan = torch.tensor(0.915965594177219015054603514932384110774).float()
eps = torch.tensor(1e-10).float()
e = torch.tensor(math.e).float()
pi = torch.tensor(math.pi).float()
