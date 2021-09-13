import torch
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from torch.autograd import Variable
import torch.nn.functional as F

bos = load_boston()
df = pd.DataFrame(bos.data)
df.columns = bos.feature_names
df['Price'] = bos.target

X = df.drop ('Price', axis=1).to_numpy()
Y = df['Price'].to_numpy()
X = torch.tensor (X, dtype=torch.float)
Y = torch.tensor (Y, dtype=torch.float).view(-1,1)
x,y = Variable(X), Variable(Y)
x_cuda,y_cuda = x.cuda(), y.cuda()
datasets = torch.utils.data.TensorDataset(x_cuda, y_cuda)

class NN (torch.nn.Module):
    def __init__(self,in_layer,hid_layer_0, hid_layer_1, out_layer):
        super().__init__()
        self.hidden_0 = torch.nn.Linear (in_layer, hid_layer_0)
        self.hidden_1 = torch.nn.Linear (hid_layer_0,hid_layer_1)
        self.predict = torch.nn.Linear (hid_layer_1, out_layer)

    def forward (self,x):
        x = F.leaky_relu (self.hidden_0(x_cuda))
        x = F.leaky_relu (self.hidden_1(x))
        x = self.predict(x)
        return x

NN_Model = NN (in_layer=13, hid_layer_0=13*8, hid_layer_1=13*4, out_layer=1)
NN_Model.cuda()
optmizer = torch.optim.Adam (NN_Model.parameters(),lr=1E-3)
loss_func = torch.nn.MSELoss()

for i in range (3000):
    prediction = NN_Model (x_cuda)
    loss = loss_func(prediction,y_cuda)
    optmizer.zero_grad ()
    loss.backward ()
    optmizer.step ()
   
print (NN_Model(x)[27])
print (y[27])
