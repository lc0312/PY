import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


x = torch.unsqueeze (torch.linspace(-10,10,500),dim=1)
y = torch.sin(x) + 0.15*x*torch.rand(x.size()) + 0.01*x*np.exp(0.1*x)
x,y = Variable(x), Variable(y)
x_cuda,y_cuda = x.cuda(), y.cuda()
x_np,y_np = x.numpy(),y.numpy()

class t_net (torch.nn.Module):
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

t_net_0 = t_net(in_layer=1, hid_layer_0=500, hid_layer_1=200, out_layer=1)
t_net_0.cuda()

t_net_1 = torch.nn.Sequential(
        torch.nn.Linear (1, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1),
    )
t_net_1.cuda()

optmizer_0 = torch.optim.Adam (t_net_0.parameters(),lr=1E-4)
optmizer_1 = torch.optim.Adam (t_net_1.parameters(), lr=1E-4)
loss_func = torch.nn.MSELoss()

for i in range (5000):

    prediction = t_net_0(x_cuda)
    loss = loss_func(prediction,y_cuda)
    optmizer_0.zero_grad ()
    loss.backward ()
    optmizer_0.step ()

for i in range (5000):

    prediction = t_net_1(x_cuda)
    loss = loss_func(prediction,y_cuda)
    optmizer_1.zero_grad()
    loss.backward ()
    optmizer_1.step ()

plt.plot (x_np,y_np,'o')
plt.plot(x_np,t_net_0(x_cuda).cpu().data.detach().numpy(),color='blue')
plt.plot(x_np,t_net_1(x_cuda).cpu().data.detach().numpy(),color='red')
plt.show()
