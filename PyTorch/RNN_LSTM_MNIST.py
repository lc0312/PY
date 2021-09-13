import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import time
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np

train_data = torchvision.datasets.MNIST( 
    root='/media/l/SD/PY/PyTorch/mnist/', train=True,transform=torchvision.transforms.ToTensor(),
    download=0)
BATCH_SIZE = 32
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = dsets.MNIST(root='/media/l/SD/PY/PyTorch/mnist/', train=False, transform=transforms.ToTensor())
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000].cuda()/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels[:2000].cuda()

class RNN (nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.LSTM(
            input_size=28,
            hidden_size=28*3,
            num_layers=2,
            batch_first=True,
        )
        
        self.out = nn.Linear (28*3, 10)
    
    def forward(self,x):
        r_out, (h_n, h_c)=self.rnn(x, None) # None is first hiddenstate, which is none
        out = self.out(r_out[:,-1,:]) # batch, time step, input
        return out

rnn=RNN()
rnn.cuda()

optimizer = torch.optim.Adam (rnn.parameters(), lr=5E-4)
loss_func = nn.CrossEntropyLoss()

start=time.time()
for i in range (1):
    for step, (x,y) in enumerate(train_loader):
        
        b_x=Variable(x.view(-1,28,28)).cuda()
        b_y=Variable(y).cuda()
        output=rnn(b_x)
        loss=loss_func(output, b_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%200 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output,1)[1].data.cuda().squeeze()
            accuracy = sum(pred_y==test_y)/float(test_y.size(0))
            print ('loss', loss.data, accuracy)

end = time.time()
print (start-end)