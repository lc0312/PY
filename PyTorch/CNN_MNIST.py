import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import time
BATCH_SIZE = 32

train_data = torchvision.datasets.MNIST( 
    root='/media/l/SD/PY/PyTorch/mnist/', train=True,transform=torchvision.transforms.ToTensor(),
    download=0)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST (root='/media/l/SD/PY/PyTorch/mnist/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000].cuda()/255.
test_y = test_data.test_labels[:2000].cuda()

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d( # 1*28*28 pixel
                in_channels=1, # thickness of picture, for color rgb is 3, b&w is 1
                out_channels=16,
                kernel_size=5, # how many pixel per scan
                stride=1, # jump between per scan
                padding=2), # 0 surrouding picture, padding=(kernel_size-1)/2=0
            
            nn.ReLU(), #16*28*28
            nn.MaxPool2d(kernel_size=2), # pick max value of area 2*2 here
            nn.BatchNorm2d(16),
            )
        
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2), # 32*14*14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),) #32*7*7
        
        self.out=nn.Linear(32*7*7,10)
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1) #展平
        output=self.out(x)
        return output

cnn=CNN()
cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(),lr=1.5E-4)
loss_func = nn.CrossEntropyLoss()

start=time.time()
for i in range (1):
    for step, (x,y) in enumerate(train_loader):
        b_x=Variable(x).cuda()
        b_y=Variable(y).cuda()
        
        output=cnn(b_x)
        loss=loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step%200 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output,1)[1].cuda().data.squeeze()
            accuracy=(pred_y==test_y).sum().item()/float(test_y.size(0))
            print (accuracy)

end = time.time()
print (start-end)