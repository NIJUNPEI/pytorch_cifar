import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='/home/nijunpei/pywork/cifar-100/data',train=True,
                                         download=True,transform=transform)
testset = torchvision.datasets.CIFAR100(root='/home/nijunpei/pywork/cifar-100/data',train=False,
                                        download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=15,shuffle=True,num_workers=2)
testloader = torch.utils.data.DataLoader(testset,batch_size=15,shuffle=False,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ResBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.adap_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(planes * 4, round(planes / 4))
        self.linear2 = nn.Linear(round(planes / 4), planes * 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out

class SEBlock(nn.Module):
    def __init__(self, planes, r=16):
        super(SEBlock, self).__init__()
        self.adap_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(planes*4, round(planes*4/r))
        self.linear2 = nn.Linear(round(planes*4/r), planes*4)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        se = x
        se = self.adap_pool(se)
        se = torch.squeeze(se)
        se = self.linear1(se)
        se = self.relu(se)
        se = self.linear2(se)
        se = self.sigmoid(se)

        se = torch.unsqueeze(torch.unsqueeze(se, 2), 3)
        x = x * se
        return x

class ResSEBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResSEBlock, self).__init__()
        self.res_b = ResBlock(inplanes, planes, stride)
        self.downsample = downsample
        self.se_b = SEBlock(planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.res_b(x)
        out = self.se_b(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResSENet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResSENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ressenet50(pretrained=False, **kwargs):
    model = ResSENet(ResSEBlock, [3, 4, 6, 3], **kwargs)
    return model

net = ressenet50()
print(net)
net.cuda()


criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

def train(epoch):
    net.train()
    for batch_index,(data, target) in enumerate(trainloader):
        #data.cuda()
        #target.cuda()
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_index % 50 == 0 :
            print('Train epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(trainloader.dataset),
                100. * batch_index / len(trainloader), loss.data[0]))
def test(epoch):
    net.eval()#把module设置为评估模式，只对Dropout和BatchNorm模块有影响
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = net(data)
        test_loss += criterion(output, target).data[0]#Variable.data
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testloader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

for epoch in range(1):
    train(epoch)
    test(epoch)