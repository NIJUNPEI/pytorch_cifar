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

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, reduction=1):
        super(BasicBlock, self).__init__()
        self.expansion = 1 / float(reduction)
        self.in_planes = in_planes
        self.mid_planes = mid_planes = int(self.expansion * planes)
        self.out_planes = planes

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h*int_w*9*self.mid_planes*self.in_planes + out_h*out_w*9*self.mid_planes*self.out_planes
        if len(self.shortcut) > 0:
            flops += self.in_planes*self.out_planes*out_h*out_w
        return flops

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = out.size()
        out = self.bn2(self.conv2(out))
        self.out_nchw = out.size()
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, reduction=1, num_classes=100):
        super(ResNet, self).__init__()
        self.reduction = float(reduction) ** 0.5
        self.num_classes = num_classes
        self.in_planes = int(16 / self.reduction)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(32 / self.reduction), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(64 / self.reduction), num_blocks[2], stride=2)
        self.linear = nn.Linear(int(64 / self.reduction), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        planes = int(planes)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (out_h, out_w) = self.int_nchw, self.out_hw
        flops = 0
        for mod in (self.layer1, self.layer2, self.layer3):
            for layer in mod:
                flops += layer.flops()
        return int_h*int_w*9*self.in_planes + out_w*self.num_classes + flops

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = out.size()
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        self.out_hw = out.size()
        out = self.linear(out)
        return out


def ResNetWrapper(num_blocks, reduction=1, reduction_mode='net', num_classes=10):
    if reduction_mode == 'block':
        block = lambda in_planes, planes, stride: \
            BasicBlock(in_planes, planes, stride, reduction=reduction)
        return ResNet(block, num_blocks, num_classes=num_classes)
    return ResNet(BasicBlock, num_blocks, num_classes=num_classes, reduction=reduction)



def ResNet56(reduction=1, reduction_mode='net', num_classes=100):
    return ResNetWrapper([9, 9, 9], reduction, reduction_mode, num_classes)


net = ResNet56()
print(net)
net.cuda()


criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

def train(epoch):
    net.train()
    for batch_index,(data, target) in enumerate(trainloader):
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

for epoch in range(15):
    train(epoch)
    test(epoch)