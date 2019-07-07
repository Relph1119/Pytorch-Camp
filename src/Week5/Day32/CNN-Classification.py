import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

rootPath = '../data'


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)


# 实现残差块
class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)

        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x + out, True)


"""
测试一下一个 residual block 的输入和输出
"""
# 1.输入输出形状相同
test_net = residual_block(32, 32)
test_x = torch.zeros(1, 32, 96, 96)
print('input: {}'.format(test_x.shape))
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))

# 2.输入输出形状不同
test_net = residual_block(3, 32, False)
test_x = torch.zeros(1, 3, 96, 96)
print('input: {}'.format(test_x.shape))
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))


# 实现一个 ResNet，它就是 residual block 模块的堆叠
class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )

        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )

        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )

        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512),
            nn.AvgPool2d(3)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def data_tf(x):
    x = x.resize((96, 96), 2)  # 将图片放大到 96 x 96
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到
    x = x.transpose((2, 0, 1))  # 将 channel 放到第一维，只是 pytorch 要求的输入方式
    x = torch.from_numpy(x)
    return x


train_set = CIFAR10(rootPath, train=True, transform=data_tf, download=True)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10(rootPath, train=False, transform=data_tf, download=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

net = resnet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


# 数据处理
def data_tf(x):
    x = x.resize((96, 96), 2)  # 将图片放大到 96 x 96
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到
    x = x.transpose((2, 0, 1))  # 将 channel 放到第一维，只是 pytorch 要求的输入方式
    x = torch.from_numpy(x)
    return x


# 读取数据
train_set = CIFAR10(rootPath, train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10(rootPath, train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

net = resnet(3, 10)
# 优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# 损失函数
criterion = nn.CrossEntropyLoss()
epochs = 20
for epoch in range(epochs):

    net.train()
    loss_sigma = 0.0
    correct = 0.0
    total = 0.0

    for i, (data, label) in enumerate(train_data):
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
            net = net.cuda()
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, dim=1)
        total += label.size(0)
        correct += (predicted == label).squeeze().sum().cpu().numpy()
        loss_sigma += loss.item()
        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 2 == 0:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}" \
                  .format(epoch + 1, epochs, i + 1, len(train_data), loss_avg, correct / total))

        # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    if epoch % 1 == 0:
        loss_sigma = 0.0

        net.eval()
        for i, (data, label) in enumerate(test_data):
            # forward
            data = data.to("cuda")
            label = label.to("cuda")
            outputs = net(data)
            outputs.detach_()
            # 计算loss
            loss = criterion(outputs, label)
            loss_sigma += loss.item()
            # 统计
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).squeeze().sum().cpu().numpy()
            loss_sigma += loss.item()
            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            if i % 2 == 0:
                loss_avg = loss_sigma / 10
                loss_sigma = 0.0
                print("test: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}" \
                      .format(epoch + 1, epochs, i + 1, len(test_data), loss_avg, correct / total))

print('Finished Training')

torch.save(net, 'net.pkl' + str(epochs))  # 保存整个神经网络的结构和模型参数
torch.save(net.state_dict(), 'net_params_BN1_Nopool.pkl' + str(epoch))  # 只保存神经网络的模型参数
