# coding: utf-8

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

sys.path.append("..")
from Code.utils.utils import MyDataset, validate, show_confMat
from tensorboardX import SummaryWriter
from datetime import datetime

train_txt_path = '../../Data/train.txt'
valid_txt_path = '../../Data/valid.txt'

classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_bs = 16
valid_bs = 16
lr_init = 0.001
max_epoch = 1

# log
result_dir = '../../Result/'

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(logdir=log_dir)

# ------------------------------------ step 1/5 : 加载数据------------------------------------

# 数据预处理设置
# 前三行设置均值，标准差，以及数据标准化
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
# 用 transforms.Compose 将所需要进行的处理给 compose 起来
trainTransform = transforms.Compose([
    transforms.Resize(32),
    # 随机裁剪
    # 在裁剪之前先对图片的上下左右均填充上 4 个 pixel，值为0，即变成一个 40*40 的数据，然后再随机进行 32*32 的裁剪。
    transforms.RandomCrop(32, padding=4),
    # Totensor
    transforms.ToTensor(),
    # 数据标准化(减均值，除以标准差)
    normTransform
])

validTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset实例
# 从 MyDataset 类中初始化 txt， txt 中有图片路径和标签
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

# 构建DataLoder
# 初始化 DataLoder 时，将 train_data 传入，从而使 DataLoder 拥有图片的路径
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)


# ------------------------------------ step 2/5 : 定义网络------------------------------------

# 1.必须继承 nn.Module 这个类，要让 PyTorch 知道这个类是一个 Module
class Net(nn.Module):
    # 2.在__init__(self)中设置好需要的“组件"
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 3.在 forward(self, x)中用定义好的“组件”进行组装
    def forward(self, x):
        # 第1-2行:x 经过 conv1，然后经过激活函数 relu，再经过 pool1 操作
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # 第3行:表示将 x 进行 reshape，为了后面做为全连接层的输入
        x = x.view(-1, 16 * 5 * 5)
        # 第4-5行:先经过全连接层 fc，然后经过 relu；
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 第6行:模型的最终输出是 fc3 输出
        x = self.fc3(x)
        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            # 判断各层属于什么类型，然后根据不同类型的层，设定不同的权值初始化方法
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


net = Net()  # 创建一个网络
net.initialize_weights()  # 初始化权值

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

criterion = nn.CrossEntropyLoss()  # 选择损失函数
optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)  # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------

for epoch in range(max_epoch):

    loss_sigma = 0.0  # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    # 在一个iteration进行时，才读取一个batch的图片数据enumerate()函数会返回可迭代数据的一个“元素”
    # 在这里data是一个batch的图片数据和标签， data是一个list
    for i, data in enumerate(train_loader):
        # if i == 30 : break
        # 获取图片和标签
        inputs, labels = data
        # 将图片数据转换成 Variable 类型，然后称为模型真正的输入
        inputs, labels = Variable(inputs), Variable(labels)

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()
        loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, correct / total))

            # 记录训练loss
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # 记录learning rate
            writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

    # 每个epoch，记录梯度，权值
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    if epoch % 2 == 0:
        loss_sigma = 0.0
        cls_num = len(classes_name)
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        net.eval()
        for i, data in enumerate(valid_loader):

            # 获取图片和标签
            images, labels = data
            images, labels = Variable(images), Variable(labels)

            # forward
            outputs = net(images)
            outputs.detach_()

            # 计算loss
            loss = criterion(outputs, labels)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)
            # labels = labels.data    # Variable --> tensor

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].numpy()
                pre_i = predicted[j].numpy()
                conf_mat[cate_i, pre_i] += 1.0

        print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
        # 记录Loss, accuracy
        writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(valid_loader)}, epoch)
        writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
print('Finished Training')

# ------------------------------------ step5: 保存模型 并且绘制混淆矩阵图 ------------------------------------
net_save_path = os.path.join(log_dir, 'net_params.pkl')
# 保存模型
torch.save(net.state_dict(), net_save_path)

conf_mat_train, train_acc = validate(net, train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(net, valid_loader, 'valid', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
