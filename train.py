# 模块导入
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# tensorboard
writer = SummaryWriter('./logs/')

# 定义设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 核显不存在cuda,所以0就是独显

# 训练参数
EPOCH = 30
BATCH_SIZE = 128
LR = 1E-3

# 下载数据集，并转换PIL图像为张量
train_file = datasets.MNIST(
    root='./dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_file = datasets.MNIST(
    root='./dataset/',
    train=False,
    transform=transforms.ToTensor()
)

# img, target = test_file[0]
# print(img.shape)  1*28*28  C*H*W

"""
# 数据可视化；训练数据可视化
train_data = train_file.data
train_targets = train_file.targets
# 获取学习数据以及监督数据
print(train_data.size())  # [60000，28, 28]
print(train_targets.size())  # [60000]

# 学习数据展示
plt.figure(figsize=(9, 9))  # 设置窗口大小 9*9英寸
for i in range(9):
    plt.subplot(3, 3, i+1)
    # 窗口分为3行3列，当前位置为第i+1个
    plt.title(train_targets[i].numpy())
    # 当前图像标签（tensor）强制转换array,并作为title
    plt.axis('off')
    # 设置关闭坐标轴（用不到）
    plt.imshow(train_data[i], cmap='gray')
    # imshow()函数将图像数据进行处理，设置显示色彩（camp参数）
plt.show()

# 测试数据展示
test_data = test_file.data
test_targets = test_file.targets
print(test_data.size())  # [10000, 28, 28]
print(test_targets.size())  # [10000]

plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title(test_targets[i].numpy())
    plt.axis('off')
    plt.imshow(test_data[i], cmap='gray')
plt.show()
"""

# DataLoader批量加载mnist数据
train_loader = DataLoader(
    dataset=train_file,
    batch_size=BATCH_SIZE,
    shuffle=True)
# 返回值是 idx, (data, targets) idx表示这是第几个batch
test_loader = DataLoader(
    dataset=test_file,
    batch_size=BATCH_SIZE,
    shuffle=False)


# 模型结构
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷基层定义
        self.conv = nn.Sequential(
            # [BATCH_SIZE, 1, 28, 28]
            nn.Conv2d(1, 32, 5, 1, 2),
            # [BATCH_SIZE, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [BATCH_SIZE, 32, 14, 14]
            nn.Conv2d(32, 64, 5, 1, 2),
            # [BATCH_SIZE, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [BATCH_SIZE, 64, 7, 7]
        )
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 变换形状
        y = self.fc(x)
        return y


# 实例化模型
model = CNN().to(device)  # 放置可用的设备上

# Adam optimizer. lambda1、2都默认，不添加正则化项（L2范数）
optim = torch.optim.Adam(model.parameters(), LR)

# 损失函数
loss_fn = nn.CrossEntropyLoss()  # 默认求损失函数的平均值


# 定义计算整个训练集或测试集loss及acc的函数
def calc(data_loader):
    loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        # 关闭backward时梯度，不再学习更新参数（进入测试验证阶段）
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)  # 数据送入可用设备
            output = model(data)  # 网络预测输出值
            loss += loss_fn(output, targets)
            correct += (output.argmax(1) == targets).sum()  # 预测正确的总数
            total += data.size(0)  # 送入验证的数据总量
    loss = loss.item() / len(data_loader)
    acc = correct.item() / total
    return loss, acc


# 训练过程显示函数
def show():
    # 训练过程初始时（epoch=0)，定义全局变量
    if epoch == 0:
        global model_saved_list
        global temp
        temp = 0
    # 输出训练过程中EPOCH和STEP信息
    header_list = [
        f'EPOCH: {epoch+1:0>{len(str(EPOCH))}}/{EPOCH}',
        f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}']
    header_show = ' || '.join(header_list)  # 以||为分隔符连接列表
    print(header_show, end=' ')  # 不换行
    # EPOCH: 01 / 10 || STEP: 469 / 469

    # 打印训练的LOSS和ACC信息
    train_set_loss, train_set_acc = calc(train_loader)  # 计算当前模型参数对训练集的loss和acc
    writer.add_scalar('loss', train_set_loss, epoch+1)
    writer.add_scalar('acc', train_set_acc, epoch+1)
    train_list = [
        f'TRAIN_SET_LOSS: {train_set_loss:.4f}',
        f'TRAIN_SET_ACC: {train_set_acc:.4f}']
    # 均保留四位小数
    train_show = ' || '.join(train_list)
    print(train_show, end=' ')

    # 打印测试的LOSS和ACC信息
    val_set_loss, val_set_acc = calc(test_loader)  # 计算当前模型参数对测试集的loss和acc
    writer.add_scalar('val_set_loss', val_set_loss, epoch+1)
    writer.add_scalar('val_set_acc', val_set_acc, epoch+1)
    test_list = [
        f'VAL_SET_LOSS: {val_set_loss:.4f}',
        f'VAL_SET_ACC: {val_set_acc:.4f}'
    ]
    test_show = ' '.join(test_list)
    print(test_show, end=' ')

    # 保存最佳模型(设置阈值temp)
    if val_set_acc > temp:
        model_saved_list = header_list + train_list + test_list
        torch.save(model.state_dict(), 'model.pt')  # 只保留模型参数信息
        temp = val_set_acc  # 更新最优阈值


# 训练模型
for epoch in range(EPOCH):  # 30次epoch
    start_time = time.time()  # 记录开始时间
    for step, (data, targets) in enumerate(train_loader):
        # 获取到当前送入batch编号以及学习监督数据

        # forward
        optim.zero_grad()  # 优化器初始化
        data = data.to(device)
        targets = targets.to(device)
        output = model(data)
        loss = loss_fn(output, targets)
        acc = (output.argmax(1) == targets).sum().item() / BATCH_SIZE  # 当前训练批次数据的精准度

        # backward
        loss.backward()
        optim.step()
        # print(f"当前处于{epoch + 1}次遍历训练集，第{step+1}个batch")
        print(
            f'EPOCH: {epoch+1:0>{len(str(EPOCH))}}/{EPOCH}',
            f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}',
            f'LOSS: {loss.item():.4f}',
            f'ACC: {acc:.4f}',
            end='\r')
    print(f"第{epoch+1}次遍历完成，模型验证数据：")
    show()
    end_time = time.time()  # 结束时间
    print(f'TOTAL-TIME: {round(end_time-start_time)}')  # 四舍五入计算每个epoch耗时

# 输出并保存最优模型信息
model_saved_show = ' '.join(model_saved_list)
print('| BEST-MODEL | '+model_saved_show)
with open('model.txt', 'a') as f:  # 末尾写入信息
    f.write(model_saved_show+'\n')


# tensorboard  ： tensorboard --logdir=./logs/ --port 9000
